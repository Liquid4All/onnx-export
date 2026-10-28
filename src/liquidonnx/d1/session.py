"""
d1 on onnxruntime: the checkpoint's own host code, which d1-export copies into the folder, with the
ONNX graphs in place of its PyTorch forward.

The host code (about 2.5k lines of the checkpoint's Python) is d1's API and everything around the
forward: question validation, prompts, set packing (attention.pack_set), answer slots, pointer
reads, readouts, calibration and typed answers (runner.SlotReader, api.SystemOneApi). Its
SlotReader runs under any engine that gives `_batches` and `_slot_logits`; OrtSystemOne's:

    _batches      one sequence a forward (no right padding)
    _slot_logits  the vision and audio encoders over the row's processor outputs, then the decoder
                  with question_ids from the packed row: log-softmax of the slot logits, and
                  pointer.read_logits over hidden_states for the read slots, as slots.slot_logits
                  does over the PyTorch model's last hidden state

Everything else (run, the batched readouts) is the checkpoint's SystemOne's. Pictures and audio go
through the checkpoint's processor (trust_remote_code on the export folder).
"""

import hashlib
import importlib
import json
import logging
import pathlib
import sys
import types

import numpy as np

from liquidonnx.d1.config import load_config
from liquidonnx.lfm3.session import PREFERRED, default_ep, load_session, load_tokenizer
from liquidonnx.lfm3_omni.session import BUNDLES as OMNI_BUNDLES
from liquidonnx.lfm3_vl.session import BUNDLES as VL_BUNDLES
from liquidonnx.lfm3_vl.session import run_vision

logger = logging.getLogger(__name__)

# precision -> graph role -> file: LFM3-VL's decoder and vision encoder names, LFM3-Omni's audio
# encoder (d1-Omni only)
BUNDLES = {
    precision: {
        "decoder": decoder,
        "vision_encoder": vision,
        "audio_encoder": OMNI_BUNDLES[precision]["audio_encoder"],
    }
    for precision, (decoder, vision) in VL_BUNDLES.items()
}
HOST_MODULES = ("api", "calibration", "lfm3", "pointer", "prompt", "runner")


def roles(model_dir: str | pathlib.Path) -> tuple[str, ...]:
    """The graphs a bundle of the folder runs."""
    audio = load_config(model_dir).audio is not None
    return ("decoder", "vision_encoder") + ("audio_encoder",) * audio


def available_precisions(model_dir: str | pathlib.Path, ep: str = "cpu") -> list[str]:
    """Bundles whose files the export folder has, in PREFERRED order for ep."""
    onnx_dir = pathlib.Path(model_dir) / "onnx"
    needed = roles(model_dir)
    return [
        precision
        for precision in PREFERRED[ep]
        if precision in BUNDLES
        and all((onnx_dir / BUNDLES[precision][role]).exists() for role in needed)
    ]


def host_code(model_dir: str | pathlib.Path) -> types.SimpleNamespace:
    """HOST_MODULES of the folder's copy of the checkpoint code, imported as one package (its
    modules import each other relatively), once a folder a process. lfm3 registers the LFM3
    families' answer lead (prompt.LEADS).

    attention.py registers its attention function with transformers as "d1" on import, which
    this runtime never calls. A PyTorch d1 model loaded in the process keeps its own: the
    function checks for its own copy's mask class."""
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    folder = pathlib.Path(model_dir).resolve()
    # The folder stays as exported (it is what gets uploaded): no __pycache__ beside the code. Its
    # modules import more of it lazily, so this holds for the process.
    sys.dont_write_bytecode = True
    package = f"liquidonnx_d1_host_{hashlib.sha1(str(folder).encode()).hexdigest()[:12]}"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        sys.modules[package] = module
    live = ALL_ATTENTION_FUNCTIONS.get("d1")
    modules = {name: importlib.import_module(f"{package}.{name}") for name in HOST_MODULES}
    if live is not None:
        AttentionInterface.register("d1", live)
    return types.SimpleNamespace(**modules)


def load(
    model_dir: str | pathlib.Path,
    precision: str | None = None,
    ep: str | None = None,
    tf32: bool = True,
    set_size: int | None = None,
):
    """The folder's OrtSystemOne: system_one(state, questions, media) and system_one_batch as the
    checkpoint's model has them. precision: a BUNDLES key, by default the first exported one of
    PREFERRED for ep (CUDA when available). set_size: questions packed after one state (the
    checkpoint's d1_set_size by default)."""
    return engine_class(host_code(model_dir))(model_dir, precision, ep, tf32, set_size)


def engine_class(host: types.SimpleNamespace) -> type:
    """OrtSystemOne over the host code's SlotReader and SystemOneApi."""
    import torch

    runner = host.runner

    class OrtSystemOne(runner.SlotReader, host.api.SystemOneApi):
        run = runner.SystemOne.run
        _readouts = runner.SystemOne._readouts

        def __init__(self, model_dir, precision, ep, tf32, set_size):
            """What SystemOne.__init__ sets, from the export folder: tokenizer, answer lead,
            calibration, prompt styles and the config's d1_* settings (_serve)."""
            self.model_dir = pathlib.Path(model_dir)
            raw = json.loads((self.model_dir / "config.json").read_text())
            self.config = load_config(self.model_dir)
            self.tokenizer = load_tokenizer(self.model_dir)
            bos = getattr(self.tokenizer, "bos_token", None)
            self.bos = bos if isinstance(bos, str) else ""
            self.lead = host.prompt.default_lead(raw["model_type"], self.tokenizer)
            calibration = self.model_dir / "calibration.json"
            self.calibration = (
                host.calibration.Calibration.load(calibration) if calibration.exists() else None
            )
            self.state_style = host.prompt.DEFAULT_STATE_STYLE
            self.system, self.option_style = host.prompt.DEFAULT_SYSTEM, "desc"
            self._serve(types.SimpleNamespace(**raw), set_size=set_size)
            self.processor, self.revision = None, None
            self.load_sessions(precision, ep or default_ep(), tf32)

        def load_sessions(self, precision: str | None, ep: str, tf32: bool):
            self.ep = ep
            self.precision = precision or next(iter(available_precisions(self.model_dir, ep)), None)
            if self.precision is None:
                raise FileNotFoundError(f"no exported bundle in {self.model_dir / 'onnx'}")
            onnx_dir, files = self.model_dir / "onnx", BUNDLES[self.precision]
            self.sessions = {
                role: load_session(onnx_dir / files[role], ep=ep, tf32=tf32)
                for role in roles(self.model_dir)
            }

        def _load_processor(self):
            from transformers import AutoConfig

            # The checkpoint's config module (native_lfm3*.py) registers its processor classes
            # with transformers; without it AutoProcessor gives the bare tokenizer.
            AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=True)
            return runner.auto_processor(str(self.model_dir))

        def _batches(self, seqs: list[tuple]) -> list[list[int]]:
            # shortcut: one row a forward; the graph's GroupQueryAttention mask assumes no
            # padding (LFM3Builder), so batching right-padded rows needs that validated first
            return [[j] for j in range(len(seqs))]

        def features(self, ids: list[int], inputs: dict) -> dict[str, np.ndarray]:
            """The decoder's feature inputs of one row's processor outputs, each checked against
            the row's placeholders."""
            H, merge = self.config.language.hidden_size, self.config.vision.merge_size
            out = dict.fromkeys(["image_features", "video_features"])
            tokens = {
                "image_features": self.config.tokens.image,
                "video_features": self.config.tokens.video,
            }
            for pixels, grid, name in (
                ("pixel_values", "image_grid_thw", "image_features"),
                ("pixel_values_videos", "video_grid_thw", "video_features"),
            ):
                if pixels in inputs:
                    out[name] = run_vision(
                        self.sessions["vision_encoder"],
                        inputs[pixels].float().numpy(),
                        inputs[grid].numpy(),
                        merge,
                    )
            if self.config.audio is not None:
                tokens["audio_features"] = self.config.audio.tokens.audio
                out["audio_features"] = None
                if "input_features" in inputs:
                    mels = inputs["input_features"].float().numpy()
                    lengths = inputs["feature_attention_mask"].sum(-1).tolist()
                    out["audio_features"] = np.concatenate(
                        [
                            self.sessions["audio_encoder"].run(
                                ["audio_features"], {"input_features": mel[None, :, :n]}
                            )[0]
                            for mel, n in zip(mels, lengths, strict=True)
                        ]
                    )
            for name, token in tokens.items():
                rows = out[name] if out[name] is not None else np.zeros((0, H), np.float32)
                if ids.count(token) != len(rows):
                    raise ValueError(f"{ids.count(token)} placeholders, {len(rows)} {name} rows")
                out[name] = rows.astype(np.float32)
            return out

        def _slot_logits(self, seqs: list[tuple], batch: list[int], slots: list[tuple], pointer):
            """The row's code slots' log-probabilities over the vocabulary, and its read slots'
            position logits (pointer.read_logits), as SystemOne._slot_logits gives them."""
            [b] = batch
            _, _, ids, _, question, _, inputs = seqs[b]
            n = len(ids)
            positions = [s[3] for s in slots]
            feed = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, n), dtype=np.int64),
                "position_ids": np.arange(n, dtype=np.int64)[None],
                "question_ids": np.array([question], dtype=np.int64),
                "slot_positions": np.array(positions, dtype=np.int64),
                **self.features(ids, inputs),
            }
            logits, hidden = self.sessions["decoder"].run(["logits", "hidden_states"], feed)
            codes = torch.from_numpy(logits)
            reads = None
            if pointer is not None:
                codes = codes[pointer["code_slots"]]
                hidden = torch.from_numpy(hidden)
                states = hidden[pointer["rows"], : pointer["width"]]
                reads = host.pointer.read_logits(hidden[0, positions], states, pointer)
            return codes - torch.logsumexp(codes, dim=-1, keepdim=True), reads

    return OrtSystemOne
