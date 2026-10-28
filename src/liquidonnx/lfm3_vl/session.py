"""
LFM3-VL generation on onnxruntime.

Images go through transformers' Qwen2VLImageProcessor configured from the export's
preprocessor_config.json (the LFM3-VL processor is that class with LFM3 defaults; LFM3-VL-2B
keeps its settings under image_processor in processor_config.json), one vision encoder run per
patch grid, and the decoder's cached greedy loop. LFM3-VL-2B's chat template writes
<|vision_start|><|image|><|vision_end|> per image, and a template may write a bare <|image|>;
either way each <|image|> expands to grid_h*grid_w/merge² tokens, which the decoder replaces with
the image's features.
"""

import json
import logging
import pathlib
import time

import numpy as np

from liquidonnx.lfm3.session import (
    PREFERRED,
    CachedDecoder,
    default_ep,
    kv_feeds_attention,
    load_session,
    load_tokenizer,
    log_speed,
)
from liquidonnx.lfm3_vl.config import MAX_GRID_SIDE, parse_config
from liquidonnx.session import initialize_cache

logger = logging.getLogger(__name__)

# precision -> (decoder, vision encoder). By answer-token KL from the fp32 reference, int8 vision
# alone costs more than the q8 decoders' own error, so they keep fp16 vision; next to q4's int4
# decoder it doesn't show.
BUNDLES = {
    "fp32": ("decoder.onnx", "vision_encoder.onnx"),
    "fp16": ("decoder_fp16.onnx", "vision_encoder_fp16.onnx"),
    "q8": ("decoder_q8.onnx", "vision_encoder_fp16.onnx"),
    "q4": ("decoder_q4.onnx", "vision_encoder_q8.onnx"),
    # fp16 activations for CUDA, whose GroupQueryAttention has no float32 kernel
    "q8f16": ("decoder_q8f16.onnx", "vision_encoder_fp16.onnx"),
    "q4f16": ("decoder_q4f16.onnx", "vision_encoder_fp16.onnx"),
}


def available_precisions(model_dir: pathlib.Path, ep: str = "cpu") -> list[str]:
    """Bundles whose files the export folder has, in PREFERRED order for ep."""
    onnx_dir = pathlib.Path(model_dir) / "onnx"
    return [
        precision
        for precision in PREFERRED[ep]
        if precision in BUNDLES and all((onnx_dir / name).exists() for name in BUNDLES[precision])
    ]


def image_processor(model_dir: pathlib.Path):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    model_dir = pathlib.Path(model_dir)
    path = model_dir / "preprocessor_config.json"
    if path.exists():
        config = json.loads(path.read_text())
    else:
        config = json.loads((model_dir / "processor_config.json").read_text())["image_processor"]
    for key in ("image_processor_type", "auto_map"):
        config.pop(key, None)
    return Qwen2VLImageProcessor(**config)


def preprocess_images(processor, images, max_image_tokens: int | None = None):
    """pixel_values [patches, patch_dim] (block-major rows per image) and grids [n, 3] (t, h, w)."""
    kwargs = {}
    if max_image_tokens:
        pixels_per_token = (processor.patch_size * processor.merge_size) ** 2
        kwargs["size"] = {
            "shortest_edge": processor.size["shortest_edge"],
            "longest_edge": max_image_tokens * pixels_per_token,
        }
    out = processor(images=images, return_tensors="np", **kwargs)
    return out["pixel_values"].astype(np.float32), out["image_grid_thw"].astype(np.int64)


def expand_image_tokens(
    ids: list[int], grids: np.ndarray, image_token: int, merge: int
) -> list[int]:
    """Each image placeholder repeated once per image token of its grid."""
    counts = [int(np.prod(grid)) // merge**2 for grid in grids]
    if ids.count(image_token) != len(counts):
        raise ValueError(f"prompt has {ids.count(image_token)} images, got {len(counts)}")
    expanded, image = [], 0
    for token in ids:
        if token == image_token:
            expanded.extend([image_token] * counts[image])
            image += 1
        else:
            expanded.append(token)
    return expanded


def run_vision(session, pixel_values: np.ndarray, grids: np.ndarray, merge: int) -> np.ndarray:
    """Features of every grid, one run per temporal patch, in prompt order [tokens, hidden]."""
    features, offset = [], 0
    patch_dim = pixel_values.shape[-1]
    for t, h, w in grids.tolist():
        if h % merge or w % merge or max(h, w) > MAX_GRID_SIDE:
            raise ValueError(
                f"patch grid {h}x{w}: sides must be multiples of {merge} and at most {MAX_GRID_SIDE}"
            )
        for _ in range(t):
            grid = pixel_values[offset : offset + h * w]
            offset += h * w
            blocks = grid.reshape(h // merge, w // merge, merge, merge, patch_dim)
            features.append(session.run(None, {"pixel_values": blocks})[0])
    if offset != len(pixel_values):
        raise ValueError(f"grids cover {offset} patch rows, pixel_values has {len(pixel_values)}")
    return np.concatenate(features)


class ONNXVLModel:
    def __init__(
        self,
        model_dir: str,
        precision: str | None = None,
        force_cpu: bool = False,
        tf32: bool = True,
    ):
        """precision: a BUNDLES key; by default the first exported one of PREFERRED for the
        execution provider the sessions load on, as lfm3-vl-infer picks (available_precisions).
        tf32: let the CUDA EP run fp32 matmuls and convolutions in TF32 (faster, ~1e-3 off)."""
        self.model_dir = pathlib.Path(model_dir)
        self.requested_precision = precision
        self.precision = precision
        self.force_cpu = force_cpu
        self.tf32 = tf32
        self.config = json.loads((self.model_dir / "config.json").read_text())
        language, vision, tokens = parse_config(self.config)
        self.merge = vision.merge_size
        self.image_token = tokens.image
        self.video_token = tokens.video
        self.eos = tokens.eos
        self.hidden = language.hidden_size

    def load(self):
        self.load_sessions()
        self.load_preprocessing()

    def load_preprocessing(self):
        self.tokenizer = load_tokenizer(self.model_dir)
        self.processor = image_processor(self.model_dir)

    def load_sessions(self, ep: str | None = None):
        ep = ep or default_ep(self.force_cpu)
        self.precision = self.requested_precision or next(
            iter(available_precisions(self.model_dir, ep)), None
        )
        if self.precision is None:
            raise FileNotFoundError(f"no exported bundle in {self.model_dir / 'onnx'}")
        decoder_file, vision_file = BUNDLES[self.precision]
        onnx_dir = self.model_dir / "onnx"
        self.vision = load_session(onnx_dir / vision_file, ep=ep, tf32=self.tf32)
        self.decoder = load_session(onnx_dir / decoder_file, ep=ep, tf32=self.tf32)
        self.share_kv = kv_feeds_attention(onnx_dir / decoder_file)

    def encode_images(
        self, images, max_image_tokens: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Image features [tokens, hidden] and grids for PIL images."""
        pixel_values, grids = preprocess_images(self.processor, images, max_image_tokens)
        for grid in grids.tolist():
            logger.info(f"Image grid {grid}: {int(np.prod(grid)) // self.merge**2} tokens")
        return run_vision(self.vision, pixel_values, grids, self.merge), grids

    def decoder_feed(
        self, ids: list[int], image_features, past: int, logits_to_keep: int, cache
    ) -> dict:
        return {
            "input_ids": np.array([ids], dtype=np.int64),
            "image_features": image_features.astype(np.float32),
            "video_features": np.zeros((0, self.hidden), dtype=np.float32),
            "attention_mask": np.ones((1, past + len(ids)), dtype=np.int64),
            "position_ids": np.arange(past, past + len(ids), dtype=np.int64)[None, :],
            "logits_to_keep": np.array([logits_to_keep], dtype=np.int64),
            **cache,
        }

    def check_features(self, ids: list[int], image_features: np.ndarray):
        if ids.count(self.image_token) != len(image_features):
            raise ValueError(
                f"{ids.count(self.image_token)} image tokens, {len(image_features)} feature rows"
            )

    def prefill_logits(
        self, ids: list[int], image_features=None, logits_to_keep: int = 1
    ) -> np.ndarray:
        """Logits [logits_to_keep, vocab] of the last positions of one uncached forward."""
        if image_features is None:
            image_features = np.zeros((0, self.hidden), dtype=np.float32)
        self.check_features(ids, image_features)
        feed = self.decoder_feed(
            ids, image_features, 0, logits_to_keep, initialize_cache(self.decoder)
        )
        return self.decoder.run(["logits"], feed)[0][0]

    def generate_ids(
        self,
        input_ids: list[int],
        image_features: np.ndarray | None = None,
        max_new_tokens: int = 256,
        on_token=None,
    ) -> list[int]:
        """Greedy tokens after input_ids, whose image placeholders take image_features' rows."""
        no_features = np.zeros((0, self.hidden), dtype=np.float32)
        features = no_features if image_features is None else image_features
        self.check_features(input_ids, features)
        capacity = len(input_ids) + max_new_tokens if self.share_kv else None
        decoder = CachedDecoder(self.decoder, capacity)
        ids, tokens = input_ids, []
        for _ in range(max_new_tokens):
            logits = decoder.step(
                ids,
                image_features=features.astype(np.float32),
                video_features=no_features,
                logits_to_keep=np.array([1], dtype=np.int64),
            )
            token = int(np.argmax(logits[0, -1]))
            tokens.append(token)
            if on_token:
                on_token(token)
            if token == self.eos:
                break
            ids, features = [token], no_features
        return tokens

    def prompt_ids(
        self, messages: list[dict], grids: np.ndarray, enable_thinking: bool = True
    ) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
        )
        ids = self.tokenizer(prompt)["input_ids"]
        return expand_image_tokens(ids, grids, self.image_token, self.merge)

    def generate(
        self,
        messages: list[dict],
        images: list | None = None,
        max_new_tokens: int = 4096,
        stream: bool = True,
        enable_thinking: bool = True,
        max_image_tokens: int | None = None,
    ) -> str:
        start = time.perf_counter()
        features, grids = None, np.zeros((0, 3), dtype=np.int64)
        if images:
            features, grids = self.encode_images(images, max_image_tokens)
        ids = self.prompt_ids(messages, grids, enable_thinking)
        encoded = time.perf_counter()
        streamer = None
        if stream:
            from transformers import TextStreamer

            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
        slots = []

        def on_token(token: int):
            slots.append(("token", time.perf_counter()))
            if streamer:
                streamer.put(np.array([token]))

        tokens = self.generate_ids(ids, features, max_new_tokens, on_token)
        if streamer:
            streamer.end()
        logger.info(f"{len(ids)} prompt tokens, {len(tokens)} new tokens")
        log_speed(start, encoded, slots)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
