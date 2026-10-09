"""
Run liquidonnx exports with onnxruntime-genai.

An export folder is an onnxruntime-genai model: genai_config.json names the graphs of the default
precision, and every other precision is loaded by overlaying its file names (each family's
export module has genai_files(precision)).

Needs onnxruntime-genai 0.17.1 or later (PyPI), or a build of main; CI tests the build of main at
liquidonnx.genai_builder.GENAI_COMMIT (see README).
"""

import json
import logging
import pathlib
from collections.abc import Callable

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og
from packaging.version import Version

from liquidonnx.session import preload_cuda_libraries

logger = logging.getLogger(__name__)

EXECUTION_PROVIDERS = ("cpu", "cuda")
GENAI_MIN_VERSION = Version("0.17.1")  # tested release; 0.17.0 added lfm2_moe, lfm2_vl, lfm2_audio
MOE_ORT_MIN_VERSION = Version("1.31.0.dev0")  # fast CPU QMoE kernels; the 1.31 nightlies pass


def _filenames(entries: dict) -> list[str]:
    names = []
    for key, value in entries.items():
        if key == "filename":
            names.append(value)
        elif isinstance(value, dict):
            names += _filenames(value)
    return names


def check_genai_version(version: str):
    """Fail on an onnxruntime-genai release older than GENAI_MIN_VERSION.

    Builds of main report a dev version (e.g. 0.18.0-dev) and pass.
    """
    parsed = Version(version)
    if not parsed.is_devrelease and parsed < GENAI_MIN_VERSION:
        raise RuntimeError(
            f"liquidonnx needs onnxruntime-genai {GENAI_MIN_VERSION} or later, or a build of main; "
            f"found {version}. Run `uv sync` or see the README."
        )


def check_moe_onnxruntime_version(model_type: str, version: str):
    """Warn that LFM2-MoE decodes slowly on CPU with an onnxruntime older than MOE_ORT_MIN_VERSION."""
    if model_type == "lfm2_moe" and Version(version) < MOE_ORT_MIN_VERSION:
        logger.warning(
            f"LFM2-MoE decodes at about 1 tok/s on CPU with onnxruntime {version}, against about "
            "59 tok/s with the onnxruntime 1.31 nightly. The README (2. Installation) shows which "
            "installs get the nightly."
        )


def load_model(
    model_dir: pathlib.Path, files: dict | None = None, ep: str = "cpu", tf32: bool = True
) -> og.Model:
    """The export at model_dir; files replaces genai_config.json model entries (a precision).

    The CUDA EP runs fp32 matmuls and convolutions in TF32 unless tf32 is False.
    """
    check_genai_version(og.__version__)
    config = og.Config(str(model_dir))
    if ep == "cpu":
        genai_config = json.loads((model_dir / "genai_config.json").read_text())
        check_moe_onnxruntime_version(genai_config["model"]["type"], ort.__version__)
    if files:
        missing = [f for f in _filenames(files) if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"{model_dir} has no {', '.join(missing)}")
        config.overlay(json.dumps({"model": files}))
    if ep != "cpu":
        config.clear_providers()
        config.append_provider(ep)
    if ep == "cuda":
        if not tf32:
            config.set_provider_option(ep, "use_tf32", "0")
        preload_cuda_libraries()
    logger.info(f"Loading {model_dir} ({ep}) with onnxruntime-genai {og.__version__}...")
    return og.Model(config)


def generate(
    model: og.Model,
    inputs: np.ndarray | og.NamedTensors,
    max_new_tokens: int,
    on_step: Callable[[og.Generator], None] | None = None,
    **search_options,
) -> og.Generator:
    """Run one answer to the end; inputs are prompt token ids or a multimodal processor's output.

    Text is decoded greedily unless search_options says otherwise. on_step sees the generator
    after every new token.
    """
    if isinstance(inputs, np.ndarray):
        prompt_length = inputs.shape[-1]
    else:
        prompt_length = inputs["input_ids"].as_numpy().shape[-1]
    params = og.GeneratorParams(model)
    params.set_search_options(
        **{"do_sample": False, "max_length": prompt_length + max_new_tokens, **search_options}
    )
    generator = og.Generator(model, params)
    if isinstance(inputs, np.ndarray):
        generator.append_tokens(inputs.astype(np.int32))
    else:
        generator.set_inputs(inputs)
    while not generator.is_done():
        generator.generate_next_token()
        if on_step:
            on_step(generator)
    return generator


class TokenPrinter:
    """generate() step that streams the answer to stdout, leaving out the ids in skip and the
    end-of-turn tokens, which genai leaves out of the sequence.

    Each step prints what its token adds to tokenizer.decode() of the answer so far (0.4 ms at
    4,096 tokens), so the printed text is the decoded answer, where genai's TokenizerStream pads
    special tokens with spaces for LFM2-350M, -700M and -1.2B and the VL models. A trailing
    U+FFFD, a character whose bytes have not all come, waits for the next token.
    """

    def __init__(self, tokenizer: og.Tokenizer, skip: tuple[int, ...] = ()):
        self.tokenizer = tokenizer
        self.skip = {*skip, *map(int, tokenizer.eos_token_ids)}
        self.tokens: list[int] = []
        self.printed = ""

    def __call__(self, generator: og.Generator):
        token = int(generator.get_next_tokens()[0])
        done = generator.is_done()
        if token not in self.skip:
            self.tokens.append(token)
        elif not done:
            return
        text = self.tokenizer.decode(np.array(self.tokens, dtype=np.int32))
        if not done:
            text = text.rstrip("\ufffd")
        print(text[len(self.printed) :], end="", flush=True)
        self.printed = text


def add_runtime_arguments(parser, precisions: tuple[str, ...]):
    """--precision, --ep and --no-tf32, parsed into args.precision, args.ep and args.tf32."""
    parser.add_argument(
        "--precision",
        choices=["fp32", *precisions],
        help="Precision to load (default: the one genai_config.json points at)",
    )
    parser.add_argument(
        "--ep",
        choices=EXECUTION_PROVIDERS,
        default="cpu",
        help="Execution provider (cuda needs onnxruntime-genai-cuda; default: cpu)",
    )
    parser.add_argument(
        "--no-tf32",
        dest="tf32",
        action="store_false",
        help="On CUDA, run fp32 matmuls and convolutions in full fp32 instead of onnxruntime's "
        "default TF32, to match an fp32 reference",
    )
