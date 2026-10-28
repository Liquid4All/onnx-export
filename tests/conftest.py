import logging
import pathlib
import sys

import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)  # huggingface-hub 2.x
logging.getLogger("transformers").setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption(
        "--exports-dir",
        action="store",
        default=".",
        help="Base directory containing exports/ folder",
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device of the PyTorch references and execution provider of the ONNX sessions, "
        "for the tests that take the device fixture (TF32 off on CUDA)",
    )
    parser.addoption(
        "--ep",
        action="store",
        choices=["cpu", "cuda"],
        help="Execution provider of the ONNX sessions when it differs from --device, e.g. the "
        "int4 bundles on the CPU against a CUDA reference",
    )
    parser.addoption(
        "--lfm3-model",
        action="store",
        default="LiquidAI/LFM3-1.7B",
        help="LFM3 checkpoint of the tests in tests/test_lfm3 that use a real one; its export is "
        "looked up under --exports-dir",
    )
    parser.addoption(
        "--lfm3-vl-model",
        action="store",
        default="LiquidAI/LFM3-VL-2B",
        help="LFM3-VL checkpoint of the tests in tests/test_lfm3_vl that use a real one; its "
        "export is looked up under --exports-dir",
    )
    parser.addoption(
        "--lfm3-audio-model",
        action="store",
        default="LiquidAI/LFM3-Audio-2B",
        help="LFM3-Audio checkpoint of the tests in tests/test_lfm3_audio that use a real one "
        "(LiquidAI/LFM3-Omni-2B for its audio parts); its export is looked up under "
        "--exports-dir",
    )
    parser.addoption(
        "--lfm3-omni-model",
        action="store",
        default="LiquidAI/LFM3-Omni-2B",
        help="LFM3-Omni checkpoint of the tests in tests/test_lfm3_omni that use a real one; its "
        "export is looked up under --exports-dir",
    )
    parser.addoption(
        "--d1-model",
        action="store",
        default="LiquidAI/d1-2B",
        help="d1 checkpoint of the tests in tests/test_d1 that use a real one "
        "(LiquidAI/d1-2B or LiquidAI/d1-Omni-2B); its export is looked up under --exports-dir",
    )
    parser.addoption(
        "--matmulnbits-accuracy-level",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Diagnostic: every graph the tests load with MatMulNBits gets this accuracy_level, "
        "in an in-memory copy (the export stays as it is); 0 keeps the activations float where "
        "the q8 recipe's 4 quantizes them to int8 on the CPU EP",
    )
    parser.addoption(
        "--reference-cache",
        action="store",
        help="Folder of the PyTorch reference's greedy slots in tests/test_lfm3_audio's end-to-end "
        "tests: read when there, written when computed, keyed by the checkpoint revision, prompt "
        "and clips (not the device), so a folder filled on CUDA spares a CPU host the reference",
    )
    parser.addoption(
        "--run-long",
        action="store_true",
        help="Run the tests marked long (long contexts, wikitext-2, bf16 floors: minutes to hours "
        "per test); they are skipped otherwise",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        help="Run the tests marked slow (end-to-end runs of minutes per cell on a CPU EP, TF32 "
        "runs); they are skipped otherwise",
    )
    parser.addoption(
        "--wikitext",
        action="store",
        type=pathlib.Path,
        help="wiki.test.raw of wikitext-2-raw-v1, for the long tests that read it (README 5.1 "
        "downloads it)",
    )


# Opt-in markers: their tests are skipped unless the option is given or -m names the marker
OPT_IN = {
    "long": ("--run-long", "measurement on long inputs"),
    "slow": ("--run-slow", "long end-to-end runs"),
}


def pytest_configure(config):
    for marker, (option, what) in OPT_IN.items():
        config.addinivalue_line(
            "markers", f"{marker}: opt-in {what}; runs with {option} or -m {marker}"
        )


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr") or ""
    for marker, (option, _) in OPT_IN.items():
        if config.getoption(option) or marker in markexpr:
            continue
        skip = pytest.mark.skip(reason=f"opt-in: pass {option} or -m {marker}")
        for item in items:
            if item.get_closest_marker(marker):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def wikitext(request) -> pathlib.Path:
    path = request.config.getoption("--wikitext")
    if path is None or not path.exists():
        pytest.fail(f"--wikitext {path}: give the path of wikitext-2's wiki.test.raw (README 5.1)")
    return path


@pytest.fixture(scope="session", autouse=True)
def matmulnbits_accuracy_level(request):
    """--matmulnbits-accuracy-level: onnxruntime.InferenceSession loads each graph file with
    MatMulNBits as helpers.with_accuracy_level's copy, its weights from the export."""
    level = request.config.getoption("--matmulnbits-accuracy-level")
    if level is None:
        yield
        return
    import onnxruntime as ort
    from helpers import EXTERNAL_DATA_FOLDER, with_accuracy_level

    session = ort.InferenceSession

    def patched(model, *args, **kwargs):
        if isinstance(model, str | pathlib.Path) and str(model).endswith(".onnx"):
            path = pathlib.Path(model)
            if (copy := with_accuracy_level(path, level)) is not None:
                options = (args[0] if args else kwargs.get("sess_options")) or ort.SessionOptions()
                options.add_session_config_entry(EXTERNAL_DATA_FOLDER, str(path.parent))
                model, args, kwargs = copy, args[1:], {**kwargs, "sess_options": options}
        return session(model, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ort, "InferenceSession", patched)
        yield


@pytest.fixture(scope="session")
def exports_dir(request) -> pathlib.Path:
    base = pathlib.Path(request.config.getoption("--exports-dir"))
    return base / "exports"


@pytest.fixture(scope="session")
def ep(request, device) -> str:
    return request.config.getoption("--ep") or device


@pytest.fixture(scope="session")
def device(request) -> str:
    device = request.config.getoption("--device")
    if device == "cuda":
        import torch

        # float32 references stay float32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device
