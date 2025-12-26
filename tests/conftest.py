import logging
import os
import pathlib
import sys

import pytest

# Add tests directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption(
        "--exports-dir",
        action="store",
        default=".",
        help="Base directory containing exports/ folder",
    )


@pytest.fixture(scope="session")
def exports_dir(request) -> pathlib.Path:
    base = pathlib.Path(request.config.getoption("--exports-dir"))
    return base / "exports"


@pytest.fixture(scope="session")
def community_dir() -> pathlib.Path:
    """Base directory for onnx-community models."""
    env_dir = os.environ.get("ONNX_COMMUNITY_DIR")
    if env_dir:
        return pathlib.Path(env_dir)
    return pathlib.Path("onnx-community")
