"""
d1-infer end to end in a process of its own (no PyTorch d1 model loaded beside it, as in the other
tests): the README's picture request, and the clip one on d1-Omni-2B, with the bundle --ep prefers.

Run with:
    uv run --no-sync pytest tests/test_d1/test_infer.py -v --d1-model LiquidAI/d1-2B --device cuda
"""

import json
import pathlib
import subprocess
import sys

import pytest

from .conftest import cats

SAMPLES = pathlib.Path(__file__).parents[2] / "samples"


@pytest.mark.parametrize("request_name", ["picture", "clip"])
def test_infer(model_id, export_dir, ep, request_name):
    if request_name == "clip" and "omni" not in model_id.lower():
        pytest.skip("audio is d1-Omni's")
    clip = SAMPLES / "audio" / "woodworks_question.wav"
    media = {"picture": cats() if request_name == "picture" else None, "clip": clip}
    command = [sys.executable, "-m", "liquidonnx.d1.infer", "--model", str(export_dir)]
    command += ["--request", str(SAMPLES / "d1" / f"{request_name}.json")]
    command += ["--media", str(media[request_name])] + ["--cpu"] * (ep == "cpu")
    out = subprocess.run(command, capture_output=True, text=True, check=True)
    answers = json.loads(out.stdout)["answers"]
    if request_name == "picture":
        assert answers["cats"]["number"] == 2.0
        assert answers["every_cat"]["count"] == 2
        assert answers["remote"]["found"] > 0.5
    else:  # the clip went through the processor and the audio encoder (the counts are checked)
        assert all(0 <= answers[name]["noul"] <= 1 for name in ("speech", "question"))
