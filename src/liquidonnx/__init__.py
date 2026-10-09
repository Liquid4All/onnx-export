"""
LiquidONNX - ONNX export and inference tools for LFM2 models.
"""

import os

# The onnxruntime 1.32 nightlies upload telemetry from a thread that can crash a process holding
# torch (SIGSEGV at exit); the variable must be set before onnxruntime starts. Child processes, such as
# the genai builder, inherit it. ORT_DISABLE_TELEMETRY=0 keeps telemetry on.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"


def remote_code_enabled() -> bool:
    """Return whether loading Python code from model repositories is allowed."""
    return os.environ.get("LIQUIDONNX_TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"}
