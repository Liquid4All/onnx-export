"""
LiquidONNX - ONNX export and inference tools for LFM2 models.
"""

import os

__version__ = "0.1.0"


def remote_code_enabled() -> bool:
	"""Return whether loading Python code from model repositories is allowed."""
	return os.environ.get("LIQUIDONNX_TRUST_REMOTE_CODE", "").lower() in {
		"1",
		"true",
		"yes",
	}
