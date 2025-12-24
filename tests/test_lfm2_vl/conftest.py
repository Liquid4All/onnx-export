"""LFM2-VL test fixtures."""

import pathlib

import pytest

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


@pytest.fixture
def cardinal_image() -> pathlib.Path:
    return ASSETS_DIR / "cardinal.jpg"


@pytest.fixture
def bluejay_image() -> pathlib.Path:
    return ASSETS_DIR / "bluejay.jpg"
