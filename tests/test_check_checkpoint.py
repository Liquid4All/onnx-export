"""
export_cli.check_checkpoint, which every LFM3 exporter runs on config.json before it creates its
folder or downloads weights: the CLI of each family, the LFM2-based d1 models that no CLI exports,
and quantized checkpoints (no download).

Run with:
    uv run pytest tests/test_check_checkpoint.py -v
"""

import json
import pathlib
import sys

import huggingface_hub
import pytest

from liquidonnx.d1 import export as d1_export
from liquidonnx.export_cli import checkpoint_error
from liquidonnx.lfm3 import export as text_export
from liquidonnx.lfm3_audio import export as audio_export
from liquidonnx.lfm3_omni import export as omni_export
from liquidonnx.lfm3_vl import export as vl_export

ASSETS = pathlib.Path(__file__).parent / "test_lfm3_audio" / "assets"
AUDIO = json.loads((ASSETS / "LFM3-Audio-2B" / "config.json").read_text())
OMNI = json.loads((ASSETS / "LFM3-Omni-2B" / "config.json").read_text())

# The CLIs whose checkpoints each LFM3 exporter takes, as its main() checks them
TAKES = {
    "lfm3-export": ("lfm3-export",),
    "lfm3-vl-export": ("lfm3-vl-export",),
    "lfm3-audio-export": ("lfm3-audio-export", "lfm3-omni-export"),
    "lfm3-omni-export": ("lfm3-omni-export",),
    "d1-export": ("d1-export",),
}


def config(architecture: str, model_type: str, **keys) -> dict:
    return {"architectures": [architecture], "model_type": model_type, **keys}


# The routing keys of each family's config.json (the public checkpoints'), and its CLI
FAMILIES = {
    "LFM3-1.7B": (config("Lfm3ForCausalLM", "lfm3"), "lfm3-export"),
    "LFM3-VL-2B": (config("Lfm3VlForConditionalGeneration", "lfm3_vl"), "lfm3-vl-export"),
    "LFM3-Audio-2B": (AUDIO, "lfm3-audio-export"),
    "LFM3-Omni-2B": (OMNI, "lfm3-omni-export"),
    "d1-2B": (
        config("D1Model", "lfm3_vl", auto_map={"AutoModel": "modeling_d1_lfm3_vl.D1Model"}),
        "d1-export",
    ),
    "d1-Omni-2B": (
        config("D1Model", "lfm3_omni", auto_map={"AutoModel": "modeling_d1_lfm3_omni.D1Model"}),
        "d1-export",
    ),
    "LFM2.5-350M": (config("Lfm2ForCausalLM", "lfm2"), "lfm2-export"),
    "LFM2.5-VL-1.6B": (config("Lfm2VlForConditionalGeneration", "lfm2_vl"), "lfm2-vl-export"),
    "LFM2.5-8B-A1B": (config("Lfm2MoeForCausalLM", "lfm2_moe"), "lfm2-moe-export"),
    # no model_type
    "LFM2.5-Audio-1.5B": (
        {"architectures": ["Lfm2AudioForConditionalGeneration"]},
        "lfm2-audio-export",
    ),
}

D1_3B = config(
    "Lfm2VlForConditionalGeneration", "lfm2_vl", auto_map={"AutoModel": "modeling_d1.D1Model"}
)
LFM2_D1 = {
    "d1-3B": D1_3B,
    "d1-3B-w8a8": {**D1_3B, "quantization_config": {"quant_method": "torchao"}},
    "d1-omni-600M": config(
        "D1OmniModel",
        "d1_omni",
        auto_map={"AutoConfig": "modeling_d1.D1OmniConfig", "AutoModel": "modeling_d1.D1OmniModel"},
    ),
}

QUANTIZED = (
    "is quantized (config.json has a quantization_config): export the original checkpoint it "
    "was converted from, not an FP8, NVFP4 or MLX conversion"
)


@pytest.mark.parametrize("cli", TAKES)
@pytest.mark.parametrize("name", FAMILIES)
def test_each_family_names_its_cli(name, cli):
    cfg, exporter = FAMILIES[name]
    error = checkpoint_error(cfg, f"LiquidAI/{name}", *TAKES[cli])
    if exporter in TAKES[cli]:
        assert error is None
    else:
        assert error == f"export LiquidAI/{name} ({cfg['architectures'][0]}) with {exporter}"


@pytest.mark.parametrize("cli", TAKES)
@pytest.mark.parametrize("name", LFM2_D1)
def test_lfm2_based_d1_models_have_no_cli(name, cli):
    assert checkpoint_error(LFM2_D1[name], f"LiquidAI/{name}", *TAKES[cli]) == (
        f"LiquidAI/{name} is an LFM2-based d1 model, which these exporters don't support "
        "(see README 1)"
    )


# quantization_config of LFM3-1.7B-FP8, LFM3-VL-2B-NVFP4 and an MLX conversion
@pytest.mark.parametrize(
    "name,quantization,cli",
    [
        ("LFM3-1.7B", {"quant_method": "compressed-tensors"}, "lfm3-export"),
        ("LFM3-VL-2B", {"quant_method": "modelopt"}, "lfm3-vl-export"),
        ("LFM3-1.7B", {"bits": 4, "group_size": 64, "mode": "affine"}, "lfm3-export"),
    ],
    ids=["FP8", "NVFP4", "MLX"],
)
def test_quantized_checkpoints_are_no_export_input(name, quantization, cli):
    cfg = {**FAMILIES[name][0], "quantization_config": quantization}
    assert checkpoint_error(cfg, "m", cli) == f"m {QUANTIZED}"
    # Another family's CLI names the right one first
    assert (
        checkpoint_error(cfg, "m", "d1-export")
        == f"export m ({cfg['architectures'][0]}) with {cli}"
    )


@pytest.mark.parametrize("cfg", [{}, config("Qwen3ForCausalLM", "qwen3")], ids=["none", "other"])
def test_other_architectures_are_left_to_the_parsers(cfg):
    for clis in TAKES.values():
        assert checkpoint_error(cfg, "m", *clis) is None


@pytest.mark.parametrize(
    "cli,module",
    [
        ("lfm3-export", text_export),
        ("lfm3-vl-export", vl_export),
        ("lfm3-audio-export", audio_export),
        ("lfm3-omni-export", omni_export),
        ("d1-export", d1_export),
    ],
)
def test_exporters_check_before_downloading(cli, module, tmp_path, monkeypatch, capsys):
    """main() fetches config.json alone, at the revision it pins, and stops on it before it creates
    its folder or downloads weights. The checkpoint is a quantized LFM3-Omni: the error is the
    quantization for the two CLIs that take LFM3-Omni, the family's CLI for the others."""
    model, revision = "example/tiny-omni", "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setitem(module.REVISIONS, model, revision)
    cfg = {**OMNI, "quantization_config": {"quant_method": "compressed-tensors"}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    fetched = []

    def hf_hub_download(repo_id, filename, revision=None):
        fetched.append((repo_id, filename, revision))
        return str(tmp_path / "config.json")

    def snapshot_download(*args, **kwargs):
        raise AssertionError("downloaded the checkpoint")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [cli, model, "--precision"])
    with pytest.raises(SystemExit) as stop:
        module.main()
    assert stop.value.code == 2
    error = checkpoint_error(cfg, model, *TAKES[cli])
    assert (QUANTIZED in error) == (cli in ("lfm3-audio-export", "lfm3-omni-export"))
    assert capsys.readouterr().err.endswith(f"{cli}: error: {error}\n")
    assert fetched == [(model, "config.json", revision)]
    assert not (tmp_path / "exports").exists()
