"""Write the synthetic LFM2.5-Audio export the browser tests load.

Produces fp32 and q4 variants of every graph plus a manifest the JS harness uses
to build feeds and locate external-data files.

    uv run python tests/web/export_synthetic.py            # -> tests/web/.synthetic
    uv run python tests/web/export_synthetic.py /some/dir
"""

import json
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_lfm2_audio.synthetic import (  # noqa: E402
    HIDDEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    build_model_dir,
    decoder_config,
)

from liquidonnx.quantize import quantize_model  # noqa: E402

GRAPHS = ["decoder", "audio_encoder", "audio_detokenizer", "vocoder_depthformer"]


def main(out_dir: pathlib.Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model_dir = build_model_dir(out_dir)
    onnx_dir = model_dir / "onnx"

    graphs: dict[str, dict[str, dict]] = {}
    for name in GRAPHS:
        fp32 = onnx_dir / f"{name}.onnx"
        q4 = onnx_dir / f"{name}_q4.onnx"
        quantize_model(fp32, q4, bits=4, block_size=32, exclude_lm_head=False, symmetric=True)
        graphs[name] = {
            precision: {
                "file": path.name,
                "external_data": [p.name for p in onnx_dir.glob(f"{path.stem}.onnx_data*")],
            }
            for precision, path in (("fp32", fp32), ("q4", q4))
        }

    lfm = decoder_config()["lfm"]
    manifest = {
        "model_dir": model_dir.name,
        "hidden": HIDDEN,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HIDDEN // NUM_HEADS,
        "conv_L": lfm["conv_L_cache"],
        "layer_types": lfm["layer_types"],
        "graphs": graphs,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"synthetic export written to {out_dir}")


if __name__ == "__main__":
    target = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else pathlib.Path(__file__).parent / ".synthetic"
    )
    main(target)
