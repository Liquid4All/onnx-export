"""Arguments, output folder and summary shared by the lfm2*-export CLIs."""

import argparse
import logging
import pathlib

from liquidonnx.external_data import split_external_data
from liquidonnx.quantize import DEFAULT_BLOCK_SIZE

logger = logging.getLogger(__name__)


def add_export_arguments(
    parser: argparse.ArgumentParser,
    precisions: tuple[str, ...],
    default_precisions: tuple[str, ...],
    example: str,
):
    parser.add_argument("model", help=f"HuggingFace model ID or local path (e.g., {example})")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory (default: current directory)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output folder name (default: {model-name}-ONNX)",
    )
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help=f"Output precisions: {', '.join(precisions)} "
        f"(no value: {' '.join(default_precisions)})",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for quantization (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--split-data",
        type=float,
        default=2.0,
        metavar="GB",
        help="Split external data into chunks (default: 2GB per chunk)",
    )
    parser.add_argument(
        "--no-split-data",
        action="store_true",
        help="Disable external data splitting",
    )


def parse_precisions(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    precisions: tuple[str, ...],
    default_precisions: tuple[str, ...],
) -> list[str]:
    """--precision in the order of precisions: none without the flag, the defaults without values."""
    if args.precision is None:
        return []
    requested = [p.lower() for p in args.precision] or list(default_precisions)
    for p in requested:
        if p not in precisions:
            parser.error(f"Invalid precision: {p}. Use {', '.join(precisions)}.")
    return sorted(set(requested), key=precisions.index)


def output_dir(args: argparse.Namespace) -> pathlib.Path:
    name = args.output_name or f"{pathlib.Path(args.model).name}-ONNX"
    return args.output_dir / "exports" / name


def finish(args: argparse.Namespace, output_dir: pathlib.Path):
    """Split large external data files, then log what the export folder holds."""
    onnx_dir = output_dir / "onnx"
    if not args.no_split_data:
        chunk_size = int(args.split_data * 1024**3)
        for onnx_file in sorted(onnx_dir.glob("*.onnx")):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size:
                logger.info(f"Splitting {onnx_file.name} ({args.split_data:.1f} GB chunks)")
                split_external_data(onnx_file, chunk_size=chunk_size)

    logger.info("=" * 60)
    logger.info("Output summary")
    logger.info("=" * 60)
    files = ", ".join(f.name for f in sorted(onnx_dir.glob("*.onnx")))
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(f"  {output_dir} ({total_size / 1e9:.2f} GB)")
    logger.info(f"    Files: {files}")


def log_step(message: str):
    logger.info("=" * 60)
    logger.info(message)
    logger.info("=" * 60)
