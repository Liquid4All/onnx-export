"""Arguments, output folder, provenance file and summary shared by the export CLIs."""

import argparse
import datetime
import importlib.metadata
import json
import logging
import os
import pathlib
import platform
import stat
import subprocess
import sys

from liquidonnx.external_data import split_external_data
from liquidonnx.quantize import DEFAULT_BLOCK_SIZE

logger = logging.getLogger(__name__)

EXPORT_INFO = "export_info.json"
# Import names of the packages whose versions can change the exported bytes
PROVENANCE_MODULES = ("onnxruntime", "onnx", "torch", "transformers", "onnxruntime_genai")


def add_export_arguments(
    parser: argparse.ArgumentParser,
    precisions: tuple[str, ...],
    default_precisions: tuple[str, ...],
    example: str,
    block_size_help: str = f"Block size for quantization (default: {DEFAULT_BLOCK_SIZE})",
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
        help=block_size_help,
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


def apply_umask(output_dir: pathlib.Path):
    """Give every file in output_dir the mode a new file gets (0o666 & ~umask).

    onnx.save_model creates external data files 0600, so other users could not read the weights,
    and shutil.copy2 keeps the modes of the checkpoint files it copies.
    """
    umask = os.umask(0)
    os.umask(umask)
    mode = 0o666 & ~umask
    for path in output_dir.rglob("*"):
        info = path.lstat()
        if stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) != mode:
            path.chmod(mode)


# === Provenance ===


def source_commit() -> tuple[str | None, bool | None]:
    """(commit, uncommitted changes to tracked files) of the liquidonnx checkout that runs; (None,
    None) for an installed package."""
    package = pathlib.Path(__file__).resolve().parent
    if package.parent.name != "src":
        return None, None

    def git(*command: str) -> str:
        run = subprocess.run(
            ["git", "-C", str(package), *command], capture_output=True, text=True, check=True
        )
        return run.stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def cached_revision(model: str, revision: str | None = None) -> str | None:
    """The commit of the cached Hugging Face snapshot that model (an ID) and revision (None: main)
    resolve to; None for a local folder or a snapshot not in the cache."""
    if pathlib.Path(model).is_dir():
        return None
    from huggingface_hub import try_to_load_from_cache

    try:
        path = try_to_load_from_cache(model, "config.json", revision=revision)
    except ValueError:  # not a valid repository ID (HFValidationError)
        return None
    return pathlib.Path(path).parent.name if isinstance(path, str) else None


def package_versions() -> dict[str, str]:
    """Distribution -> version of PROVENANCE_MODULES' installed packages, e.g. onnxruntime-gpu."""
    distributions = importlib.metadata.packages_distributions()
    return {
        name: importlib.metadata.version(name)
        for module in PROVENANCE_MODULES
        for name in distributions.get(module, [])
    }


def write_export_info(
    output_dir: pathlib.Path, model: str, precisions: list[str], revision: str | None = None
) -> dict:
    """Append this run to output_dir/export_info.json and return its record: the liquidonnx
    commit, the checkpoint and its revision (revision is a pin; None resolves main), the package
    versions, the host and the precisions it built.

    The int8 weights onnxruntime quantizes can differ in the last bit between CPU architectures
    (README 3.1b), so the host is part of what an export's bytes depend on. Earlier runs' records
    stay: --skip-export adds precisions to a folder another run exported.
    """
    commit, dirty = source_commit()
    record = {
        "time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "liquidonnx": {"commit": commit, "dirty": dirty},
        "command": [pathlib.Path(sys.argv[0]).name, *sys.argv[1:]],
        "checkpoint": {
            "model": model,
            "revision": cached_revision(model, revision),
            "pinned": revision,
        },
        "precisions": precisions,
        "packages": package_versions(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    path = output_dir / EXPORT_INFO
    runs = json.loads(path.read_text())["exports"] if path.exists() else []
    path.write_text(json.dumps({"exports": [*runs, record]}, indent=2) + "\n")
    return record


def finish(
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    precisions: list[str],
    revisions: dict[str, str] | None = None,
):
    """Split large external data files, record the run (write_export_info, with args.model's pin
    in revisions) and reset file modes, then log what the folder holds. precisions are the derived
    ones; fp32 is recorded unless --skip-export reused it."""
    built = ([] if getattr(args, "skip_export", False) else ["fp32"]) + precisions
    write_export_info(output_dir, args.model, built, (revisions or {}).get(args.model))
    onnx_dir = output_dir / "onnx"
    if not args.no_split_data:
        chunk_size = int(args.split_data * 1024**3)
        for onnx_file in sorted(onnx_dir.glob("*.onnx")):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size:
                logger.info(f"Splitting {onnx_file.name} ({args.split_data:.1f} GB chunks)")
                split_external_data(onnx_file, chunk_size=chunk_size)
    apply_umask(output_dir)

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
