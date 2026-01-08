"""
External data utilities for ONNX models.

Provides functions to split large external data files into smaller chunks
for better compatibility with file systems and web deployment.
"""

import logging
import pathlib

import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model, set_external_data

logger = logging.getLogger(__name__)

# Default chunk size: 2GB (safe for most filesystems and git LFS)
DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes

# Minimum tensor size to externalize (1KB)
MIN_EXTERNAL_SIZE = 1024


def _model_uses_external_data(model: onnx.ModelProto) -> bool:
    """Check if any tensor in the model uses external data."""
    for tensor in model.graph.initializer:
        if (
            tensor.HasField("data_location")
            and tensor.data_location == onnx.TensorProto.EXTERNAL
        ):
            return True
    return False


def _get_tensor_bytes(tensor: onnx.TensorProto) -> bytes:
    """Get raw bytes from a tensor, handling all data types."""
    # If tensor has raw_data, use it directly
    if tensor.raw_data:
        return bytes(tensor.raw_data)

    # Otherwise convert through numpy
    try:
        arr = numpy_helper.to_array(tensor)
        return arr.tobytes()
    except Exception:
        # Fallback: return empty if we can't convert
        return b""


def split_external_data(
    model_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pathlib.Path:
    """Split ONNX model's external data into multiple chunk files.

    Transforms a model with a single large .onnx_data file into one with
    multiple numbered chunks: model.onnx_data, model.onnx_data_1, model.onnx_data_2, etc.

    This matches the onnx-community convention for large models.

    Args:
        model_path: Path to the ONNX model file
        output_path: Output path (default: overwrite input)
        chunk_size: Maximum size per chunk file in bytes (default: 2GB)

    Returns:
        Path to the output model file
    """
    model_path = pathlib.Path(model_path)
    if output_path is None:
        output_path = model_path
    output_path = pathlib.Path(output_path)

    # Load model with all external data loaded into memory
    logger.info(f"Loading model from {model_path}")
    model = onnx.load(str(model_path), load_external_data=False)

    has_external = _model_uses_external_data(model)
    if has_external:
        logger.info("Loading external data into memory...")
        load_external_data_for_model(model, str(model_path.parent))

    # Prepare output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem + ".onnx_data"

    # Clean up old data files
    for old_file in output_path.parent.glob(f"{output_path.stem}.onnx_data*"):
        old_file.unlink()
        logger.debug(f"Removed old file: {old_file}")

    # Write tensors to chunk files
    chunk_idx = 0
    current_chunk_path = output_path.parent / base_name
    current_chunk_file = open(current_chunk_path, "wb")
    current_chunk_size = 0
    tensors_externalized = 0

    logger.info(f"Splitting external data (chunk size: {chunk_size / 1e9:.2f} GB)")

    for tensor in model.graph.initializer:
        tensor_data = _get_tensor_bytes(tensor)
        tensor_size = len(tensor_data)

        # Skip small tensors - keep them inline
        if tensor_size < MIN_EXTERNAL_SIZE:
            continue

        # Check if we need a new chunk
        if current_chunk_size > 0 and current_chunk_size + tensor_size > chunk_size:
            current_chunk_file.close()
            logger.info(
                f"  Chunk {chunk_idx}: {current_chunk_path.name} "
                f"({current_chunk_size / 1e9:.2f} GB)"
            )
            chunk_idx += 1
            chunk_name = f"{base_name}_{chunk_idx}"
            current_chunk_path = output_path.parent / chunk_name
            current_chunk_file = open(current_chunk_path, "wb")
            current_chunk_size = 0

        # Write tensor data
        offset = current_chunk_file.tell()
        current_chunk_file.write(tensor_data)

        # Update tensor to point to the chunk file
        location = current_chunk_path.name
        set_external_data(tensor, location=location, offset=offset, length=tensor_size)

        # Clear inline data from tensor (it's now external)
        tensor.ClearField("raw_data")
        tensor.ClearField("float_data")
        tensor.ClearField("int32_data")
        tensor.ClearField("int64_data")
        tensor.ClearField("double_data")
        tensor.ClearField("uint64_data")

        current_chunk_size += tensor_size
        tensors_externalized += 1

    # Close final chunk
    current_chunk_file.close()
    if current_chunk_size > 0:
        logger.info(
            f"  Chunk {chunk_idx}: {current_chunk_path.name} "
            f"({current_chunk_size / 1e9:.2f} GB)"
        )
    else:
        # Remove empty file
        current_chunk_path.unlink()

    # Save updated model
    logger.info(f"Saving model to {output_path}")
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=False,  # Already set up external data refs
    )

    total_chunks = chunk_idx + 1 if current_chunk_size > 0 else chunk_idx
    logger.info(f"Split complete: {tensors_externalized} tensors in {total_chunks} chunk(s)")

    return output_path


def get_external_data_files(model_path: pathlib.Path) -> list[pathlib.Path]:
    """Get list of external data files for a model.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        List of paths to external data files
    """
    model_path = pathlib.Path(model_path)
    base_pattern = model_path.stem + ".onnx_data"

    files = []
    # Main data file
    main_data = model_path.parent / base_pattern
    if main_data.exists():
        files.append(main_data)

    # Numbered chunks
    for chunk_file in sorted(model_path.parent.glob(f"{base_pattern}_*")):
        files.append(chunk_file)

    return files


def get_total_external_data_size(model_path: pathlib.Path) -> int:
    """Get total size of all external data files in bytes."""
    return sum(f.stat().st_size for f in get_external_data_files(model_path))
