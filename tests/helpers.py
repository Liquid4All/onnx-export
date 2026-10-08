"""Shared test utilities."""

import logging
import pathlib
import shutil

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def derive_reference(
    onnx_dir: pathlib.Path, tmp_path: pathlib.Path, precision: str, name: str = "model", **kwargs
) -> pathlib.Path:
    """The decoder liquidonnx.quantize derives from onnx_dir/{name}.onnx, written to tmp_path."""
    from liquidonnx.quantize import derive_precision

    for path in onnx_dir.glob(f"{name}.onnx*"):
        shutil.copy(path, tmp_path)
    return derive_precision(tmp_path, precision, name, **kwargs)


def assert_same_graph(actual: pathlib.Path, expected: pathlib.Path):
    """Same nodes, graph I/O, opsets and initializers (names, dtypes, shapes and bytes)."""
    import onnx
    from onnx import numpy_helper

    a, b = onnx.load(str(actual)), onnx.load(str(expected))
    assert list(a.graph.node) == list(b.graph.node)
    assert list(a.graph.input) == list(b.graph.input)
    assert list(a.graph.output) == list(b.graph.output)
    assert list(a.opset_import) == list(b.opset_import)

    a_tensors = {init.name: numpy_helper.to_array(init) for init in a.graph.initializer}
    b_tensors = {init.name: numpy_helper.to_array(init) for init in b.graph.initializer}
    assert a_tensors.keys() == b_tensors.keys()
    for name, tensor in a_tensors.items():
        assert tensor.dtype == b_tensors[name].dtype, name
        assert tensor.shape == b_tensors[name].shape, name
        assert tensor.tobytes() == b_tensors[name].tobytes(), name


def get_model_name(model_id: str) -> str:
    """Extract model name from HF slug (e.g., 'LiquidAI/LFM2-350M' -> 'LFM2-350M')."""
    return model_id.split("/")[-1]


def get_export_dir(exports_dir: pathlib.Path, model_id: str) -> pathlib.Path:
    return exports_dir / f"{get_model_name(model_id)}-ONNX"


def get_onnx_dir(exports_dir: pathlib.Path, model_id: str) -> pathlib.Path:
    return get_export_dir(exports_dir, model_id) / "onnx"


def generate_with_logits(model, inputs, max_new_tokens: int) -> tuple[list[int], np.ndarray]:
    """Greedy answer through onnxruntime-genai (stop token included, as transformers does) and
    the logits that chose each token."""
    from liquidonnx.genai_runtime import generate

    tokens, logits = [], []

    def step(generator):
        tokens.append(int(generator.get_next_tokens()[0]))
        logits.append(np.asarray(generator.get_output("logits"))[0, -1].astype(np.float32))

    generate(model, inputs, max_new_tokens, step)
    return tokens, np.stack(logits)


def generate_pytorch(model, inputs: dict, max_new_tokens: int) -> tuple[list[int], np.ndarray]:
    """Greedy answer of a transformers model and the logits that chose each token."""
    import torch

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
    tokens = output.sequences[0, inputs["input_ids"].shape[1] :].tolist()
    return tokens, np.stack([step[0].float().numpy() for step in output.logits])


def generate_pytorch_cached(model, input_ids: list[int], max_new_tokens: int, eos: int):
    """Greedy answer of a transformers causal LM, one cached step at a time, and its logits.

    Unlike generate(), the steps match the ONNX decoder's; for LFM2-MoE, generate()'s expert
    routing drifts from a plain forward pass.
    """
    import torch

    tokens, logits, past = [], [], None
    ids = torch.tensor([input_ids])
    with torch.no_grad():
        for _ in range(max_new_tokens):
            length = len(input_ids) + len(tokens)
            output = model(
                input_ids=ids,
                attention_mask=torch.ones(1, length, dtype=torch.long),
                position_ids=torch.arange(length - ids.shape[1], length)[None],
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            logits.append(output.logits[0, -1].float().numpy())
            tokens.append(int(logits[-1].argmax()))
            if tokens[-1] == eos:
                break
            ids = torch.tensor([[tokens[-1]]])
    return tokens, np.stack(logits)


def text_coherence(model, tokenizer, genai_model, prompts: list[str], max_new_tokens: int) -> float:
    """Mean logit cosine of a multi-turn chat; each side answers its own conversation."""
    from liquidonnx.verify import compare_logits_similarity

    conversations = {"pytorch": [], "genai": []}
    similarities = []
    for turn, prompt in enumerate(prompts, 1):
        answers = {}
        for side, messages in conversations.items():
            messages.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            if side == "pytorch":
                answers[side] = generate_pytorch_cached(
                    model, ids, max_new_tokens, tokenizer.eos_token_id
                )
            else:
                answers[side] = generate_with_logits(genai_model, np.array(ids), max_new_tokens)
            tokens = answers[side][0]
            messages.append(
                {"role": "assistant", "content": tokenizer.decode(tokens, skip_special_tokens=True)}
            )
        similarities.append(compare_logits_similarity(answers["pytorch"][1], answers["genai"][1]))
        logger.info(f"  Turn {turn}: similarity={similarities[-1]:.4f}")
        for side, messages in conversations.items():
            logger.info(f"    {side}: {messages[-1]['content'][:80]}")
    return float(np.mean(similarities))


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad image to square with black borders, centered."""
    w, h = image.size
    if w == h:
        return image
    size = max(w, h)
    square = Image.new("RGB", (size, size), (0, 0, 0))
    square.paste(image, ((size - w) // 2, (size - h) // 2))
    return square
