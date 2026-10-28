"""
ONNXOmniModel against the float32 reference: prompts (chat template, image and speech placeholder
expansion, pixels) equal the processor's, and greedy generation through each bundle's graphs
(images in, text or interleaved text and speech out, MaskGIT orders injected) follows the
reference's generate.

ASR, TTS, speech-to-speech and the TTS round trip are tests/test_lfm3_audio/test_runtime.py's, run
with --lfm3-audio-model LiquidAI/LFM3-Omni-2B.

Run with:
    uv run --no-sync pytest tests/test_lfm3_omni/test_generation.py -v --device cuda
    uv run pytest tests/test_lfm3_omni/test_generation.py -v
"""

import logging
import re
import types

import numpy as np
import pytest
import torch

from liquidonnx.lfm3_audio.session import load_audio, rng_orders
from liquidonnx.lfm3_omni.session import ONNXOmniModel
from liquidonnx.lfm3_vl.session import preprocess_images

from ..test_lfm3_audio.remote import inject_orders, maskgit_orders
from ..test_lfm3_audio.test_decoder import SAMPLES
from ..test_lfm3_audio.test_runtime import AUDIO_OUT, TF32_PRECISIONS, pattern, text_of
from .conftest import load_images

logger = logging.getLogger(__name__)

PRECISIONS = ["fp32", "fp16", "q8", "q8f16", "q4", "q4f16"]
QUESTION = "What bird is this? Answer in one sentence."
INTERLEAVED = "Respond with interleaved text and audio."
SEED = 20261024


def user(images: int, clips: int, text: str) -> dict:
    content = [{"type": "image"}] * images + [{"type": "audio"}] * clips
    return {"role": "user", "content": [*content, {"type": "text", "text": text}]}


# name -> (messages, images, clips, continue_final_message)
CONVERSATIONS = {
    "image": ([user(1, 0, QUESTION)], ["cardinal.jpg"], [], False),
    "two-images": ([user(2, 0, "What birds are these?")], ["cardinal.jpg", "bluejay.jpg"], [], False),
    "image-speech": (
        [user(1, 1, "Describe the image, then answer the question.")],
        ["bluejay.jpg"],
        ["woodworks_question.wav"],
        False,
    ),
    "image-interleaved": (
        [
            {"role": "system", "content": INTERLEAVED},
            user(1, 0, QUESTION),
            {"role": "assistant", "content": "<|mixed_start|>"},
        ],
        ["cardinal.jpg"],
        [],
        True,
    ),
}  # fmt: skip


def processor_inputs(processor, name: str) -> dict:
    messages, images, clips, continue_final = CONVERSATIONS[name]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
    )
    kwargs = {"images": load_images(images)}
    if clips:
        kwargs |= {
            "audio": [load_audio(SAMPLES / clip, 16000) for clip in clips],
            "sampling_rate": 16000,
        }
    return processor(text=text, return_tensors="pt", **kwargs)


def onnx_prompt(model, name: str, grids: np.ndarray) -> list[int]:
    messages, _, clips, continue_final = CONVERSATIONS[name]
    mel = model.config.mel
    counts = [
        mel.num_tokens(mel.num_frames(len(load_audio(SAMPLES / clip, 16000)))) for clip in clips
    ]
    return model.prompt_ids(
        messages,
        counts,
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
        grids=grids,
    )


@pytest.mark.parametrize("name", list(CONVERSATIONS))
def test_prompt_ids(checkpoint, reference_processor, name):
    """Token ids and pixels of ONNXOmniModel's preprocessing equal the Lfm3OmniProcessor's."""
    model = ONNXOmniModel(checkpoint)
    model.load_preprocessing()
    expected = processor_inputs(reference_processor, name)
    pixels, grids = preprocess_images(model.image_processor, load_images(CONVERSATIONS[name][1]))
    assert np.array_equal(grids, expected["image_grid_thw"].numpy())
    assert np.abs(pixels - expected["pixel_values"].numpy()).max() == 0
    assert onnx_prompt(model, name, grids) == expected["input_ids"][0].tolist()


def media(inputs: dict, device) -> dict:
    return {k: v.to(device) for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}


@pytest.fixture(scope="module")
def references(reference, reference_processor, device):
    """The reference's outputs, computed once per argument set (a CPU decode step is slow where
    torch runs the depthwise convolutions group by group):
    generate(name, **kwargs) -> (prompt ids, generated slots, frames [F, 12]) of CONVERSATIONS[name]
    (MaskGIT orders of SEED, end-of-audio frames kept), and logits(name, tokens) -> the logits
    [len(tokens), V] predicting tokens after the prompt, teacher-forced in one forward."""
    cache = {}

    def inputs_of(name):
        if name not in cache:
            cache[name] = processor_inputs(reference_processor, name)
        return cache[name]

    def generate(name: str, **kwargs):
        key = (name, *sorted(kwargs.items()))
        if key not in cache:
            inputs = inputs_of(name)
            ids = inputs["input_ids"][0].tolist()
            with torch.no_grad(), inject_orders(maskgit_orders(SEED, 4096)):
                out = reference.generate(
                    inputs["input_ids"].to(device),
                    **media(inputs, device),
                    return_audio_eos=True,
                    **kwargs,
                )
            tokens = out.sequences[0, len(ids) : int(out.sequence_lengths[0])].tolist()
            frames = out.audio_codes[0, :, : int(out.audio_lengths[0])].T.cpu().numpy()
            cache[key] = ids, tokens, frames
        return cache[key]

    def logits(name: str, tokens: list[int]) -> np.ndarray:
        key = (name, tuple(tokens))
        if key not in cache:
            inputs = inputs_of(name)
            ids = inputs["input_ids"][0].tolist()
            sequence = torch.tensor([ids + tokens[:-1]], device=device)
            with torch.no_grad():
                hidden = reference.model(
                    input_ids=sequence, **media(inputs, device), use_cache=False
                )
                cache[key] = (
                    reference.lm_head(hidden.last_hidden_state[0, len(ids) - 1 :])
                    .float()
                    .cpu()
                    .numpy()
                )
        return cache[key]

    return types.SimpleNamespace(generate=generate, logits=logits)


def assert_image_answer(references, model, precision: str):
    suppress = {"suppress_mixed": True, "suppress_audio_out": True}
    ids, expected, _ = references.generate(
        "image", max_new_tokens=40, generation_mode="text", **suppress
    )

    features, grids = model.encode_images(load_images(["cardinal.jpg"]))
    assert onnx_prompt(model, "image", grids) == ids
    tokens = model.generate_ids(ids, image_features=features, max_new_tokens=40, **suppress).tokens

    top5 = np.argsort(-references.logits("image", tokens), axis=-1)[:, :5]
    in_top5 = np.array([token in row for token, row in zip(tokens, top5, strict=True)])
    logger.info(
        f"  {precision}: {model.tokenizer.decode(tokens)!r} (reference "
        f"{model.tokenizer.decode(expected)!r}), {in_top5.sum()}/{len(tokens)} in the top 5"
    )
    if precision in ("fp32", "fp16"):
        assert tokens == expected
    assert in_top5.all(), f"tokens outside the reference top 5 at steps {np.flatnonzero(~in_top5)}"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_image_answer(references, onnx_models, precision):
    """Greedy text answer to an image question (text mode, mode markers suppressed): fp32 and
    fp16 give the reference's tokens; the others a token of the reference's top 5 at every step,
    teacher-forced on their own answer."""
    assert_image_answer(references, onnx_models(precision), precision)


@pytest.mark.slow
@pytest.mark.parametrize("precision", TF32_PRECISIONS)
def test_image_answer_tf32(references, onnx_models, ep, precision):
    """test_image_answer on lfm3-omni-infer's default runtime on CUDA, TF32 on. The speech side's
    TF32 runs are test_lfm3_audio's test_onnx_generation_tf32."""
    if ep != "cuda":
        pytest.skip("TF32 is a CUDA EP option")
    assert_image_answer(references, onnx_models(precision, tf32=True), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_image_interleaved(references, onnx_models, precision):
    """Interleaved text and speech about an image, greedy with injected MaskGIT orders: fp32 gives
    the reference's slots and codes exactly; every bundle follows the interleaved schedule: 6 text
    slots and 12 frames until its text ends (<|text_end|>, in the last text block), then speech to
    the end-of-audio frame, then <|im_end|>. The reference's may be cut off at the slot cap;
    quantized bundles may word the answer shorter."""
    slots = 120
    ids, ref_tokens, ref_frames = references.generate("image-interleaved", max_new_tokens=slots)

    model = onnx_models(precision)
    features, _ = model.encode_images(load_images(["cardinal.jpg"]))
    generation = model.generate_ids(
        ids,
        image_features=features,
        max_new_tokens=slots,
        orders=rng_orders(SEED),
        return_audio_eos=True,
    )
    tokens, frames = generation.tokens, generation.audio_codes
    same = np.mean([a == b for a, b in zip(tokens, ref_tokens, strict=False)])
    count = min(len(frames), len(ref_frames))
    codes = np.mean(frames[:count] == ref_frames[:count]) if count else 1.0
    logger.info(
        f"  {precision}: {pattern(tokens)} (reference {pattern(ref_tokens)}), slot agreement "
        f"{same:.3f}, code agreement {codes:.3f}: {text_of(model.tokenizer, tokens)!r}"
    )
    schedule = r"(T6 A12 )*T[1-6] A\d+( T1)?"
    assert re.fullmatch(schedule, pattern(ref_tokens))
    if precision == "fp32":
        assert tokens == ref_tokens
        assert np.array_equal(frames, ref_frames)
    assert re.fullmatch(schedule, pattern(tokens)) and AUDIO_OUT in tokens
