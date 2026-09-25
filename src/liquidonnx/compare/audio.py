"""LFM2.5-Audio exports against liquid-audio (needs the dev extra).

Both sides take the audio codes greedily (audio_top_k=1), so that they can be compared at all.
"""

import logging
import pathlib
import time

import numpy as np
import onnxruntime_genai as og

from liquidonnx.compare.metrics import guarded, logit_metrics, merge_metrics, sequence_match
from liquidonnx.genai_runtime import generate, load_model
from liquidonnx.lfm2_audio.export import AUDIO_TOKEN_ID, PRECISIONS, bundle, genai_files
from liquidonnx.lfm2_audio.infer import NUM_CODEBOOKS, SYSTEM_PROMPTS, chat_prompt, user_content
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import cached_outputs, load_onnx_session

logger = logging.getLogger(__name__)

SAMPLES = pathlib.Path(__file__).parents[3] / "samples/audio"
END_OF_AUDIO = 2048
CASES = [
    # (name, system prompt, user text, clip, interleaved)
    ("asr_short", SYSTEM_PROMPTS["asr"], None, "woodworks_question.wav", False),
    ("asr_long", SYSTEM_PROMPTS["asr"], None, "fool_me_once_mono.wav", False),
    ("chat_audio", None, None, "woodworks_question.wav", False),
    ("tts", SYSTEM_PROMPTS["tts"], "The quick brown fox jumps over the lazy dog.", None, False),
    ("interleaved", SYSTEM_PROMPTS["interleaved"], None, "woodworks_question.wav", True),
]


def precisions(export: pathlib.Path) -> list[str]:
    onnx_dir = export / "onnx"
    return [
        p for p in ("fp32", *PRECISIONS) if all((onnx_dir / f).exists() for f in bundle(p).values())
    ]


def reference_answer(
    model,
    processor,
    system: str | None,
    text: str | None,
    clip: str | None,
    interleaved: bool,
    max_new: int,
):
    """liquid-audio's answer to one user turn, text greedy and audio codes greedy.

    Returns the text tokens, the audio frames [F, 8] without the end-of-audio frames (which
    onnxruntime-genai leaves out too) and the prompt embeddings [S, H].
    """
    import soundfile
    import torch
    from liquid_audio import ChatState

    state = ChatState(processor, dtype=torch.float32)
    if system:
        state.new_turn("system")
        state.add_text(system)
        state.end_turn()
    state.new_turn("user")
    if clip:
        samples, rate = soundfile.read(str(clip), dtype="float32", always_2d=True)
        state.add_audio(torch.from_numpy(samples.mean(1))[None], rate)
    if text:
        state.add_text(text)
    state.end_turn()
    state.new_turn("assistant")

    run = model.generate_interleaved if interleaved else model.generate_sequential
    prefill = ("text", "audio_in", "audio_in_lens", "audio_out", "modality_flag")
    with torch.no_grad():
        prompt_embeds = model._prefill(**{k: state[k] for k in prefill})[0]
        stream = [
            item.cpu().numpy().copy()
            for item in run(**state, max_new_tokens=max_new, audio_temperature=0.0, audio_top_k=1)
        ]
    tokens = [int(t[0]) for t in stream if t.size == 1]
    frames = [t for t in stream if t.size > 1 and t[0] != END_OF_AUDIO]
    return tokens, np.array(frames, np.int64).reshape(-1, NUM_CODEBOOKS), prompt_embeds


def reference(checkpoint: pathlib.Path, max_new: int) -> list[dict]:
    """liquid-audio's answers to CASES and, for the text-only answers, the prompt embeddings,
    logits and hidden states that predict each answer token."""
    import torch
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    model = LFM2AudioModel.from_pretrained(checkpoint, dtype=torch.float32, device="cpu").eval()
    processor = LFM2AudioProcessor.from_pretrained(checkpoint, device="cpu")
    items = []
    for name, system, text, clip, interleaved in CASES:
        start = time.time()
        path = SAMPLES / clip if clip else None
        tokens, frames, prompt_embeds = reference_answer(
            model, processor, system, text, path, interleaved, max_new
        )
        item = {
            "name": name,
            "prompt": chat_prompt(system, [("user", user_content(clip, text))]),
            "clip": clip,
            "interleaved": interleaved,
            "tokens": tokens,
            "frames": frames,
            "text": processor.text.decode(tokens),
        }
        if not len(frames):
            with torch.no_grad():
                answer = model.lfm.embed_tokens(torch.tensor(tokens, dtype=torch.long))
                embeds = torch.cat([prompt_embeds, answer])
                hidden = model.lfm(inputs_embeds=embeds[None], use_cache=False).last_hidden_state
                logits = torch.nn.functional.linear(hidden[0], model.lfm.embed_tokens.weight)
            n = prompt_embeds.shape[0]
            item.update(
                embeds=embeds.numpy(),
                n_prompt=n,
                logits=logits[n - 1 : -1].numpy(),
                hidden=hidden[0, n - 1 : -1].numpy(),
            )
        logger.info(
            f"reference {name}: {len(tokens)} tokens, {len(frames)} frames, "
            f"{time.time() - start:.1f}s"
        )
        items.append(item)
    return items


def score_decoder(path: pathlib.Path, refs: list[dict]) -> dict:
    """The decoder fed liquid-audio's embeddings of the text-only answers."""
    session = load_onnx_session(path, ["CPUExecutionProvider"])
    forced, hidden_diff = [], 0.0
    for ref in refs:
        if "embeds" not in ref:
            continue
        embeds = ref["embeds"][None, :-1]
        got = cached_outputs(session, embeds, ref["n_prompt"], ("logits", "hidden_states"))
        forced.append(logit_metrics(ref["logits"], got["logits"]))
        hidden_diff = max(hidden_diff, float(np.abs(got["hidden_states"] - ref["hidden"]).max()))
    return {
        "teacher_forced": merge_metrics(forced),
        "hidden_max_abs": hidden_diff,
        "size_mb": get_total_model_size_mb(path),
    }


def score_genai(export: pathlib.Path, precision: str, refs: list[dict], max_new: int) -> dict:
    model = load_model(export, genai_files(precision))
    processor = model.create_multimodal_processor()
    cases = {}
    for ref in refs:
        clips = [str(SAMPLES / ref["clip"])] if ref["clip"] else []
        inputs = processor(ref["prompt"], audios=og.Audios.open(*clips) if clips else None)
        n = inputs["input_ids"].as_numpy().shape[-1]
        generator = generate(
            model,
            inputs,
            max_new,
            audio_interleaved=ref["interleaved"],
            audio_temperature=0.0,
            audio_top_k=1,
        )
        tokens = [int(t) for t in generator.get_sequence(0)[n:] if t != AUDIO_TOKEN_ID]
        codes = np.asarray(generator.get_output("audio_codes")).reshape(-1, NUM_CODEBOOKS)
        equal = sum(np.array_equal(a, b) for a, b in zip(ref["frames"], codes, strict=False))
        cases[ref["name"]] = {
            "text": sequence_match(ref["tokens"], tokens, {7}),
            "frames_equal": int(equal),
            "frames": len(ref["frames"]),
            "genai_frames": len(codes),
        }
    return cases


def score(export: pathlib.Path, precision: str, refs: list[dict], genai: bool, max_new: int):
    files = bundle(precision)
    row = {"decoder": guarded(score_decoder, export / "onnx" / files["decoder"], refs)}
    if genai:
        row["genai"] = guarded(score_genai, export, precision, refs, max_new)
    return row
