"""
Tokenizer compatibility checks for LFM2-MoE models.

Validates tokenizer/chat-template behavior that MoE export and inference rely on.

Run with:
    uv run pytest tests/test_lfm2_moe/test_tokenizer.py -v
"""

import pathlib
import tempfile

import pytest
from transformers import AutoConfig

MODELS = [
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2.5-8B-A1B",
]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello and use a tool if needed."},
]

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


@pytest.mark.parametrize("model_tokenizer", MODELS, indirect=True)
def test_tokenizer_embedding_alignment(model_tokenizer):
    """Tokenizer IDs and template output must fit the embedding table."""
    model_id, tokenizer = model_tokenizer
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
    )
    encoded = tokenizer(prompt, add_special_tokens=False)
    input_ids = encoded["input_ids"]

    assert input_ids, f"{model_id}: chat template encoded to an empty prompt"
    assert max(input_ids) < config.vocab_size
    assert len(tokenizer) <= config.vocab_size
    assert tokenizer.bos_token_id in input_ids
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None
    assert tokenizer.bos_token_id < config.vocab_size
    assert tokenizer.eos_token_id < config.vocab_size
    assert tokenizer.pad_token_id < config.vocab_size


@pytest.mark.parametrize("model_tokenizer", MODELS, indirect=True)
def test_tokenizer_roundtrip(model_tokenizer):
    """Saved tokenizers must preserve chat-template and tokenization behavior."""
    model_id, tokenizer = model_tokenizer

    prompt = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
    )
    expected_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    with tempfile.TemporaryDirectory(prefix="lfm2-moe-tokenizer-") as tmp:
        tmp_path = pathlib.Path(tmp)
        tokenizer.save_pretrained(tmp_path)

        from transformers import AutoTokenizer

        reloaded = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=True)
        reloaded_prompt = reloaded.apply_chat_template(
            MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            tools=TOOLS,
        )
        reloaded_ids = reloaded(reloaded_prompt, add_special_tokens=False)["input_ids"]

    assert reloaded_prompt == prompt, f"{model_id}: chat template changed after save/load"
    assert reloaded_ids == expected_ids, f"{model_id}: tokenization changed after save/load"
