"""Unit tests for the VLModelInference public API (no models, no downloads).

Serving consumers (e.g. model-pipeline's ONNX smoke server) import
`VLModelInference` from `liquidonnx.lfm2_vl` and rely on `force_cpu` and
`skip_special_tokens` behaving like `ONNXTextModel`'s.

Run: uv run pytest tests/test_lfm2_vl/test_infer_api.py -v
"""

from unittest.mock import Mock

import numpy as np
import pytest

from liquidonnx.lfm2_vl import VISION_MODE_TILED, VLModelInference, resolve_precision_files
from liquidonnx.lfm2_vl import infer as infer_module


def test_public_api_reexports_the_infer_module():
    assert VLModelInference is infer_module.VLModelInference
    assert resolve_precision_files is infer_module.resolve_precision_files


class TestForceCpu:
    def _load(self, tmp_path, monkeypatch, force_cpu: bool) -> list[list[str] | None]:
        """Run load() with stubbed processor/sessions; return the providers
        each of the three load_onnx_session calls received."""
        (tmp_path / "tokenizer.json").write_text("{}")

        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.return_value = 396
        processor = Mock()
        processor.tokenizer = tokenizer
        monkeypatch.setattr(
            infer_module.AutoProcessor, "from_pretrained", Mock(return_value=processor)
        )

        seen_providers = []

        def fake_load_onnx_session(path, providers=None):
            seen_providers.append(providers)
            return Mock()

        monkeypatch.setattr(infer_module, "load_onnx_session", fake_load_onnx_session)
        monkeypatch.setattr(
            infer_module, "detect_vision_format", Mock(return_value=VISION_MODE_TILED)
        )

        model = VLModelInference(str(tmp_path), force_cpu=force_cpu)
        model.load()
        return seen_providers

    def test_force_cpu_pins_every_session_to_cpu(self, tmp_path, monkeypatch):
        providers = self._load(tmp_path, monkeypatch, force_cpu=True)
        assert providers == [["CPUExecutionProvider"]] * 3

    def test_default_auto_detects_providers(self, tmp_path, monkeypatch):
        providers = self._load(tmp_path, monkeypatch, force_cpu=False)
        assert providers == [None] * 3


class TestGenerateSkipSpecialTokens:
    EOS = 5

    def _model(self) -> tuple[VLModelInference, Mock]:
        """A text-only model with stubbed tokenizer and sessions: the decoder
        emits EOS immediately, so generate() decodes a one-token response."""
        model = VLModelInference("unused")

        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.encode.return_value = [1, 2]
        tokenizer.eos_token_id = self.EOS
        tokenizer.decode.return_value = "response"
        model.tokenizer = tokenizer

        embed_tokens = Mock()
        # [1, seq_len, hidden] for the prompt, then [1, 1, hidden] per step.
        embed_tokens.run.side_effect = lambda _out, feed: [
            np.zeros((1, feed["input_ids"].shape[1], 4), dtype=np.float32)
        ]
        model.embed_tokens_sess = embed_tokens

        decoder = Mock()
        decoder.get_inputs.return_value = []  # no KV cache inputs, no position_ids
        decoder.get_outputs.return_value = []
        logits = np.zeros((1, 2, self.EOS + 1), dtype=np.float32)
        logits[0, -1, self.EOS] = 1.0  # argmax -> EOS: stop after one token
        decoder.run.return_value = [logits]
        model.decoder_sess = decoder
        return model, tokenizer

    @pytest.mark.parametrize("skip", [True, False])
    def test_final_decode_honors_the_flag(self, skip):
        model, tokenizer = self._model()
        out = model.generate(
            [{"role": "user", "content": "hi"}],
            stream=False,
            skip_special_tokens=skip,
        )
        assert out == "response"
        tokenizer.decode.assert_called_with([self.EOS], skip_special_tokens=skip)

    def test_default_strips_special_tokens(self):
        model, tokenizer = self._model()
        model.generate([{"role": "user", "content": "hi"}], stream=False)
        tokenizer.decode.assert_called_with([self.EOS], skip_special_tokens=True)
