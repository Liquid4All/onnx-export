"""
Multi-turn coherence testing utilities for LFM2 ONNX models.

Provides classes and functions for testing whether quantized models
maintain coherent multi-turn conversations compared to PyTorch ground truth.

Metrics:
- Token-level: exact match of generated tokens per turn
- Semantic: cosine similarity of logits per turn
- Accumulated error across turns
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

MODELS = {
    "350M": "LiquidAI/LFM2-350M",
    "700M": "LiquidAI/LFM2-700M",
    "1.2B": "LiquidAI/LFM2-1.2B",
    "2.6B": "LiquidAI/LFM2-2.6B",
}

DEFAULT_PROMPTS = [
    "My name is Sarah and I work as a software engineer. I have two cats named Luna and Milo. Can you remember these facts?",
    "What are the names of my cats?",
    "What is my profession?",
    "How many pets do I have?",
    "If Luna ran away, what kind of animal would I put on the lost poster?",
]


@dataclass
class TurnResult:
    """Result of a single conversation turn."""

    turn: int
    prompt: str
    pytorch_response: str
    onnx_response: str
    token_match_rate: float
    semantic_similarity: float
    max_logit_diff: float
    mean_logit_diff: float


@dataclass
class CoherenceResult:
    """Result of multi-turn coherence test."""

    model_size: str
    quant_type: str
    turns: list[TurnResult] = field(default_factory=list)
    avg_token_match: float = 0.0
    avg_semantic_sim: float = 0.0
    accumulated_error: float = 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class MultiTurnTester:
    """Tests multi-turn coherence of quantized models."""

    def __init__(self, pytorch_path: str, max_new_tokens: int = 20):
        self.pytorch_path = pytorch_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.torch_model = None
        self.onnx_session = None

    def load_pytorch(self):
        """Load PyTorch model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading PyTorch model: {self.pytorch_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.pytorch_path, trust_remote_code=True)
        self.torch_model = AutoModelForCausalLM.from_pretrained(
            self.pytorch_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.torch_model.eval()

    def load_onnx(self, onnx_path: str):
        """Load ONNX model."""
        import onnxruntime as ort

        logger.info(f"Loading ONNX model: {onnx_path}")
        self.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def generate_pytorch(
        self, input_ids: list[int], max_new_tokens: int
    ) -> tuple[list[int], np.ndarray]:
        """Generate tokens with PyTorch model using KV cache."""
        import torch

        generated = input_ids.copy()
        all_logits = []
        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    ids = torch.tensor([generated], dtype=torch.long)
                    pos = torch.arange(len(generated), dtype=torch.long).unsqueeze(0)
                    attn = torch.ones(1, len(generated), dtype=torch.long)
                else:
                    ids = torch.tensor([[generated[-1]]], dtype=torch.long)
                    pos = torch.tensor([[len(generated) - 1]], dtype=torch.long)
                    attn = torch.ones(1, len(generated), dtype=torch.long)

                outputs = self.torch_model(
                    input_ids=ids,
                    attention_mask=attn,
                    position_ids=pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1].numpy()
                all_logits.append(logits)
                next_token = int(np.argmax(logits))
                generated.append(next_token)

                if next_token == self.tokenizer.eos_token_id:
                    break

        return generated, np.stack(all_logits) if all_logits else np.array([])

    def generate_onnx(
        self, input_ids: list[int], max_new_tokens: int
    ) -> tuple[list[int], np.ndarray]:
        """Generate tokens with ONNX model."""
        sess = self.onnx_session
        generated = input_ids.copy()
        all_logits = []

        input_names = {inp.name for inp in sess.get_inputs()}
        has_position_ids = "position_ids" in input_names

        # Initialize caches
        cache = {}
        for inp in sess.get_inputs():
            if inp.name not in ["input_ids", "attention_mask", "position_ids"]:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                cache[inp.name] = np.zeros(shape, dtype=np.float32)

        outputs_info = sess.get_outputs()

        for step in range(max_new_tokens):
            cur_len = len(generated)

            if step == 0:
                ids = np.array([generated], dtype=np.int64)
                pos = np.arange(cur_len, dtype=np.int64).reshape(1, -1)
            else:
                ids = np.array([[generated[-1]]], dtype=np.int64)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": attn_mask}
            if has_position_ids:
                feed["position_ids"] = pos
            feed.update(cache)

            result = sess.run(None, feed)
            logits = result[0][0, -1]
            all_logits.append(logits)

            # Update caches
            for i, out_info in enumerate(outputs_info[1:], 1):
                out_name = out_info.name
                if "present_conv" in out_name:
                    cache_name = out_name.replace("present_conv", "past_conv")
                elif "present." in out_name:
                    cache_name = out_name.replace("present.", "past_key_values.")
                else:
                    continue
                if cache_name in cache:
                    cache[cache_name] = result[i]

            next_token = int(np.argmax(logits))
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        return generated, np.stack(all_logits) if all_logits else np.array([])

    def compare_turn(
        self,
        turn: int,
        prompt: str,
        messages_pytorch: list[dict],
        messages_onnx: list[dict],
    ) -> TurnResult:
        """Compare a single turn between PyTorch and ONNX."""
        messages_pytorch = messages_pytorch + [{"role": "user", "content": prompt}]
        messages_onnx = messages_onnx + [{"role": "user", "content": prompt}]

        pytorch_text = self.tokenizer.apply_chat_template(
            messages_pytorch, tokenize=False, add_generation_prompt=True
        )
        onnx_text = self.tokenizer.apply_chat_template(
            messages_onnx, tokenize=False, add_generation_prompt=True
        )

        pytorch_input = self.tokenizer.encode(pytorch_text, add_special_tokens=False)
        onnx_input = self.tokenizer.encode(onnx_text, add_special_tokens=False)

        pytorch_output, pytorch_logits = self.generate_pytorch(pytorch_input, self.max_new_tokens)
        onnx_output, onnx_logits = self.generate_onnx(onnx_input, self.max_new_tokens)

        # Extract new tokens only
        pytorch_new = pytorch_output[len(pytorch_input) :]
        onnx_new = onnx_output[len(onnx_input) :]

        # Token match rate
        min_len = min(len(pytorch_new), len(onnx_new))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if pytorch_new[i] == onnx_new[i])
            token_match_rate = matches / max(len(pytorch_new), len(onnx_new))
        else:
            token_match_rate = 1.0 if len(pytorch_new) == len(onnx_new) == 0 else 0.0

        # Semantic similarity
        if len(pytorch_logits) > 0 and len(onnx_logits) > 0:
            min_steps = min(len(pytorch_logits), len(onnx_logits))
            similarities = [
                cosine_similarity(pytorch_logits[i], onnx_logits[i]) for i in range(min_steps)
            ]
            semantic_similarity = np.mean(similarities)

            diffs = [np.abs(pytorch_logits[i] - onnx_logits[i]) for i in range(min_steps)]
            max_logit_diff = float(np.max([d.max() for d in diffs]))
            mean_logit_diff = float(np.mean([d.mean() for d in diffs]))
        else:
            semantic_similarity = 1.0
            max_logit_diff = 0.0
            mean_logit_diff = 0.0

        pytorch_response = self.tokenizer.decode(pytorch_new, skip_special_tokens=True)
        onnx_response = self.tokenizer.decode(onnx_new, skip_special_tokens=True)

        return TurnResult(
            turn=turn,
            prompt=prompt,
            pytorch_response=pytorch_response,
            onnx_response=onnx_response,
            token_match_rate=token_match_rate,
            semantic_similarity=semantic_similarity,
            max_logit_diff=max_logit_diff,
            mean_logit_diff=mean_logit_diff,
        )

    def test_coherence(
        self,
        model_size: str,
        onnx_path: str,
        quant_type: str,
        prompts: list[str],
    ) -> CoherenceResult:
        """Run multi-turn coherence test."""
        self.load_onnx(onnx_path)

        result = CoherenceResult(
            model_size=model_size,
            quant_type=quant_type,
        )

        messages_pytorch: list[dict] = []
        messages_onnx: list[dict] = []
        accumulated_error = 0.0

        for turn, prompt in enumerate(prompts, 1):
            turn_result = self.compare_turn(turn, prompt, messages_pytorch, messages_onnx)
            result.turns.append(turn_result)

            accumulated_error += 1.0 - turn_result.semantic_similarity

            messages_pytorch = messages_pytorch + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": turn_result.pytorch_response},
            ]
            messages_onnx = messages_onnx + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": turn_result.onnx_response},
            ]

        if result.turns:
            result.avg_token_match = np.mean([t.token_match_rate for t in result.turns])
            result.avg_semantic_sim = np.mean([t.semantic_similarity for t in result.turns])
            result.accumulated_error = accumulated_error

        return result
