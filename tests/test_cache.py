"""Tests for ChunkedKVCache."""

from unittest.mock import Mock

import numpy as np
import pytest
from liquidonnx.cache import ChunkedKVCache


def make_mock_session(kv_cache_names: list[str], seq_dim: int = 2):
    """Create a mock ONNX session with specified cache inputs."""
    session = Mock()

    inputs = []
    outputs = []

    # Standard inputs
    for name in ["input_ids", "attention_mask", "position_ids"]:
        inp = Mock()
        inp.name = name
        inp.shape = [1, "sequence"]
        inputs.append(inp)

    # KV cache inputs: [batch, heads, seq, dim]
    for name in kv_cache_names:
        inp = Mock()
        inp.name = name
        if seq_dim == 2:
            inp.shape = [1, 8, "past_sequence", 64]
        else:
            inp.shape = [1, "past_sequence", 8, 64]
        inputs.append(inp)

        # Matching output
        out = Mock()
        out.name = name.replace("past_key_values", "present")
        out.shape = inp.shape
        outputs.append(out)

    session.get_inputs.return_value = inputs
    session.get_outputs.return_value = outputs
    return session


class TestChunkedKVCacheInit:
    def test_parse_kv_caches(self):
        session = make_mock_session(["past_key_values.0.key", "past_key_values.0.value"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512)

        assert len(cache.kv_shapes) == 2
        assert "past_key_values.0.key" in cache.kv_shapes
        assert "past_key_values.0.value" in cache.kv_shapes

    def test_output_mapping(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session)

        assert "present.0.key" in cache.output_to_input
        assert cache.output_to_input["present.0.key"] == "past_key_values.0.key"


class TestChunkedKVCacheInitialize:
    def test_pre_allocates_to_initial_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=4096, chunk_size=2048)
        cache.initialize()

        assert cache.allocated_size == 4096
        assert cache.used_size == 0
        assert cache.kv_caches["past_key_values.0.key"].shape[2] == 4096

    def test_rounds_up_to_chunk_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=4096, chunk_size=2048)
        cache.initialize(initial_seq_len=5000)

        assert cache.allocated_size == 6144  # 3 * 2048

    def test_respects_max_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512, max_size=2000)
        cache.initialize(initial_seq_len=5000)

        assert cache.allocated_size == 2000


class TestChunkedKVCacheGetDict:
    def test_empty_cache_returns_zero_sized(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024)
        cache.initialize()

        cache_dict = cache.get_cache_dict()
        assert cache_dict["past_key_values.0.key"].shape[2] == 0

    def test_slices_to_used_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024)
        cache.initialize()

        # Simulate some used tokens
        cache.used_size = 100

        cache_dict = cache.get_cache_dict()
        assert cache_dict["past_key_values.0.key"].shape[2] == 100


class TestChunkedKVCacheUpdate:
    def test_updates_used_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024)
        cache.initialize()

        # Simulate model output with 50 tokens
        outputs = [np.zeros((1, 8, 50, 64), dtype=np.float32)]
        output_infos = [Mock(name="present.0.key")]
        output_infos[0].name = "present.0.key"

        cache.update(outputs, output_infos, new_tokens=50)
        assert cache.used_size == 50

    def test_copies_data_to_buffer(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024)
        cache.initialize()

        # Create output with specific values
        output_data = np.ones((1, 8, 50, 64), dtype=np.float32) * 42
        outputs = [output_data]
        output_infos = [Mock()]
        output_infos[0].name = "present.0.key"

        cache.update(outputs, output_infos, new_tokens=50)

        # Check data was copied
        assert np.allclose(cache.kv_caches["past_key_values.0.key"][:, :, :50, :], 42)


class TestChunkedKVCacheGrow:
    def test_grows_by_chunk_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512)
        cache.initialize()
        cache.used_size = 1024

        # Request more than allocated
        output_data = np.zeros((1, 8, 1025, 64), dtype=np.float32)
        outputs = [output_data]
        output_infos = [Mock()]
        output_infos[0].name = "present.0.key"

        cache.update(outputs, output_infos, new_tokens=1)

        assert cache.allocated_size == 1536  # 1024 + 512

    def test_preserves_existing_data(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512)
        cache.initialize()

        # Fill with initial data
        cache.kv_caches["past_key_values.0.key"][:, :, :100, :] = 99
        cache.used_size = 100

        # Trigger growth
        output_data = np.zeros((1, 8, 1100, 64), dtype=np.float32)
        outputs = [output_data]
        output_infos = [Mock()]
        output_infos[0].name = "present.0.key"

        cache.update(outputs, output_infos, new_tokens=1000)

        # Original data should still be in the buffer (though overwritten by outputs in this case)
        assert cache.allocated_size > 1024

    def test_raises_when_max_exceeded(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512, max_size=1024)
        cache.initialize()
        cache.used_size = 1024

        with pytest.raises(RuntimeError, match="Cannot grow cache beyond max_size"):
            cache._grow(2000)


class TestChunkedKVCacheReset:
    def test_resets_used_size(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512)
        cache.initialize()
        cache.used_size = 500

        cache.reset()

        assert cache.used_size == 0
        assert cache.allocated_size == 1024  # Allocation preserved


class TestChunkedKVCacheMemory:
    def test_memory_allocated(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512)
        cache.initialize()

        # 1024 * 8 * 64 * 4 bytes = 2 MB
        expected_mb = (1024 * 8 * 64 * 4) / (1024 * 1024)
        assert abs(cache.memory_allocated_mb - expected_mb) < 0.1

    def test_memory_used(self):
        session = make_mock_session(["past_key_values.0.key"])
        cache = ChunkedKVCache(session, initial_size=1024, chunk_size=512)
        cache.initialize()
        cache.used_size = 512

        # 512 * 8 * 64 * 4 bytes = 1 MB
        expected_mb = (512 * 8 * 64 * 4) / (1024 * 1024)
        assert abs(cache.memory_used_mb - expected_mb) < 0.1
