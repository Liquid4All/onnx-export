# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference kernels for LFM3's width-7 depthwise convolution."""

import torch

from transformers.utils import is_triton_available


if is_triton_available():
    import triton
    import triton.language as tl

    # the length and strides are runtime arguments, or every prompt length would compile its own kernel
    @triton.jit(do_not_specialize=["T", "X0", "X1", "X2", "S0", "S1", "ST"])
    def _prefill(
        X,
        W,
        BIAS,
        SEQ,
        Y,
        T,
        D: tl.constexpr,
        X0,
        X1,
        X2,
        S0,
        S1,
        ST,
        HAS_BIAS: tl.constexpr,
        PACKED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # int64 offsets: batch * channels * length can exceed 2**31 elements
        b, d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
        t = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.full((BLOCK,), 0, tl.float32)
        if PACKED:
            doc = tl.load(SEQ + b * S0 + (t - T + ST) * S1, (t < T) & (t >= T - ST), other=-1)
        for k in tl.static_range(7):
            pos = t + k - 6
            valid = (t < T) & (pos >= 0)
            if PACKED:
                prev = tl.load(SEQ + b * S0 + (pos - T + ST) * S1, valid & (pos >= T - ST), other=-1)
                valid = valid & (doc == prev)
            x = tl.load(X + b * X0 + d * X1 + pos * X2, valid, other=0).to(tl.float32)
            w = tl.load(W + d * 7 + k).to(tl.float32)
            acc += x * w
        if HAS_BIAS:
            acc += tl.load(BIAS + d).to(tl.float32)
        tl.store(Y + (b * D + d) * T + t, acc, t < T)

    @triton.jit
    def _decode(
        X,
        W,
        BIAS,
        STATE,
        Y,
        D: tl.constexpr,
        K: tl.constexpr,
        X0: tl.constexpr,
        X1: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        S2: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        k = tl.arange(0, 8)
        # Load the complete state before writing the shifted window.
        old = tl.load(STATE + b * S0 + d[:, None] * S1 + k[None, :] * S2, (d[:, None] < D) & (k[None, :] < K), other=0)
        x = tl.load(X + b * X0 + d * X1, d < D, other=0)
        shifted = tl.gather(old, tl.broadcast_to(tl.minimum(k + 1, 7)[None, :], (BLOCK, 8)), 1)
        shifted = tl.where(k[None, :] == K - 1, x[:, None], shifted)
        # A cache may store either K=6 previous inputs or K=7 including the oldest discarded input.
        positions = k + K - 6
        values = tl.gather(old, tl.broadcast_to(tl.minimum(positions, 7)[None, :], (BLOCK, 8)), 1)
        values = tl.where(k[None, :] == 6, x[:, None], values)
        weights = tl.load(W + d[:, None] * 7 + k[None, :], (d[:, None] < D) & (k[None, :] < 7), other=0)
        acc = tl.sum(values.to(tl.float32) * weights.to(tl.float32), 1)
        if HAS_BIAS:
            acc += tl.load(BIAS + d, d < D, other=0).to(tl.float32)
        tl.store(STATE + b * S0 + d[:, None] * S1 + k[None, :] * S2, shifted, (d[:, None] < D) & (k[None, :] < K))
        tl.store(Y + b * D + d, acc, d < D)


def can_use_lfm3_conv(hidden_states, weight):
    return (
        is_triton_available()
        and hasattr(tl, "gather")  # the decode kernel needs Triton >= 3.3
        and hidden_states.is_cuda
        and not torch.is_grad_enabled()
        and hidden_states.dtype == weight.dtype
        and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.shape[-1] == 7
    )


def lfm3_causal_conv1d(hidden_states, weight, bias=None, seq_idx=None):
    batch, channels, length = hidden_states.shape
    # the kernels index every tensor by the batch and channel of `hidden_states` and cannot bound the others
    if weight.shape != (channels, 7) or (bias is not None and bias.shape != (channels,)):
        raise ValueError(
            f"`weight` must have shape ({channels}, 7) and `bias` ({channels},), got {tuple(weight.shape)}"
        )
    if seq_idx is not None and (
        seq_idx.ndim != 2 or seq_idx.shape[0] not in (1, batch) or seq_idx.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            f"`seq_idx` must be an int32 or int64 tensor of shape (1, length) or ({batch}, length), got "
            f"{seq_idx.dtype} {tuple(seq_idx.shape)}"
        )
    if any(t is not None and t.device != hidden_states.device for t in (weight, bias, seq_idx)):
        raise ValueError(
            f"`weight`, `bias` and `seq_idx` must be on the device of `hidden_states` ({hidden_states.device})"
        )
    output = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    # Triton launches on the current device; the model may live on another one (`device_map`, `.to("cuda:1")`).
    with torch.cuda.device(hidden_states.device):
        _prefill[(batch, channels, triton.cdiv(length, 128))](
            hidden_states,
            # the kernels index `bias` and `weight` densely
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            seq_idx,
            output,
            length,
            channels,
            *hidden_states.stride(),
            *(
                (0 if seq_idx.shape[0] == 1 else seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            seq_idx.shape[-1] if seq_idx is not None else 0,
            bias is not None,
            seq_idx is not None,
            128,
            enable_fp_fusion=False,
        )
    return output


def lfm3_causal_conv1d_update(hidden_states, conv_state, weight, bias=None):
    batch, channels, length = hidden_states.shape
    # the kernel addresses one position per row and bounds nothing by the other tensors' shapes
    if length != 1:
        raise ValueError(f"`hidden_states` must hold one position per row, got {tuple(hidden_states.shape)}")
    # the 8-wide tile holds a 6- or 7-wide state (see `positions` in `_decode`)
    if (
        conv_state.ndim != 3
        or conv_state.shape[:2] != (batch, channels)
        or conv_state.shape[-1] not in (6, 7)
        or weight.shape != (channels, 7)
    ):
        raise ValueError(
            f"`conv_state` must have shape ({batch}, {channels}, 6 or 7) and `weight` ({channels}, 7), got "
            f"{tuple(conv_state.shape)} and {tuple(weight.shape)}"
        )
    if bias is not None and bias.shape != (channels,):
        raise ValueError(f"`bias` must have shape ({channels},), got {tuple(bias.shape)}")
    # the kernel writes the state in place: rows or channels sharing memory (an `expand`ed cache) would race
    extent = 0
    for stride, size in sorted(zip(conv_state.stride(), conv_state.shape)):
        if size > 1 and stride <= extent:
            raise ValueError(f"`conv_state` must not overlap itself, got strides {conv_state.stride()}")
        extent += stride * (size - 1)
    if any(t is not None and t.device != hidden_states.device for t in (conv_state, weight, bias)):
        raise ValueError(
            f"`conv_state`, `weight` and `bias` must be on the device of `hidden_states` ({hidden_states.device})"
        )
    output = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    with torch.cuda.device(hidden_states.device):
        _decode[(batch, triton.cdiv(channels, 32))](
            hidden_states,
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            conv_state,
            output,
            channels,
            conv_state.shape[-1],
            *hidden_states.stride()[:2],
            *conv_state.stride(),
            bias is not None,
            32,
            enable_fp_fusion=False,
        )
    return output
