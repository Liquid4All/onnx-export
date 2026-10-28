# Changed from LiquidAI/d1-2B at 97c57047d5852d77e02dfac211f8768aff77a8b2: timing and hardware notes cut.
"""LFM's depthwise causal convolution, the same numbers in every process: on ROCm as shifted sums, never MIOpen's
kernel, whose solver search runs again for every new length."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def on_rocm(t: torch.Tensor) -> bool:
    return t.is_cuda and bool(torch.version.hip)


def depthwise(padded: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise conv, `(B, C, T + k - 1)` to `(B, C, T)`. On ROCm, `shifted_sum`: MIOpen's depthwise kernel
    searches a solver for every new length, logs a warning each time, may settle on another one in another process."""
    if not on_rocm(padded):
        return F.conv1d(padded, weight, groups=padded.shape[1])
    return shifted_sum(padded, weight)


def shifted_sum(padded: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The depthwise conv as k shifted products summed in fp32: one fp32 copy of `padded`, the shifts views of it."""
    k, t = weight.shape[-1], padded.shape[-1] - weight.shape[-1] + 1
    p, w = padded.float(), weight[:, 0, :, None].float()
    return sum(p[..., i:i + t] * w[:, i] for i in range(k)).to(padded.dtype)


def pinned_conv(conv: torch.nn.Conv1d | torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """`conv(x)` without MIOpen on ROCm: a 1x1 conv as the matmul it is, a 2D conv with one input channel a group
    (`groups` = input channels) as `shifted_sum2d`. MIOpen's solver for the omni downsampler's 1x1 `lin_out` changed
    from one call to the next within a process. Elsewhere, and for any other conv, `conv`'s own forward."""
    if on_rocm(x) and conv.padding_mode == "zeros" and set(conv.dilation) == {1}:
        if set(conv.kernel_size) == {1} and set(conv.stride) == {1} and conv.padding in ((0,), (0, 0)) and conv.groups == 1:
            return F.linear(x.movedim(1, -1), conv.weight.flatten(1), conv.bias).movedim(-1, 1)
        if x.ndim == 4 and conv.groups == conv.in_channels and not isinstance(conv.padding, str):
            return shifted_sum2d(x, conv.weight, conv.bias, conv.stride, conv.padding)
    return type(conv).forward(conv, x)


def shifted_sum2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: tuple[int, int],
                  padding: tuple[int, int]) -> torch.Tensor:
    """A 2D conv with one input channel a group, `weight` `(C * m, 1, kh, kw)`, as kh x kw strided products summed in
    fp32: each of the C input channels feeds its m output channels."""
    (kh, kw), (sh, sw), (ph, pw) = weight.shape[-2:], stride, padding
    c = x.shape[1]
    ho, wo = (x.shape[-2] + 2 * ph - kh) // sh + 1, (x.shape[-1] + 2 * pw - kw) // sw + 1
    p = F.pad(x.float(), (pw, pw, ph, ph))[:, :, None]
    w = weight.float().view(c, -1, kh, kw)[..., None, None]
    out = sum(p[..., i:i + sh * (ho - 1) + 1:sh, j:j + sw * (wo - 1) + 1:sw] * w[:, :, i, j]
              for i in range(kh) for j in range(kw)).flatten(1, 2)
    return (out if bias is None else out + bias.float()[:, None, None]).to(x.dtype)
