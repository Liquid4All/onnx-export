# Changed from LiquidAI/d1-2B at 97c57047d5852d77e02dfac211f8768aff77a8b2: timing and hardware notes cut.
"""LFM3: text (`lfm3`), vision-language (`lfm3_vl`) and omni (`lfm3_omni`) checkpoints on transformers' classes
(`native_lfm3*`), loaded as their Hub class (`modeling_d1_lfm3*.D1Model`) with d1's attention. d1 adds only:

- the masks: `attention.d1_attention` under `attn_implementation="d1"`, its mask function registered;
- the short convolution's torch path as `depthwise.depthwise` (ROCm: no MIOpen search), taken with autograd and
  under torch.compile (`lfm3_conv`); eager inference keeps the classes' fused kernel; the audio encoder's
  convolutions off MIOpen on ROCm (`lfm3_conv`);
- the media encoders `SystemOne.compile` compiles beside the language model (`compiled_parts`), and its inductor
  options (`compile_options`); the vision tower in one graph for every picture size (`lfm3_tower`).
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import sys

import torch
import torch.nn.functional as F

from .depthwise import depthwise, pinned_conv
from .modeling_d1 import SystemOneModel
from .prompt import LEADS

MODULES = {"lfm3": "modeling_d1_lfm3", "lfm3_vl": "modeling_d1_lfm3_vl", "lfm3_omni": "modeling_d1_lfm3_omni"}
# the templates open a reasoning block; a non-thinking reply follows it closed at once
LEADS.update(dict.fromkeys(MODULES, "<think></think>"))


class Lfm3SystemOne(SystemOneModel):
    """What every LFM3 Hub class adds to transformers': d1's attention and its convolution."""

    attention = "d1"
    compiled_parts: tuple[str, ...] = ()  # the media encoders `SystemOne.compile` compiles beside the language model
    # compiled for inference with the weights as constants (inductor's freezing)
    compile_options = {"freezing": True}
    # the patch grids stay on the host: the tower and the image features read their sizes there, without a sync
    host_inputs = ("image_grid_thw", "video_grid_thw")

    def __init__(self, config):
        super().__init__(config)
        lfm3_conv(self)
        lfm3_tower(self)

    def fuse_projections(self) -> None:
        """For inference (`SystemOne`): the linears that read one input as one matmul (`fuse_projections`)."""
        fuse_projections(self)


def lfm3_conv(model) -> None:
    """Each LFM3 short convolution's torch path (`Lfm3ShortConv._conv`) as `depthwise`, and that path under
    torch.compile: inductor fuses its shifted sums with the gates around them, where the classes' fused kernel is a
    call of its own. Eager inference keeps the kernel.

    The omni's audio encoder, off MIOpen on ROCm as well: its LFM2 convolutions (transformers' `Lfm2ShortConv`, which
    calls its modeling module's `causal_conv1d_fn`, `F.conv1d` without causal-conv1d) as `depthwise`, its downsampler's
    as `pinned_conv`. MIOpen's solver for the downsampler's 1x1 `lin_out` changed from one call to the next, and a
    clip's P(yes) by 3.1e-03."""
    for m in model.modules():
        module = sys.modules[type(m).__module__]
        if type(m).__name__ == "Lfm3ShortConv":
            m._conv = functools.partial(_causal_depthwise, m.conv)
            if not getattr(module.can_use_lfm3_conv, "eager_only", False):
                module.can_use_lfm3_conv = _eager_only(module.can_use_lfm3_conv)
        elif type(m).__name__ == "Lfm2ShortConv" and importlib.util.find_spec("causal_conv1d") is None:
            module.causal_conv1d_fn = _lfm2_conv
        elif type(m).__name__ == "Lfm3OmniDownsampler":
            for conv in m.children():
                if isinstance(conv, torch.nn.modules.conv._ConvNd):
                    conv.forward = functools.partial(pinned_conv, conv)


def lfm3_tower(model) -> None:
    """The vision tower under torch.compile, one graph for every picture size and no graph break:

    - what its forward derives from the patch grids (positions, `cu_seqlens`, the frames' sizes: a sync and a graph
      break each), made before it on the host once a grid, the way transformers' `vision_utils` takes them
      precomputed; the frames' sizes come in as the grid itself, a list (`_frame_shapes`);
    - its `len(set(frame sizes)) == 1` (one batch for frames of one size) compared by `==`: hashing the sizes makes
      torch.compile guard on their values, a recompile for every picture size.

    Delete both module patches when transformers' tower compares the sizes and takes the frames' sizes precomputed."""
    for m in model.modules():
        if type(m).__name__ == "Lfm3VlVisionModel":
            m.register_forward_pre_hook(_grid_inputs, with_kwargs=True)
            module = sys.modules[type(m).__module__]
            module.set, module.get_vision_frame_shapes = _distinct, _frame_shapes


def _grid_inputs(tower, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """The grid's positions, `cu_seqlens`, longest frame and frame sizes, made once a grid."""
    from transformers.vision_utils import get_max_seqlen, get_vision_cu_seqlens, get_vision_position_ids

    args = list(args)
    grid = (args.pop(1) if len(args) > 1 else kwargs["grid_thw"]).cpu()  # on the host already when served
    device = (args[0] if args else kwargs["pixel_values"]).device
    made = tower.__dict__.setdefault("d1_grid_inputs", {})
    key = (tuple(grid.flatten().tolist()), device)
    if key not in made:
        if len(made) >= 256:
            made.clear()
        cu_seqlens = get_vision_cu_seqlens(grid)
        made[key] = {"cu_seqlens": cu_seqlens.to(device), "max_seqlen": get_max_seqlen(cu_seqlens, tower.config),
                     "position_ids": get_vision_position_ids(grid, tower.spatial_merge_size).to(device),
                     "grid_thw": _grid_frames(grid)}
    kwargs.update(made[key])
    return tuple(args), kwargs


def _frame_shapes(grid) -> list[list[int]]:
    """`get_vision_frame_shapes`: the grid `_grid_inputs` passes is its frames' sizes already; a tower without the
    hook (transformers' own model) passes the tensor."""
    return grid if isinstance(grid, list) else _grid_frames(grid)


def _grid_frames(grid: torch.Tensor) -> list[list[int]]:
    return grid[:, 1:].repeat_interleave(grid[:, 0], dim=0).tolist()


# the linears each LFM3 module calls on one input, in the order its forward calls them
SHARED_INPUT = {"Lfm3Attention": ("q_proj", "k_proj", "v_proj", "gate_proj"), "Lfm3MLP": ("w1", "w3"),
                "Lfm3VlVisionAttention": ("q_proj", "k_proj", "v_proj"), "Lfm3VlMLP": ("w1", "w3")}


def fuse_projections(model) -> None:
    """Each module's linears on one input (`SHARED_INPUT`) as one matmul, its output split: the first's call computes
    every part, the others read theirs, the last drops them. Exact, one weight copy, the classes untouched, the
    state dict the native one (a fused model saves as it loaded). For inference only: training keeps the separate
    linears, and a group with any layer not a plain linear stays."""
    for m in model.modules():
        names = SHARED_INPUT.get(type(m).__name__, ())
        linears = [getattr(m, n, None) for n in names]
        if names and all(type(lin) is torch.nn.Linear and lin.bias is None for lin in linears):
            m.fused = Fused(torch.cat([lin.weight for lin in linears]), names, [lin.out_features for lin in linears])
            for i, name in enumerate(names):
                setattr(m, name, Part(m.fused, i))
            m.register_state_dict_post_hook(_unfused)


def _unfused(module, state_dict, prefix, local_metadata) -> None:
    fused = module.fused
    weight = state_dict.pop(prefix + "fused.weight")
    for name, part in zip(fused.names, weight.split(fused.sizes)):
        state_dict[f"{prefix}{name}.weight"] = part.clone()


class Fused(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, names: tuple[str, ...], sizes: list[int]):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.names, self.sizes, self.parts = names, sizes, None


class Part(torch.nn.Module):
    """One linear of a `Fused` group: the first part computes the group's matmul."""

    def __init__(self, fused: Fused, index: int):
        super().__init__()
        self.group, self.index, self.last = [fused], index, index == len(fused.sizes) - 1  # a list: not a submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.group[0]
        if self.index == 0:
            fused.parts = F.linear(x, fused.weight).split(fused.sizes, dim=-1)
        out = fused.parts[self.index]
        if self.last:
            fused.parts = None
        return out


def _distinct(items) -> list:
    """`set(items)`'s members by `==`, without hashing them."""
    seen = []
    for item in items:
        if all(item != other for other in seen):
            seen.append(item)
    return seen


def _eager_only(kernel):
    def can_use(hidden_states, weight) -> bool:
        return not torch.compiler.is_compiling() and kernel(hidden_states, weight)

    can_use.eager_only = True
    return can_use


def _causal_depthwise(conv: torch.nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    """`conv` over `(B, C, T)`, causal: what `Lfm3ShortConv._conv` computes."""
    return _lfm2_conv(x, conv.weight[:, 0], conv.bias)


def _lfm2_conv(x: torch.Tensor, weight: torch.Tensor, bias=None, seq_idx=None) -> torch.Tensor:
    """transformers' `causal_conv1d_fn` as `Lfm2ShortConv` calls it: `weight` `(C, k)`, no activation, `seq_idx`
    unused, as in transformers' own torch path."""
    out = depthwise(F.pad(x.to(weight.dtype), (weight.shape[-1] - 1, 0)), weight[:, None])
    return (out if bias is None else out + bias[:, None]).to(x.dtype)
