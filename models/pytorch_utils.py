"""Utility helpers used by the PANNs‑based models (mix‑up, interpolation, FLOP
counting, …).  This version adds the `Interpolator` class expected by
models/cnn14.py and fixes a few minor typos while preserving the public API.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__: List[str] = [
    # data helpers
    "move_data_to_device",
    "do_mixup",
    "append_to_dict",
    "forward",
    # interpolation + padding helpers
    "interpolate",
    "pad_framewise_output",
    "Interpolator",
    # model statistics
    "count_parameters",
    "count_flops",
]

# ───────────────────────────────────────────── data movement ────

def move_data_to_device(x: torch.Tensor | np.ndarray, device: torch.device):
    """Cast *x* to the right Torch dtype and push it to *device*."""
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.floating):
            x = torch.tensor(x, dtype=torch.float32)
        elif np.issubdtype(x.dtype, np.integer):
            x = torch.tensor(x, dtype=torch.long)
        else:
            # non numeric → return as‑is
            return x
    return x.to(device)


# ───────────────────────────────────────────────── mix‑up ────

def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor) -> torch.Tensor:
    """Perform mix‑up on pairs (0∶even / 1∶odd).  See Zhang *et al.* (2018)."""
    # x shape: (batch*2, …)
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
        x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


# ─────────────────────────────────────── misc small helpers ────

def append_to_dict(d: Dict[str, List[Any]], key: str, value: Any):
    d.setdefault(key, []).append(value)


def forward(model: nn.Module, generator, *, return_input: bool = False, return_target: bool = False):
    """Run *generator* through *model* and collect outputs."""
    output: Dict[str, List[Any]] = {}
    device = next(model.parameters()).device
    tic = time.time()

    for i, batch in enumerate(generator):
        batch_waveform = move_data_to_device(batch['waveform'], device)
        model.eval()
        with torch.no_grad():
            pred = model(batch_waveform)

        append_to_dict(output, 'audio_name', batch['audio_name'])
        append_to_dict(output, 'clipwise_output', pred['clipwise_output'].cpu().numpy())

        for key in ('segmentwise_output', 'framewise_output'):
            if key in pred:
                append_to_dict(output, key, pred[key].cpu().numpy())

        if return_input:
            append_to_dict(output, 'waveform', batch['waveform'])
        if return_target and 'target' in batch:
            append_to_dict(output, 'target', batch['target'])

        if i % 10 == 0:
            print(f" --- Inference time: {time.time() - tic:.3f}s / 10 iters ---")
            tic = time.time()

    # concat
    return {k: np.concatenate(v, axis=0) for k, v in output.items()}


# ─────────────────────────── up‑sampling helpers (CNN alignment) ────

def interpolate(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Nearest‑neighbour upsample along *time* (dim=1)."""
    b, t, c = x.shape
    x = x[:, :, None, :].repeat(1, 1, ratio, 1)
    return x.view(b, t * ratio, c)


class Interpolator(nn.Module):
    """`nn.Module` wrapper around :pyfunc:`interpolate` expected by Cnn14."""
    def __init__(self, ratio: int = 32):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, frames, C)
        return interpolate(x, self.ratio)


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int) -> torch.Tensor:
    """Right‑pad *framewise_output* to *frames_num* using the last frame value."""
    if framewise_output.shape[1] >= frames_num:
        return framewise_output[:, :frames_num]
    pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    return torch.cat((framewise_output, pad), dim=1)


# ───────────────────────────────────── statistics utilities ────

def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (∑ over *requires_grad*)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, audio_length: int, *, multiply_adds: bool = True) -> int:
    """Very rough FLOP counter (borrowed & simplified). Only useful for research."""

    conv2d_flops: List[int] = []
    conv1d_flops: List[int] = []
    linear_flops: List[int] = []
    bn_flops: List[int] = []
    relu_flops: List[int] = []
    pool2d_flops: List[int] = []
    pool1d_flops: List[int] = []

    # ---- hook helpers --------------------------------------------------
    def conv2d_hook(layer, inp, out):
        b, _, h_in, w_in = inp[0].size()
        out_c, h_out, w_out = out.size()[1:]
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels // layer.groups
        bias_ops = 1 if layer.bias is not None else 0
        params = out_c * (kernel_ops * (2 if multiply_adds else 1) + bias_ops)
        conv2d_flops.append(b * params * h_out * w_out)

    def conv1d_hook(layer, inp, out):
        b, _, l_in = inp[0].size()
        out_c, l_out = out.size()[1:]
        kernel_ops = layer.kernel_size[0] * layer.in_channels // layer.groups
        bias_ops = 1 if layer.bias is not None else 0
        params = out_c * (kernel_ops * (2 if multiply_adds else 1) + bias_ops)
        conv1d_flops.append(b * params * l_out)

    def linear_hook(layer, inp, out):
        b = inp[0].size(0)
        weight_ops = layer.weight.numel() * (2 if multiply_adds else 1)
        bias_ops = layer.bias.numel() if layer.bias is not None else 0
        linear_flops.append(b * (weight_ops + bias_ops))

    def bn_hook(layer, inp, _out):
        bn_flops.append(inp[0].numel() * 2)

    def relu_hook(layer, inp, _out):
        relu_flops.append(inp[0].numel())

    def pool2d_hook(layer, inp, out):
        b, c_out, h_out, w_out = out.size()
        kernel_ops = layer.kernel_size ** 2 if isinstance(layer.kernel_size, int) else layer.kernel_size[0] * layer.kernel_size[1]
        pool2d_flops.append(b * c_out * h_out * w_out * kernel_ops)

    def pool1d_hook(layer, inp, out):
        b, c_out, l_out = out.size()
        kernel_ops = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        pool1d_flops.append(b * c_out * l_out * kernel_ops)

    # ---- register all sub‑modules --------------------------------------
    def register_hooks(net: nn.Module):
        for child in net.children():
            if isinstance(child, nn.Conv2d):
                child.register_forward_hook(conv2d_hook)
            elif isinstance(child, nn.Conv1d):
                child.register_forward_hook(conv1d_hook)
            elif isinstance(child, nn.Linear):
                child.register_forward_hook(linear_hook)
            elif isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                child.register_forward_hook(bn_hook)
            elif isinstance(child, nn.ReLU):
                child.register_forward_hook(relu_hook)
            elif isinstance(child, (nn.AvgPool2d, nn.MaxPool2d)):
                child.register_forward_hook(pool2d_hook)
            elif isinstance(child, (nn.AvgPool1d, nn.MaxPool1d)):
                child.register_forward_hook(pool1d_hook)
            else:
                register_hooks(child)

    register_hooks(model)

    device = next(model.parameters()).device
    dummy = torch.rand(1, audio_length, device=device)
    model(dummy)

    total = sum(conv2d_flops + conv1d_flops + linear_flops + bn_flops + relu_flops + pool2d_flops + pool1d_flops)
    return int(total)

