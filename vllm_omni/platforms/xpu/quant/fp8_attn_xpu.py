# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 attention for XPU diffusion models.

Dynamically quantizes Q/K/V to ``float8_e4m3fn`` and runs an XE3 FP8 varlen
flash-attention kernel. Two kernel providers are supported, selected by the
``kv_cache_dtype`` label:

* ``"fp8"`` -> DeepKlox, with per-token-per-head Q descales and per-tensor K/V
  descales, the layout combination that kernel is tuned for.
* ``"fp8_xpu_kernels"`` -> vllm-xpu-kernels non-paged full-FP8 prefill, which
  folds per-tensor descales into the softmax scale and so requires scalar
  descales for Q as well.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from functools import lru_cache

import torch

try:
    from vllm_omni.diffusion.profiler.device_op_timer import device_op_timer
except ModuleNotFoundError as exc:
    if exc.name != "vllm_omni.diffusion.profiler.device_op_timer":
        raise
    device_op_timer = nullcontext

# The kernel accumulates in a reduced-range format, so descales target a
# quantization range well below the fp8_e4m3 max (448) to keep headroom.
FP8_QUANT_RANGE = 200.0
_MIN_DESCALE = 1e-6
# Q descale rows are consumed in 128-token tiles.
_Q_DESCALE_TILE = 128

# kv_cache_dtype label routing to the vllm-xpu-kernels non-paged prefill kernel.
XPU_KERNELS_FP8_LABEL = "fp8_xpu_kernels"
# The full-FP8 prefill kernel is only instantiated for head_size 128.
_XPU_KERNELS_HEAD_DIM = 128
_XPU_KERNELS_OUT_DTYPES = (torch.float16, torch.bfloat16)

_FP8_KV_LABELS = frozenset({"fp8", XPU_KERNELS_FP8_LABEL})


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8 QKV quantization for the XPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


@lru_cache(maxsize=1)
def _load_fp8_attn_func():
    try:
        from deepklox import flash_attn_varlen_func
    except ImportError as e:
        raise ImportError(
            "FP8 diffusion attention on XPU requires the DeepKlox FP8 flash-attention kernel. "
            "Install deepklox, or disable KV quantization by leaving --diffusion-kv-cache-dtype unset."
        ) from e
    return flash_attn_varlen_func


@lru_cache(maxsize=1)
def _load_xpu_kernels_fp8_attn_func():
    try:
        from vllm_xpu_kernels.flash_attn_interface import (
            FA2_AVAILABLE,
            FA2_UNAVAILABLE_REASON,
            flash_attn_varlen_func,
        )
    except ImportError as e:
        raise ImportError(
            f"kv_cache_dtype='{XPU_KERNELS_FP8_LABEL}' requires vllm-xpu-kernels. "
            "Install vllm-xpu-kernels, or use kv_cache_dtype='fp8' for the DeepKlox kernel."
        ) from e
    if not FA2_AVAILABLE:
        raise ImportError(
            f"kv_cache_dtype='{XPU_KERNELS_FP8_LABEL}' requires the compiled vllm-xpu-kernels "
            f"flash-attention extension, which is unavailable: {FA2_UNAVAILABLE_REASON}"
        )
    return flash_attn_varlen_func


def _quantize_per_tensor(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    descale = (tensor.float().abs().amax() / FP8_QUANT_RANGE).clamp(min=_MIN_DESCALE)
    # Divide in fp32: the kernel descales with this exact fp32 value, so rounding
    # the divisor to the input dtype would bias every element.
    return (tensor.float() / descale).to(fp8_dtype), descale


def _quantize_per_token_per_head(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[B, S, H, D]`` tensor and build the padded ``[B, H, S_pad]`` descale."""
    batch, seq_len, num_heads = tensor.shape[:3]
    descale_flat = (tensor.float().abs().amax(dim=-1) / FP8_QUANT_RANGE).clamp(min=_MIN_DESCALE)
    quantized = (tensor.float() / descale_flat.unsqueeze(-1)).to(fp8_dtype)

    padded_seq_len = math.ceil(seq_len / _Q_DESCALE_TILE) * _Q_DESCALE_TILE
    descale = torch.zeros(
        batch,
        num_heads,
        padded_seq_len,
        dtype=torch.float32,
        device=tensor.device,
    )
    descale[:, :, :seq_len] = descale_flat.transpose(1, 2)
    return quantized, descale


def _fp8_flash_attn_varlen_deepklox(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Run the DeepKlox FP8 kernel with per-token-per-head Q descales."""
    flash_attn_varlen_func = _load_fp8_attn_func()

    batch, q_len, num_heads, head_dim = query.shape
    k_len = key.shape[1]
    out_dtype = query.dtype

    with device_op_timer("xpu.fp8_mha.quantize_qkv"):
        q_fp8, q_descale = _quantize_per_token_per_head(query, fp8_dtype)
        k_fp8, k_descale = _quantize_per_tensor(key, fp8_dtype)
        v_fp8, v_descale = _quantize_per_tensor(value, fp8_dtype)

    cu_seqlens_q = torch.arange(0, (batch + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
    cu_seqlens_k = torch.arange(0, (batch + 1) * k_len, step=k_len, dtype=torch.int32, device=query.device)
    output = torch.empty(batch * q_len, num_heads, head_dim, dtype=out_dtype, device=query.device)

    with device_op_timer("xpu.fp8_mha.attn_kernel"):
        flash_attn_varlen_func(
            q_fp8.flatten(0, 1),
            k_fp8.flatten(0, 1),
            v_fp8.flatten(0, 1),
            cu_seqlens_q,
            cu_seqlens_k,
            q_len,
            k_len,
            softmax_scale=softmax_scale if softmax_scale is not None else head_dim**-0.5,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=output,
        )
    return output.reshape(batch, q_len, num_heads, head_dim)


def _fp8_flash_attn_varlen_xpu_kernels(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Run the vllm-xpu-kernels non-paged full-FP8 prefill kernel.

    This kernel folds the Q/K descales into the softmax scale and applies the V
    descale as a post-loop rescale, so all three descales must be per-tensor
    scalars.
    """
    flash_attn_varlen_func = _load_xpu_kernels_fp8_attn_func()

    batch, q_len, num_heads, head_dim = query.shape
    k_len = key.shape[1]
    num_kv_heads = key.shape[2]
    out_dtype = query.dtype

    if head_dim != _XPU_KERNELS_HEAD_DIM:
        raise ValueError(
            f"kv_cache_dtype='{XPU_KERNELS_FP8_LABEL}' only supports head_dim "
            f"{_XPU_KERNELS_HEAD_DIM}, got {head_dim}"
        )
    if out_dtype not in _XPU_KERNELS_OUT_DTYPES:
        raise ValueError(
            f"kv_cache_dtype='{XPU_KERNELS_FP8_LABEL}' requires float16 or bfloat16 query dtype, got {out_dtype}"
        )

    with device_op_timer("xpu.fp8_mha.quantize_qkv"):
        q_fp8, q_descale = _quantize_per_tensor(query, fp8_dtype)
        k_fp8, k_descale = _quantize_per_tensor(key, fp8_dtype)
        v_fp8, v_descale = _quantize_per_tensor(value, fp8_dtype)

    cu_seqlens_q = torch.arange(0, (batch + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
    cu_seqlens_k = torch.arange(0, (batch + 1) * k_len, step=k_len, dtype=torch.int32, device=query.device)
    output = torch.empty(batch * q_len, num_heads, head_dim, dtype=out_dtype, device=query.device)
    kv_scale_shape = (batch, num_kv_heads)

    with device_op_timer("xpu.fp8_mha.attn_kernel"):
        flash_attn_varlen_func(
            q_fp8.flatten(0, 1),
            k_fp8.flatten(0, 1),
            v_fp8.flatten(0, 1),
            q_len,
            cu_seqlens_q,
            k_len,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale if softmax_scale is not None else head_dim**-0.5,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale.expand(kv_scale_shape),
            v_descale=v_descale.expand(kv_scale_shape),
            out=output,
        )
    return output.reshape(batch, q_len, num_heads, head_dim)


def fp8_flash_attn_varlen_xpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    kv_cache_dtype: str = "fp8",
) -> torch.Tensor:
    """Run XPU FP8 flash attention over dense ``[B, S, H, D]`` Q/K/V.

    Returns the attention output in ``[B, S, H, D]`` with the query dtype.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"fp8_flash_attn_varlen_xpu expects 4D BSND tensors, got q={tuple(query.shape)}, k={tuple(key.shape)}"
        )
    if key.shape[0] != query.shape[0] or value.shape[0] != query.shape[0]:
        raise ValueError("fp8_flash_attn_varlen_xpu requires matching batch sizes for Q/K/V")

    impl = (
        _fp8_flash_attn_varlen_xpu_kernels
        if kv_cache_dtype == XPU_KERNELS_FP8_LABEL
        else _fp8_flash_attn_varlen_deepklox
    )
    return impl(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=causal,
        fp8_dtype=fp8_dtype,
    )
