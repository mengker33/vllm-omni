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
* ``"fp8_sycl_tla"`` -> the SYCL-TLA (CUTLASS-SYCL) ``sycl_tla_fmha`` prefill
  binding, which has no descale inputs at all, so the scaling has to be folded
  into the quantized values themselves.
* ``"mxfp8_sycl_tla"`` -> the same binding's block-scaled entry point, which
  takes UE8M0 scale factors covering 32 elements each, so per-tensor descales
  are unnecessary and only the softmax scale has to be compensated.
* ``"e4m3qk_bf16v_sycl_tla"`` -> the same binding's mixed-precision entry point:
  E4M3 Q/K with per-tensor descale scalars folded into the softmax scale, and V
  left in BF16 so the P@V GEMM keeps full precision.

All providers first get SageAttention-style K/V sequence-mean smoothing (see
``_smooth_kv``), which removes the channel-mean outliers that otherwise consume
most of the FP8 range.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from functools import lru_cache

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Set to 1/true/yes/on to log per-call attention and quantization device times.
ATTN_TIMING_ENV = "VLLM_OMNI_XPU_ATTN_TIMING"
# Set to 0/false to disable SageAttention-style K/V sequence-mean smoothing.
SMOOTH_KV_ENV = "VLLM_OMNI_XPU_ATTN_SMOOTH_KV"


def _attn_timing_enabled() -> bool:
    return os.environ.get(ATTN_TIMING_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def _attn_timer(label: str):
    """Time a device region and log it, or do nothing when timing is disabled."""
    if not _attn_timing_enabled():
        yield
        return
    start_event = torch.Event(enable_timing=True)
    start_event.record()
    yield
    end_event = torch.Event(enable_timing=True)
    end_event.record()
    torch.xpu.synchronize()
    logger.info("%s device time: %.3f ms", label, start_event.elapsed_time(end_event))

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

# kv_cache_dtype label routing to the SYCL-TLA sycl_tla_fmha prefill binding.
SYCL_TLA_FP8_LABEL = "fp8_sycl_tla"
# Same binding, block-scaled (MX) variant with per-32-element UE8M0 scales.
SYCL_TLA_MXFP8_LABEL = "mxfp8_sycl_tla"
# Same binding, E4M3 Q/K with per-tensor scales and BF16 V.
SYCL_TLA_E4M3QK_BF16V_LABEL = "e4m3qk_bf16v_sycl_tla"
# Directory holding the built ``sycl_tla_fmha`` extension, if not on PYTHONPATH.
SYCL_TLA_PATH_ENV = "VLLM_OMNI_SYCL_TLA_FMHA_PATH"
_SYCL_TLA_HEAD_DIM = 128
_FP8_E4M3_MAX = 448.0

# MX block size and E4M3 shared-exponent encoding.
_MX_GROUP_SIZE = 32
_UE8M0_BIAS = 127

_FP8_KV_LABELS = frozenset(
    {
        "fp8",
        XPU_KERNELS_FP8_LABEL,
        SYCL_TLA_FP8_LABEL,
        SYCL_TLA_MXFP8_LABEL,
        SYCL_TLA_E4M3QK_BF16V_LABEL,
    }
)


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8 QKV quantization for the XPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


# Dynamo ignores lru_cache and retraces the wrapped body; these loaders only
# import an extension and hand back a pybind callable, so keep them opaque.
@torch._dynamo.disable
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


@torch._dynamo.disable
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


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _import_sycl_tla_fmha(label: str):
    search_path = os.environ.get(SYCL_TLA_PATH_ENV)
    if search_path and search_path not in sys.path:
        sys.path.append(search_path)
    try:
        import sycl_tla_fmha
    except ImportError as e:
        raise ImportError(
            f"kv_cache_dtype='{label}' requires the sycl_tla_fmha extension built from "
            f"cutlass-sycl examples/06_bmg_flash_attention. Put its build directory on PYTHONPATH "
            f"or set {SYCL_TLA_PATH_ENV} to it."
        ) from e
    return sycl_tla_fmha


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _load_sycl_tla_fmha_func():
    module = _import_sycl_tla_fmha(SYCL_TLA_FP8_LABEL)
    logger.info(
        "SYCL-TLA FP8 E4M3 flash-attention enabled (kv_cache_dtype='%s') from %s",
        SYCL_TLA_FP8_LABEL,
        module.__file__,
    )
    return module.prefill_fp8_e4m3_bshd


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _load_sycl_tla_mxfp8_func():
    module = _import_sycl_tla_fmha(SYCL_TLA_MXFP8_LABEL)
    if not hasattr(module, "prefill_mxfp8_e4m3_bshd"):
        raise ImportError(
            f"kv_cache_dtype='{SYCL_TLA_MXFP8_LABEL}' requires a sycl_tla_fmha build that exports "
            f"prefill_mxfp8_e4m3_bshd; rebuild the extension from {module.__file__}"
        )
    logger.info(
        "SYCL-TLA MXFP8 E4M3 flash-attention enabled (kv_cache_dtype='%s') from %s",
        SYCL_TLA_MXFP8_LABEL,
        module.__file__,
    )
    return module.prefill_mxfp8_e4m3_bshd


@torch._dynamo.disable
@lru_cache(maxsize=1)
def _load_sycl_tla_e4m3qk_bf16v_func():
    module = _import_sycl_tla_fmha(SYCL_TLA_E4M3QK_BF16V_LABEL)
    if not hasattr(module, "prefill_e4m3qk_bf16v_bshd"):
        raise ImportError(
            f"kv_cache_dtype='{SYCL_TLA_E4M3QK_BF16V_LABEL}' requires a sycl_tla_fmha build that "
            f"exports prefill_e4m3qk_bf16v_bshd; rebuild the extension from {module.__file__}"
        )
    logger.info(
        "SYCL-TLA E4M3 QK / BF16 PV flash-attention enabled (kv_cache_dtype='%s') from %s",
        SYCL_TLA_E4M3QK_BF16V_LABEL,
        module.__file__,
    )
    return module.prefill_e4m3qk_bf16v_bshd


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

    q_fp8, q_descale = _quantize_per_token_per_head(query, fp8_dtype)
    k_fp8, k_descale = _quantize_per_tensor(key, fp8_dtype)
    v_fp8, v_descale = _quantize_per_tensor(value, fp8_dtype)

    cu_seqlens_q = torch.arange(0, (batch + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
    cu_seqlens_k = torch.arange(0, (batch + 1) * k_len, step=k_len, dtype=torch.int32, device=query.device)
    output = torch.empty(batch * q_len, num_heads, head_dim, dtype=out_dtype, device=query.device)

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

    q_fp8, q_descale = _quantize_per_tensor(query, fp8_dtype)
    k_fp8, k_descale = _quantize_per_tensor(key, fp8_dtype)
    v_fp8, v_descale = _quantize_per_tensor(value, fp8_dtype)

    cu_seqlens_q = torch.arange(0, (batch + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
    cu_seqlens_k = torch.arange(0, (batch + 1) * k_len, step=k_len, dtype=torch.int32, device=query.device)
    output = torch.empty(batch * q_len, num_heads, head_dim, dtype=out_dtype, device=query.device)
    kv_scale_shape = (batch, num_kv_heads)

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


def _abs_amax(tensor: torch.Tensor) -> torch.Tensor:
    # Two reductions rather than .float().abs().amax(), which materializes a
    # full-size fp32 copy and then a full-size abs copy just to reduce them.
    return torch.maximum(tensor.amax().abs(), tensor.amin().abs()).float()


def _smooth_kv_enabled() -> bool:
    return os.environ.get(SMOOTH_KV_ENV, "1").strip().lower() in ("1", "true", "yes", "on")


def _smooth_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    smooth_v: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Subtract the per-channel sequence mean from K, and optionally from V.

    K/V channel means are the dominant outliers in DiT attention: they consume
    most of the FP8 range while carrying no information that survives softmax.
    Both corrections are exact, so neither needs kernel support.

    * K: dropping ``mu_K`` subtracts ``q_i . mu_K`` from every logit in row
      ``i``, and softmax is invariant to a per-row constant.
    * V: attention rows sum to one, so the output shifts by exactly ``mu_V``;
      it is returned for the caller to add back.

    Q cannot be smoothed this way -- ``mu_Q . k_j`` varies along the key axis,
    so it needs the ``mu_Q @ K_s^T`` logit correction fused into the kernel.
    """
    if not _smooth_kv_enabled():
        return key, value, None

    with _attn_timer("K/V sequence-mean smoothing"):
        # The means are accumulated in fp32; the subtraction itself stays in the
        # input dtype, whose rounding is an order of magnitude below e4m3's.
        key = key - key.mean(dim=1, keepdim=True, dtype=torch.float32).to(key.dtype)
        v_mean = None
        if smooth_v:
            v_mean = value.mean(dim=1, keepdim=True, dtype=torch.float32).to(value.dtype)
            value = value - v_mean
    return key, value, v_mean


def _scale_to_fp8(tensor: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    # e4m3fn has no infinities: PyTorch maps out-of-range inputs to NaN, so clamp first.
    # copy=True is required: .to() is a no-op for an fp32 input, and mul_ would
    # then write through into the caller's tensor.
    return (
        tensor.to(torch.float32, copy=True)
        .mul_(scale)
        .clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(fp8_dtype)
    )


def _fp8_flash_attn_sycl_tla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Run the SYCL-TLA ``sycl_tla_fmha`` FP8 E4M3 prefill kernel.

    The binding takes no descales and hardcodes a ``1/sqrt(head_dim)`` softmax
    scale, so the scaling is folded into the quantized tensors instead:

    * Q and K get reciprocal scales whose product is
      ``softmax_scale * sqrt(head_dim)``, which leaves the logits unchanged
      while equalizing how much FP8 range each side uses.
    * V is scaled to fill the FP8 range and the output is divided back out,
      which is exact because ``P @ (c*V) == c * (P @ V)``.
    """
    prefill_fp8_e4m3_bshd = _load_sycl_tla_fmha_func()

    head_dim = query.shape[-1]
    out_dtype = query.dtype
    if head_dim != _SYCL_TLA_HEAD_DIM:
        raise ValueError(
            f"kv_cache_dtype='{SYCL_TLA_FP8_LABEL}' only supports head_dim {_SYCL_TLA_HEAD_DIM}, got {head_dim}"
        )
    if fp8_dtype != torch.float8_e4m3fn:
        raise ValueError(f"kv_cache_dtype='{SYCL_TLA_FP8_LABEL}' only supports float8_e4m3fn, got {fp8_dtype}")

    # alpha compensates for the kernel's fixed 1/sqrt(head_dim) softmax scale.
    with _attn_timer("SYCL-TLA FP8 attention quantization"):
        alpha = (softmax_scale if softmax_scale is not None else head_dim**-0.5) * math.sqrt(head_dim)
        q_amax = _abs_amax(query).clamp_(min=_MIN_DESCALE)
        k_amax = _abs_amax(key).clamp_(min=_MIN_DESCALE)
        v_amax = _abs_amax(value).clamp_(min=_MIN_DESCALE)

        q_scale = torch.sqrt(alpha * k_amax / q_amax)
        v_scale = _FP8_E4M3_MAX / v_amax

        q_fp8 = _scale_to_fp8(query, q_scale, fp8_dtype)
        k_fp8 = _scale_to_fp8(key, alpha / q_scale, fp8_dtype)
        v_fp8 = _scale_to_fp8(value, v_scale, fp8_dtype)

    # The binding runs on its own compat queue, so both of Torch's pending
    # writes and the kernel's writes need an explicit fence.
    torch.xpu.synchronize()
    with _attn_timer("SYCL-TLA FP8 attention"):
        output = prefill_fp8_e4m3_bshd(q=q_fp8, k=k_fp8, v=v_fp8, is_causal=causal)

    # The binding hands back a private bf16 buffer, so rescale it in place.
    return output.div_(v_scale).to(out_dtype)


def _mx_group_amax_head_dim(tensor: torch.Tensor) -> torch.Tensor:
    """max|x| per contiguous 32-element group along the head dim of ``[B, S, H, D]``.

    Pooling instead of ``amax`` over a 32-wide innermost axis: the reduction
    kernel only reaches ~56 GB/s on that layout, which measures ~5x slower.
    """
    b, s, h, d = tensor.shape
    pooled = torch.nn.functional.max_pool1d(
        tensor.abs().reshape(1, b * s * h, d), _MX_GROUP_SIZE, _MX_GROUP_SIZE
    )
    return pooled.reshape(b, s, h, d // _MX_GROUP_SIZE).float()


def _mx_exponent_from_amax(amax: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if scale != 1.0:
        amax = amax.mul_(scale)
    amax = amax.clamp_min(torch.finfo(torch.float32).tiny)
    # A floor(log2(amax)) - 8 scale maps maxima as high as 512 into E4M3,
    # whose finite limit is 448, clipping the largest values in many blocks.
    exponent = torch.ceil(torch.log2(amax / _FP8_E4M3_MAX))
    return exponent.clamp(-_UE8M0_BIAS, _UE8M0_BIAS)


def _mx_exponent(blocked: torch.Tensor, reduce_dim: int, scale: float = 1.0) -> torch.Tensor:
    # One abs() pass plus one reduction, rather than reducing twice for amax/amin.
    return _mx_exponent_from_amax(blocked.abs().amax(reduce_dim).float(), scale)


def _mx_apply(blocked: torch.Tensor, exponent: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    # exp2(-exponent) folds the descale and ``scale`` into one broadcast multiply.
    factor = torch.exp2(exponent.neg())
    if scale != 1.0:
        factor = factor.mul_(scale)
    if blocked.dtype == torch.float16:
        # Large reciprocal UE8M0 scales overflow FP16 before multiplication;
        # in particular, an all-zero block would become 0 * inf = NaN.
        scaled = torch.mul(blocked.float(), factor)
    else:
        # Power-of-two block scales are exact in BF16 and avoid an fp32 copy.
        scaled = torch.mul(blocked, factor.to(blocked.dtype))
    return scaled.clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)


# Intel's Triton backend fails to lower the fused MX quantization kernel
# (ConvertTritonIntelGPUToLLVM), so keep this out of any Inductor graph --
# including an outer torch.compile around the attention layer.
@torch._dynamo.disable
def _mx_quantize(tensor: torch.Tensor, group_dim: int, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """MX-quantize a ``[B, S, H, D]`` tensor along ``group_dim`` in blocks of 32.

    Splits the group axis in place rather than moving it to the end, so every
    reshape stays a view; ``movedim`` + ``reshape`` would force a full
    transposing copy for the sequence-grouped case.

    ``scale`` is applied to the values as part of the quantization multiply,
    which avoids materializing a separately scaled copy of the input.

    Returns (float8_e4m3fn values in the input's axis order, uint8 UE8M0
    exponents shaped ``[B, S, H, D/32]`` for ``group_dim=3`` and
    ``[B, ceil(S/32), H, D]`` for ``group_dim=1``).
    """
    g = _MX_GROUP_SIZE
    b, s, h, d = tensor.shape

    if group_dim == 3:
        if d % g:
            raise ValueError(f"head dim {d} must be a multiple of {g}")
        blocked = tensor.reshape(b, s, h, d // g, g)
        exponent = _mx_exponent_from_amax(_mx_group_amax_head_dim(tensor), scale)
        values = _mx_apply(blocked, exponent.unsqueeze(-1), scale).reshape(b, s, h, d)
        return values, (exponent + _UE8M0_BIAS).to(torch.uint8)

    if group_dim != 1:
        raise ValueError(f"unsupported group_dim {group_dim}")

    n_full = (s // g) * g

    # Whole groups only: collapse to [groups, 32, H*D] so the broadcast multiply
    # keeps a contiguous innermost axis, and skip the staging buffer entirely.
    if n_full == s:
        blocked = tensor.reshape(b * (s // g), g, h * d)
        exponent = _mx_exponent(blocked, 1, scale)
        values = _mx_apply(blocked, exponent.unsqueeze(1), scale).reshape(b, s, h, d)
        return values, (exponent.reshape(b, s // g, h, d) + _UE8M0_BIAS).to(torch.uint8)

    # A ragged tail is reduced separately so the full tensor never needs padding.
    values = torch.empty((b, s, h, d), dtype=torch.float8_e4m3fn, device=tensor.device)

    if n_full:
        blocked = tensor[:, :n_full].reshape(b, n_full // g, g, h, d)
        exponent = _mx_exponent(blocked, 2, scale)
        values[:, :n_full] = _mx_apply(blocked, exponent.unsqueeze(2), scale).reshape(b, n_full, h, d)
    else:
        exponent = None

    tail = tensor[:, n_full:].unsqueeze(1)
    tail_exponent = _mx_exponent(tail, 2, scale)
    values[:, n_full:] = _mx_apply(tail, tail_exponent.unsqueeze(2), scale).reshape(b, s - n_full, h, d)
    exponent = tail_exponent if exponent is None else torch.cat([exponent, tail_exponent], dim=1)

    return values, (exponent + _UE8M0_BIAS).to(torch.uint8)


def _mxfp8_flash_attn_sycl_tla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Run the SYCL-TLA ``sycl_tla_fmha`` MXFP8 E4M3 block-scaled prefill kernel.

    Per-32-element UE8M0 scales are applied inside the kernel, so unlike the
    per-tensor FP8 path there is no descale to fold into V or to divide back
    out of the output. Only the kernel's fixed ``1/sqrt(head_dim)`` softmax
    scale needs compensating, which is folded into Q's quantization multiply;
    MX then picks a per-block exponent for the scaled values automatically.
    """
    prefill_mxfp8_e4m3_bshd = _load_sycl_tla_mxfp8_func()

    head_dim = query.shape[-1]
    out_dtype = query.dtype
    if head_dim != _SYCL_TLA_HEAD_DIM:
        raise ValueError(
            f"kv_cache_dtype='{SYCL_TLA_MXFP8_LABEL}' only supports head_dim {_SYCL_TLA_HEAD_DIM}, "
            f"got {head_dim}"
        )
    if fp8_dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"kv_cache_dtype='{SYCL_TLA_MXFP8_LABEL}' only supports float8_e4m3fn, got {fp8_dtype}"
        )

    with _attn_timer("SYCL-TLA MXFP8 attention quantization"):
        alpha = (softmax_scale if softmax_scale is not None else head_dim**-0.5) * math.sqrt(head_dim)

        # Q/K group along the head dim; V groups along the sequence, because the
        # kernel indexes V as (head_size_vo, seq_len_kv).
        q_fp8, q_exp = _mx_quantize(query, 3, alpha)
        k_fp8, k_exp = _mx_quantize(key, 3)
        v_fp8, v_exp = _mx_quantize(value, 1)

    # Q/K exponents are [B, S, H, D/32], V's are [B, S/32, H, D]; both
    # become the kernel's [B, H, groups, rows].
    scale_q = q_exp.permute(0, 2, 3, 1).contiguous()
    scale_k = k_exp.permute(0, 2, 3, 1).contiguous()
    scale_v = v_exp.permute(0, 2, 1, 3).contiguous()

    # The binding runs on its own compat queue, so both of Torch's pending
    # writes and the kernel's writes need an explicit fence.
    torch.xpu.synchronize()
    with _attn_timer("SYCL-TLA MXFP8 attention"):
        output = prefill_mxfp8_e4m3_bshd(
            q=q_fp8, k=k_fp8, v=v_fp8,
            scale_q=scale_q, scale_k=scale_k, scale_v=scale_v,
            is_causal=causal,
        )

    return output.to(out_dtype)


def _e4m3qk_bf16v_flash_attn_sycl_tla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Run the SYCL-TLA ``sycl_tla_fmha`` E4M3 QK / BF16 PV prefill kernel.

    Only the Q@K GEMM is quantized: the binding takes per-tensor descale
    scalars for Q and K and folds their product into the softmax scale, while V
    stays BF16 so the P@V GEMM and the output need no rescaling. The kernel's
    fixed ``1/sqrt(head_dim)`` softmax scale is compensated by scaling the Q
    descale, which is exact because the descale is applied to the logits.
    """
    prefill_e4m3qk_bf16v_bshd = _load_sycl_tla_e4m3qk_bf16v_func()

    head_dim = query.shape[-1]
    out_dtype = query.dtype
    if head_dim != _SYCL_TLA_HEAD_DIM:
        raise ValueError(
            f"kv_cache_dtype='{SYCL_TLA_E4M3QK_BF16V_LABEL}' only supports head_dim "
            f"{_SYCL_TLA_HEAD_DIM}, got {head_dim}"
        )
    if fp8_dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"kv_cache_dtype='{SYCL_TLA_E4M3QK_BF16V_LABEL}' only supports float8_e4m3fn, "
            f"got {fp8_dtype}"
        )

    with _attn_timer("SYCL-TLA E4M3 QK / BF16 PV attention quantization"):
        alpha = (softmax_scale if softmax_scale is not None else head_dim**-0.5) * math.sqrt(head_dim)
        q_amax = _abs_amax(query).clamp_(min=_MIN_DESCALE)
        k_amax = _abs_amax(key).clamp_(min=_MIN_DESCALE)

        q_fp8 = _scale_to_fp8(query, _FP8_E4M3_MAX / q_amax, fp8_dtype)
        k_fp8 = _scale_to_fp8(key, _FP8_E4M3_MAX / k_amax, fp8_dtype)
        # The binding takes Python floats, so these force a device sync.
        scale_q = (q_amax / _FP8_E4M3_MAX).item() * alpha
        scale_k = (k_amax / _FP8_E4M3_MAX).item()
        v_bf16 = value.to(torch.bfloat16)

    # The binding runs on its own compat queue, so both of Torch's pending
    # writes and the kernel's writes need an explicit fence.
    torch.xpu.synchronize()
    with _attn_timer("SYCL-TLA E4M3 QK / BF16 PV attention"):
        output = prefill_e4m3qk_bf16v_bshd(
            q=q_fp8, k=k_fp8, v=v_bf16,
            scale_q=scale_q, scale_k=scale_k,
            is_causal=causal,
        )

    return output.to(out_dtype)


_FP8_IMPLS = {
    XPU_KERNELS_FP8_LABEL: _fp8_flash_attn_varlen_xpu_kernels,
    SYCL_TLA_FP8_LABEL: _fp8_flash_attn_sycl_tla,
    SYCL_TLA_MXFP8_LABEL: _mxfp8_flash_attn_sycl_tla,
    SYCL_TLA_E4M3QK_BF16V_LABEL: _e4m3qk_bf16v_flash_attn_sycl_tla,
}


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

    impl = _FP8_IMPLS.get(kv_cache_dtype, _fp8_flash_attn_varlen_deepklox)
    # Smoothing V only pays off where V is quantized.
    key, value, v_mean = _smooth_kv(key, value, smooth_v=kv_cache_dtype != SYCL_TLA_E4M3QK_BF16V_LABEL)
    output = impl(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=causal,
        fp8_dtype=fp8_dtype,
    )
    if v_mean is not None:
        output = output.add_(v_mean.to(output.dtype))
    return output
