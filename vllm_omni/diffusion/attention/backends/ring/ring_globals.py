# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import logging

# test if flash_attn (FA2) is available
try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import _flash_attn_forward  # noqa: F401

    HAS_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    HAS_FLASH_ATTN = False

# FA3 detection: try multiple sources (forward only, no backward needed for inference)
# Source 1: flash_attn_interface (from flash-attention source build)
# Source 2: fa3_fwd_interface (from fa3-fwd PyPI package, supports Ampere/Ada/Hopper)
# Note: FA3 high-level API may or may not return softmax_lse depending on version.
#       For Ring Attention which requires LSE, we fall back to low-level API if needed.
HAS_FA3 = False
fa3_fwd_func = None  # Low-level forward function (_flash_attn_forward)
fa3_attn_func = None  # High-level attention function (flash_attn_func)

# Try flash_attn_interface first (from flash-attention source build)
try:
    from flash_attn_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
    from flash_attn_interface import flash_attn_func as fa3_attn_func  # noqa: F401

    HAS_FA3 = True
except (ImportError, ModuleNotFoundError):
    pass

# Fallback: try fa3_fwd_interface (PyPI package, supports Ampere/Ada/Hopper)
if not HAS_FA3:
    try:
        from fa3_fwd_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
        from fa3_fwd_interface import flash_attn_func as fa3_attn_func  # noqa: F401

        HAS_FA3 = True
    except (ImportError, ModuleNotFoundError):
        pass

# Legacy aliases for backward compatibility
HAS_FLASH_ATTN_HOPPER = HAS_FA3
flash_attn_forward_hopper = fa3_fwd_func
flash3_attn_func = fa3_attn_func

logger = logging.getLogger(__name__)

# SYCL-TLA XPU IPC ring-attention extensions are optional. Keep imports
# guarded so importing the ring backend remains safe on CUDA, CPU, and XPU
# installations where the extensions have not been built.
sycl_tla_fmha = None
sycl_tla_ipc_p2p = None
HAS_SYCL_TLA_FMHA = False
HAS_SYCL_TLA_IPC = False

try:
    import sycl_tla_fmha  # noqa: F401

    HAS_SYCL_TLA_FMHA = True
except (ImportError, ModuleNotFoundError):
    pass
except Exception as e:
    logger.warning("SYCL-TLA FMHA is unavailable; XPU ring attention is disabled. Reason: %s", e)

try:
    import sycl_tla_ipc_p2p  # noqa: F401

    # Require the arena-based API introduced in commit 88164c1e of sycl-tla.
    # Older builds only expose ipc_get_handle / ipc_open_handle / ipc_wait
    # which have a use-after-free hazard; treat them as unavailable so the
    # caller can surface a clear error rather than segfaulting.
    _new_api = all(
        hasattr(sycl_tla_ipc_p2p, name)
        for name in ("make_arena", "open_peer", "copy_async", "shutdown")
    )
    if _new_api:
        HAS_SYCL_TLA_IPC = True
    else:
        logger.warning(
            "sycl_tla_ipc_p2p is present but lacks the arena-based API "
            "(make_arena / open_peer / copy_async / shutdown). "
            "Please rebuild the extension at or after commit "
            "88164c1e98a5271068e5b16688850453cf1b70a2 of Wei-Lin-Intel/sycl-tla. "
            "XPU IPC ring attention is disabled."
        )
except (ImportError, ModuleNotFoundError):
    pass
except Exception as e:
    logger.warning("SYCL-TLA IPC P2P is unavailable; XPU IPC ring attention is disabled. Reason: %s", e)

HAS_SYCL_TLA = HAS_SYCL_TLA_FMHA and HAS_SYCL_TLA_IPC

try:
    from flashinfer.prefill import single_prefill_with_kv_cache  # noqa: F401

    HAS_FLASHINFER = True
except Exception as e:
    # flashinfer may raise RuntimeError at import-time for version/binary mismatches.
    # Treat it as unavailable so the runtime can gracefully fall back to other backends.
    HAS_FLASHINFER = False
    logger.warning("FlashInfer is unavailable; falling back to other attention backends. Reason: %s", e)

try:
    import aiter  # noqa: F401
    from aiter import flash_attn_func as flash_attn_func_aiter  # noqa: F401

    HAS_AITER = True
except (ImportError, ModuleNotFoundError):
    HAS_AITER = False

try:
    import sageattention  # noqa: F401

    HAS_SAGE_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    HAS_SAGE_ATTENTION = False

try:
    import spas_sage_attn  # noqa: F401

    HAS_SPARSE_SAGE_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    HAS_SPARSE_SAGE_ATTENTION = False

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except (ImportError, ModuleNotFoundError):
    HAS_NPU = False
