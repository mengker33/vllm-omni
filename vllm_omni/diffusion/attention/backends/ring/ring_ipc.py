# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Level-Zero IPC K/V ring buffers for Intel XPU ring attention.

This module only owns peer-buffer setup and copy scheduling. The attention
loop and its output/LSE accumulation are implemented by the caller.

Requires sycl_tla_ipc_p2p built at or after commit
88164c1e98a5271068e5b16688850453cf1b70a2 of Wei-Lin-Intel/sycl-tla
(arena-based allocation with explicit lifetime management).
"""

from __future__ import annotations

import atexit
import math
from typing import Any

import torch
import torch.distributed as dist

from .ring_globals import (
    HAS_SYCL_TLA,
    HAS_SYCL_TLA_IPC,
    sycl_tla_fmha as fa,
    sycl_tla_ipc_p2p as ipc,
)


NUM_BUFFERS = 4

# Process-lifetime cache: (pg_id, dev_index, k_shape, v_shape, dtype) -> IpcKVRing
_ring_cache: dict[tuple[Any, ...], IpcKVRing] = {}
_shutdown_registered = False


def _check_new_api() -> None:
    """Raise a clear error if the installed IPC module lacks the arena-based API."""
    if ipc is None or not HAS_SYCL_TLA_IPC:
        raise RuntimeError(
            "SYCL-TLA IPC P2P is unavailable; install the "
            "sycl_tla_ipc_p2p extension or use the PyTorch ring backend"
        )
    missing = [
        name
        for name in ("make_arena", "open_peer", "copy_async", "shutdown")
        if not hasattr(ipc, name)
    ]
    if missing:
        raise RuntimeError(
            f"sycl_tla_ipc_p2p is missing arena-based API symbols: {missing}. "
            "Please rebuild the extension at or after commit "
            "88164c1e98a5271068e5b16688850453cf1b70a2 of Wei-Lin-Intel/sycl-tla."
        )


def _register_shutdown() -> None:
    global _shutdown_registered
    if _shutdown_registered:
        return
    _shutdown_registered = True

    def _shutdown() -> None:
        if ipc is not None and hasattr(ipc, "shutdown"):
            try:
                ipc.shutdown()
            except Exception:
                pass

    atexit.register(_shutdown)


def align_up(n: int, align: int) -> int:
    return (n + align - 1) & ~(align - 1)


def read_slot_of(step: int) -> int:
    """Return the local slot containing the K/V block for ``step``."""
    return 0 if step == 0 else 1 + ((step - 1) % (NUM_BUFFERS - 1))


def write_slot_of(step: int) -> int:
    """Return the rotating destination slot for the copy at ``step``."""
    return 1 + (step % (NUM_BUFFERS - 1))


def current_queue_ptr(device: torch.device) -> int:
    """Return the raw SYCL queue pointer for the current XPU stream."""
    stream = torch.xpu.current_stream(device)
    for attr in ("sycl_queue", "_as_parameter_", "queue"):
        queue = getattr(stream, attr, None)
        if queue is None:
            continue
        if isinstance(queue, int):
            return queue
        try:
            return int(queue)
        except (TypeError, ValueError):
            continue
    raise RuntimeError(
        "cannot obtain sycl_queue pointer from the current XPU stream; "
        "this PyTorch build does not expose Stream.sycl_queue"
    )


class IpcKVRing:
    """Persistent arena-based K/V buffers and peer IPC mappings for one ring group.

    Slot zero permanently contains the rank's own block for the current
    attention pass (refreshed via ``stage``). The other slots are rotating
    copy destinations. Handles are exchanged once at construction via
    ``dist.all_gather_object`` so that no IPC handle is created or destroyed
    per attention layer.
    """

    def __init__(
        self,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        _check_new_api()
        if not hasattr(torch, "xpu") or k_local.device.type != "xpu":
            raise RuntimeError("IpcKVRing requires tensors on an Intel XPU")
        if k_local.device != v_local.device:
            raise ValueError("K and V must be on the same XPU device")
        if not k_local.is_contiguous() or not v_local.is_contiguous():
            raise ValueError("K and V must be contiguous before creating IpcKVRing")

        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        if self.world_size < 2:
            raise ValueError("IpcKVRing requires at least two ranks")
        self.prev_rank = (self.rank - 1) % self.world_size

        # Capture the queue pointer once at construction and reuse it for every
        # subsequent IPC call. env_for_device hard-errors if a device is used
        # with a different sycl::queue than the one it was initialized with.
        self.queue_ptr = current_queue_ptr(k_local.device)
        self.device = k_local.device
        self.dev_index = k_local.device.index or 0

        # ── Arena layout ──────────────────────────────────────────────────────
        # slot_i = [K_i (k_nbytes)] [padding] [V_i (v_nbytes)] [padding]
        # Offsets within a slot are aligned to 4096B (page / IPC alignment).
        # The arena base must be 256B-aligned (IpcArena.view requirement).
        k_nbytes = k_local.numel() * k_local.element_size()
        v_nbytes = v_local.numel() * v_local.element_size()
        self.k_nbytes = k_nbytes
        self.v_nbytes = v_nbytes

        self.k_off_in_slot = 0
        self.v_off_in_slot = align_up(k_nbytes, 4096)
        self.slot_stride = align_up(self.v_off_in_slot + v_nbytes, 4096)
        total_bytes = self.slot_stride * NUM_BUFFERS

        self.arena = ipc.make_arena(self.dev_index, total_bytes, self.queue_ptr)

        # Non-owning tensor views into the arena, one per slot for K and V.
        self.kbuf: list[torch.Tensor] = []
        self.vbuf: list[torch.Tensor] = []
        for i in range(NUM_BUFFERS):
            slot_base = i * self.slot_stride
            self.kbuf.append(
                self.arena.view(slot_base + self.k_off_in_slot, k_local.shape, k_local.dtype)
            )
            self.vbuf.append(
                self.arena.view(slot_base + self.v_off_in_slot, v_local.shape, v_local.dtype)
            )

        # ── One-time handle exchange ───────────────────────────────────────────
        local_handle = self.arena.export_handle()
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, local_handle, group=process_group)
        peer_handle = gathered[self.prev_rank]

        self.peer_base: int = ipc.open_peer(self.dev_index, peer_handle, self.queue_ptr)

        # Peer slot K/V raw pointers are just peer_base + slot offset.
        self.k_peer_ptr: list[int] = []
        self.v_peer_ptr: list[int] = []
        for i in range(NUM_BUFFERS):
            slot_base = i * self.slot_stride
            self.k_peer_ptr.append(self.peer_base + slot_base + self.k_off_in_slot)
            self.v_peer_ptr.append(self.peer_base + slot_base + self.v_off_in_slot)

        # Wait for every rank to have opened its peer before any DMA.
        dist.barrier(group=process_group)
        torch.xpu.synchronize(self.device)

    def stage(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Copy fresh per-layer K/V tensors into slot 0 of the arena."""
        self.kbuf[0].copy_(k_new)
        self.vbuf[0].copy_(v_new)
        # copy_ is enqueued on the compute queue; the copy engine cannot see it
        # until the compute queue drains. Synchronize so peers can safely DMA
        # from slot 0.
        torch.xpu.current_stream(self.device).synchronize()

    def copy_next(self, step: int) -> list[Any]:
        """Copy the previous rank's block for ``step + 1`` asynchronously.

        Returns a list of ``PendingCopy`` objects that must be waited before
        the destination slot is read.
        """
        if step >= self.world_size - 1:
            return []

        read_slot = read_slot_of(step)
        write_slot = write_slot_of(step)
        arena_base = self.arena.base_ptr()
        pending = [
            ipc.copy_async(
                self.dev_index,
                arena_base + write_slot * self.slot_stride + self.k_off_in_slot,
                self.k_peer_ptr[read_slot],
                self.k_nbytes,
                self.queue_ptr,
            ),
            ipc.copy_async(
                self.dev_index,
                arena_base + write_slot * self.slot_stride + self.v_off_in_slot,
                self.v_peer_ptr[read_slot],
                self.v_nbytes,
                self.queue_ptr,
            ),
        ]
        return pending


def _get_or_create_ring(
    process_group: dist.ProcessGroup | None,
    k: torch.Tensor,
    v: torch.Tensor,
) -> IpcKVRing:
    """Return a cached ``IpcKVRing`` for this (group, device, shape, dtype) key."""
    global _ring_cache
    pg_id = id(process_group)
    dev_index = k.device.index or 0
    cache_key = (pg_id, dev_index, tuple(k.shape), tuple(v.shape), k.dtype)

    ring = _ring_cache.get(cache_key)
    if ring is None:
        _register_shutdown()
        ring = IpcKVRing(k, v, process_group=process_group)
        _ring_cache[cache_key] = ring
    return ring


@torch.compiler.disable
def ring_ipc_attn_forward(
    process_group: dist.ProcessGroup | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    joint_tensor_key: torch.Tensor | None = None,
    joint_tensor_value: torch.Tensor | None = None,
    joint_strategy: str = "front",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run BF16 non-causal ring attention with IPC K/V rotation.

    Unlike the regular ring backend, this function keeps output and LSE
    buffers alive for the complete ring pass. The SYCL-TLA round kernel does
    the online-softmax merge in-place, so ``update_out_and_lse`` must not be
    called for this path.
    """
    if not HAS_SYCL_TLA or fa is None:
        raise RuntimeError(
            "SYCL-TLA FMHA and IPC extensions are unavailable; use the "
            "PyTorch ring backend instead"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("SYCL-TLA IPC ring attention currently requires BF16 Q, K, and V")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("SYCL-TLA IPC ring attention requires [B, S, H, D] tensors")
    if q.device.type != "xpu" or k.device != q.device or v.device != q.device:
        raise ValueError("SYCL-TLA IPC ring attention requires all tensors on one XPU device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Q, K, and V must have the same batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("Q, K, and V must have the same local sequence length")
    if q.shape[2] % k.shape[2] != 0 or k.shape[2] != v.shape[2]:
        raise ValueError("Q heads must be divisible by the number of K/V heads")
    if q.shape[3] != k.shape[3]:
        raise ValueError("Q and K must have the same head dimension")
    if q.shape[3] % 32 != 0:
        raise ValueError("SYCL-TLA requires the Q/K head dimension to be a multiple of 32")
    if v.shape[3] not in (64, 96, 128, 192):
        raise ValueError("SYCL-TLA supports V head dimensions 64, 96, 128, and 192")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Q, K, and V must be contiguous")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("SYCL-TLA IPC ring attention is inference-only")
    if dropout_p != 0.0:
        raise ValueError("SYCL-TLA IPC ring attention does not support dropout")
    if causal:
        raise NotImplementedError("SYCL-TLA IPC ring attention currently supports non-causal attention only")
    if window_size != (-1, -1) or softcap != 0.0 or alibi_slopes is not None:
        raise NotImplementedError(
            "SYCL-TLA IPC ring attention does not support windows, softcap, or ALiBi"
        )
    if joint_tensor_key is not None or joint_tensor_value is not None:
        raise NotImplementedError("SYCL-TLA IPC ring attention does not support joint tensors")
    if joint_strategy != "front":
        raise ValueError("joint_strategy must be 'front' when joint tensors are unused")
    default_scale = q.shape[-1] ** -0.5
    if softmax_scale is not None and not math.isclose(softmax_scale, default_scale, rel_tol=1e-6):
        raise NotImplementedError("SYCL-TLA IPC ring attention only supports the default softmax scale")

    ring = _get_or_create_ring(process_group, k, v)

    # Stage the freshly produced per-layer K/V into slot 0 of the arena and
    # synchronize so peers can safely DMA from it.
    ring.stage(k, v)

    out = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2], v.shape[3]),
        device=q.device,
        dtype=q.dtype,
    )
    lse = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )

    for step in range(ring.world_size):
        pending = ring.copy_next(step)
        read_slot = read_slot_of(step)
        fa.prefill_bf16_bshd_kv_round(
            q=q,
            k=ring.kbuf[read_slot],
            v=ring.vbuf[read_slot],
            out=out,
            lse=lse,
            round_idx=step,
        )

        for h in pending:
            h.wait()

    # Align all ranks before the next call can overwrite slot 0 via stage().
    dist.barrier(group=process_group)

    return out, lse.transpose(1, 2).contiguous()
