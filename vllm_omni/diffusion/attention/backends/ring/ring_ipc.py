# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Level-Zero IPC K/V ring buffers for Intel XPU ring attention.

This module only owns peer-buffer setup and copy scheduling. The attention
loop and its output/LSE accumulation are implemented by the caller.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .ring_globals import (
    HAS_SYCL_TLA,
    HAS_SYCL_TLA_IPC,
    sycl_tla_fmha as fa,
    sycl_tla_ipc_p2p as ipc,
)


NUM_BUFFERS = 4
SLOT_ALIGN = 4096


def _align_up(value: int, alignment: int = SLOT_ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


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
    """Persistent local K/V buffers and peer IPC mappings for one ring group.

    Buffer zero permanently contains the rank's original local block. The
    other buffers are rotating copy destinations. Handles are gathered in
    process-group local-rank order, so subgroup global-rank numbering does not
    affect peer selection.
    """

    def __init__(
        self,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        process_group: dist.ProcessGroup | None = None,
        queue_ptr: int | None = None,
    ) -> None:
        if not HAS_SYCL_TLA_IPC or ipc is None:
            raise RuntimeError(
                "SYCL-TLA IPC P2P is unavailable; install the "
                "sycl_tla_ipc_p2p extension or use the PyTorch ring backend"
            )
        if not hasattr(torch, "xpu") or not k_local.device.type == "xpu":
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
        self.queue_ptr = current_queue_ptr(k_local.device) if queue_ptr is None else queue_ptr

        self.dev_index = k_local.device.index
        self.k_shape = list(k_local.shape)
        self.v_shape = list(v_local.shape)
        self.k_nbytes = k_local.numel() * k_local.element_size()
        self.v_nbytes = v_local.numel() * v_local.element_size()

        # Keep every slot in one stable arena. Slot zero is initialized by
        # stage() and is never a copy destination.
        self.v_off_in_slot = _align_up(self.k_nbytes)
        self.slot_stride = _align_up(self.v_off_in_slot + self.v_nbytes)
        self.k_off = [i * self.slot_stride for i in range(NUM_BUFFERS)]
        self.v_off = [i * self.slot_stride + self.v_off_in_slot for i in range(NUM_BUFFERS)]
        self.arena = ipc.make_arena(
            self.dev_index, self.slot_stride * NUM_BUFFERS, self.queue_ptr
        )
        self.kbuf = [
            self.arena.view(offset, self.k_shape, k_local.dtype) for offset in self.k_off
        ]
        self.vbuf = [
            self.arena.view(offset, self.v_shape, v_local.dtype) for offset in self.v_off
        ]
        self.stage(k_local, v_local)

        # All ranks use identical offsets, so exchange only the arena base.
        local_handle = self.arena.export_handle()
        gathered: list[bytes] = [None] * self.world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_handle, group=process_group)
        self.peer_handle = gathered[self.prev_rank]
        peer_base = ipc.open_peer(self.dev_index, self.peer_handle, self.queue_ptr)
        self.k_peer_ptr = [peer_base + offset for offset in self.k_off]
        self.v_peer_ptr = [peer_base + offset for offset in self.v_off]
        self._closed = False

        # Do not issue a copy until every rank has opened its previous peer.
        dist.barrier(group=self.process_group)
        torch.xpu.synchronize(k_local.device)

    def stage(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Copy fresh local K/V into the fixed-address slot zero."""
        self.kbuf[0].copy_(k_new)
        self.vbuf[0].copy_(v_new)
        torch.xpu.current_stream(self.kbuf[0].device).synchronize()


    def copy_next(self, step: int) -> list:
        """Copy the previous rank's block for ``step + 1`` asynchronously."""
        if step >= self.world_size - 1:
            return []

        read_slot = read_slot_of(step)
        write_slot = write_slot_of(step)
        base = self.arena.base_ptr()
        return [
            ipc.copy_async(
                self.dev_index,
                base + self.k_off[write_slot],
                self.k_peer_ptr[read_slot],
                self.k_nbytes,
                self.queue_ptr,
            ),
            ipc.copy_async(
                self.dev_index,
                base + self.v_off[write_slot],
                self.v_peer_ptr[read_slot],
                self.v_nbytes,
                self.queue_ptr,
            ),
        ]

    @staticmethod
    def wait(handles) -> None:
        """Wait for and release the latest binding's PendingCopy objects."""
        for handle in handles:
            handle.wait()

    def close(self) -> None:
        """Close peer mappings after all copy operations have completed."""
        if ipc is None or self._closed:
            return
        self._closed = True
        try:
            torch.xpu.synchronize(self.kbuf[0].device)
            ipc.close_peer(self.dev_index, self.peer_handle, self.queue_ptr)
        except Exception:
            pass
        self.k_peer_ptr = []
        self.v_peer_ptr = []
        self.kbuf = []
        self.vbuf = []


# IpcArena allocations are retained by the extension until its process-level
# shutdown.  Keep one ring per K/V layout instead of allocating a new arena
# for every attention invocation.
_IPC_RING_CACHE: dict[tuple, IpcKVRing] = {}


def _get_cached_ring(
    process_group: dist.ProcessGroup | None,
    k: torch.Tensor,
    v: torch.Tensor,
) -> IpcKVRing:
    key = (
        id(process_group),
        k.device.index,
        tuple(k.shape),
        tuple(v.shape),
        k.dtype,
        v.dtype,
    )
    ring = _IPC_RING_CACHE.get(key)
    if ring is None:
        ring = IpcKVRing(k, v, process_group=process_group)
        _IPC_RING_CACHE[key] = ring
    else:
        ring.stage(k, v)
    # Synchronize staging across ranks before the first copy engine transfer.
    # The reference implementation keeps this synchronization at the ring
    # pass boundary rather than inserting barriers between rounds.
    dist.barrier(group=process_group)

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

    ring = _get_cached_ring(process_group, k, v)
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

        ring.wait(pending)

        if step < ring.world_size - 1 and k.shape[1] <= 4096:
        # Synchronize before the next copy engine transfer for small sequence lengths.
            dist.barrier(group=process_group)

    torch.xpu.synchronize()

    return out, lse.transpose(1, 2).contiguous()
