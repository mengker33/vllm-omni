# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Level-Zero IPC K/V ring buffers for Intel XPU ring attention.

This module only owns peer-buffer setup and copy scheduling. The attention
loop and its output/LSE accumulation are implemented by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist

from .ring_globals import (
    HAS_SYCL_TLA,
    HAS_SYCL_TLA_IPC,
    sycl_tla_fmha as fa,
    sycl_tla_ipc_p2p as ipc,
)


NUM_BUFFERS = 4


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

        # Keep slot zero as an owned, fixed allocation. ``copy_`` writes into
        # this allocation and does not replace its storage, which is required
        # because the IPC mappings use these buffers as their local context.
        self.kbuf = [torch.empty_like(k_local)]
        self.vbuf = [torch.empty_like(v_local)]
        self.kbuf[0].copy_(k_local)
        self.vbuf[0].copy_(v_local)
        for _ in range(NUM_BUFFERS - 1):
            self.kbuf.append(torch.empty_like(k_local))
            self.vbuf.append(torch.empty_like(v_local))

        self.k_nbytes = self.kbuf[0].numel() * self.kbuf[0].element_size()
        self.v_nbytes = self.vbuf[0].numel() * self.vbuf[0].element_size()

        local_handles = {
            "k": [ipc.ipc_get_handle(buf, self.queue_ptr) for buf in self.kbuf],
            "v": [ipc.ipc_get_handle(buf, self.queue_ptr) for buf in self.vbuf],
        }
        gathered: list[dict[str, list[tuple[bytes, int]]]] = [None] * self.world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, local_handles, group=process_group)
        peer = gathered[self.prev_rank]

        self.k_peer_off = [int(offset) for _, offset in peer["k"]]
        self.v_peer_off = [int(offset) for _, offset in peer["v"]]
        self.k_peer_ptr = [
            ipc.ipc_open_handle(self.kbuf[0], handle, int(offset), self.queue_ptr)
            for handle, offset in peer["k"]
        ]
        self.v_peer_ptr = [
            ipc.ipc_open_handle(self.vbuf[0], handle, int(offset), self.queue_ptr)
            for handle, offset in peer["v"]
        ]

        # Do not issue a copy until every rank has opened its previous peer.
        dist.barrier()
        torch.xpu.synchronize()


    def copy_next(self, step: int) -> list[int]:
        """Copy the previous rank's block for ``step + 1`` asynchronously."""
        if step >= self.world_size - 1:
            return []

        # The copy engine is independent from PyTorch's compute queue. A
        # barrier only synchronizes the host processes and does not establish
        # a device dependency, so a later round could otherwise overwrite a
        # rotating slot while an earlier FMHA kernel is still reading it (and
        # a peer could read that slot at the same time). Synchronizing before
        # the next DMA makes slot reuse and peer-source ordering explicit while
        # retaining overlap with the FMHA kernel launched for this round.
        torch.xpu.synchronize()

        read_slot = read_slot_of(step)
        write_slot = write_slot_of(step)
        return [
            ipc.ipc_copy_from_peer_async(
                self.kbuf[write_slot],
                self.k_peer_ptr[read_slot],
                self.k_nbytes,
                self.queue_ptr,
            ),
            ipc.ipc_copy_from_peer_async(
                self.vbuf[write_slot],
                self.v_peer_ptr[read_slot],
                self.v_nbytes,
                self.queue_ptr,
            ),
        ]

    @staticmethod
    def wait(handles: Iterable[int]) -> None:
        """Wait for and release all pending copy-engine operations."""
        if ipc is None:
            raise RuntimeError("SYCL-TLA IPC P2P is unavailable")
        for handle in handles:
            ipc.ipc_wait(handle)

    def close(self) -> None:
        """Close peer mappings after all copy operations have completed."""
        if ipc is None:
            return
        for peer_ptr, offset in zip(self.k_peer_ptr, self.k_peer_off):
            try:
                ipc.ipc_close_handle(self.kbuf[0], peer_ptr, offset, self.queue_ptr)
            except Exception:
                pass
        for peer_ptr, offset in zip(self.v_peer_ptr, self.v_peer_off):
            try:
                ipc.ipc_close_handle(self.vbuf[0], peer_ptr, offset, self.queue_ptr)
            except Exception:
                pass
        self.k_peer_ptr = []
        self.v_peer_ptr = []


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

    ring = IpcKVRing(k, v, process_group=process_group)
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

    try:
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

            if step < ring.world_size - 1:
                dist.barrier()

        # Keep all ranks aligned before the mappings are closed and before a
        # subsequent ring pass can reuse the rotating buffers.
        dist.barrier()
        torch.xpu.synchronize()
    finally:
        ring.close()

    return out, lse.transpose(1, 2).contiguous()
