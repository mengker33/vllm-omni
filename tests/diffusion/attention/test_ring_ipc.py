# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-rank correctness tests for the Intel XPU IPC ring backend."""

import socket

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.ring.ring_globals import HAS_SYCL_TLA
from vllm_omni.diffusion.attention.backends.ring.ring_ipc import (
    IpcKVRing,
    read_slot_of,
)
from vllm_omni.diffusion.attention.backends.ring.ring_ipc import ring_ipc_attn_forward


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available() or not HAS_SYCL_TLA,
    reason="requires an available XPU and both SYCL-TLA extensions",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_ipc_ring_attention_matches_full_sdpa() -> None:
    """Verify K/V rotation and fused LSE accumulation across two XPU ranks."""
    world_size = 2
    if torch.xpu.device_count() < world_size:
        pytest.skip(f"test requires {world_size} XPU devices")
    torch.multiprocessing.spawn(
        _ring_attention_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


def _ring_attention_worker(rank: int, world_size: int, port: int) -> None:
    device = torch.device("xpu", rank)
    torch.xpu.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        batch = 1
        global_seq = 8192
        local_seq = global_seq // world_size
        num_heads = 40
        qk_dim = 128
        v_dim = 128

        torch.manual_seed(2026)
        q_full = torch.randn(
            batch, global_seq, num_heads, qk_dim, device=device, dtype=torch.bfloat16
        )
        k_full = torch.randn(
            batch, global_seq, num_heads, qk_dim, device=device, dtype=torch.bfloat16
        )
        v_full = torch.randn(
            batch, global_seq, num_heads, v_dim, device=device, dtype=torch.bfloat16
        )
        start = rank * local_seq
        end = start + local_seq
        q = q_full[:, start:end].contiguous()
        k = k_full[:, start:end].contiguous()
        v = v_full[:, start:end].contiguous()

        # Validate the communication independently before running FMHA.
        ring = IpcKVRing(k, v, process_group=dist.group.WORLD)
        for step in range(world_size):
            pending = ring.copy_next(step)
            ring.wait(pending)
            read_slot = read_slot_of(step)
            source = (rank - step) % world_size
            source_start = source * local_seq
            source_end = source_start + local_seq
            torch.xpu.synchronize(device)
            torch.testing.assert_close(
                ring.kbuf[read_slot], k_full[:, source_start:source_end]
            )
            torch.testing.assert_close(
                ring.vbuf[read_slot], v_full[:, source_start:source_end]
            )
            if step < world_size - 1:
                dist.barrier()
        dist.barrier()
        ring.close()

        out, _ = ring_ipc_attn_forward(
            process_group=dist.group.WORLD,
            q=q,
            k=k,
            v=v,
            causal=False,
        )
        torch.xpu.synchronize(device)

        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_full.transpose(1, 2),
            v_full.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()
        torch.testing.assert_close(out.float(), reference.float(), atol=5e-2, rtol=5e-2)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
