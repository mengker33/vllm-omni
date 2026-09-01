# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-op device timing for diffusion models.

Provides a timestamp-based alternative to the torch profiler on platforms where
the profiler is unusable. Enabled via ``VLLM_OMNI_ENABLE_DEVICE_OP_TIMING``.
"""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_TIMING_FILE_ENV = "VLLM_OMNI_DEVICE_OP_TIMING_FILE"


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


class _DeviceOpTimer:
    """Low-overhead device timing with event-first, sync fallback semantics."""

    def __init__(self) -> None:
        self._enabled = _env_flag_enabled("VLLM_OMNI_ENABLE_DEVICE_OP_TIMING", default=False)
        self._timers_ms: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._pending: list[tuple[str, Any, Any]] = []
        self._step_index: int | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        self._timers_ms.clear()
        self._counts.clear()
        self._pending.clear()
        self._step_index = None

    def begin_step(self, step_index: int) -> None:
        """Drop the previous step's samples so only the newest step is retained."""
        if not self._enabled:
            return
        # Unresolved events are discarded rather than resolved, to avoid a per-step sync.
        self._timers_ms.clear()
        self._counts.clear()
        self._pending.clear()
        self._step_index = step_index

    def _record_event(self):
        if not current_omni_platform.is_available():
            return None
        # Not current_omni_platform.record_device_event(): that builds sync-only
        # events, and elapsed_time() requires enable_timing=True.
        try:
            event = torch.Event(enable_timing=True)
            event.record()
            return event
        except Exception:
            return None

    def _add(self, name: str, elapsed_ms: float) -> None:
        self._timers_ms[name] = self._timers_ms.get(name, 0.0) + elapsed_ms
        self._counts[name] = self._counts.get(name, 0) + 1

    def _resolve_pending(self) -> None:
        if not self._pending:
            return
        current_omni_platform.synchronize()
        for name, start_evt, end_evt in self._pending:
            try:
                elapsed_ms = float(start_evt.elapsed_time(end_evt))
            except Exception as exc:
                logger.warning_once("Device op timing dropped samples for %s: %s", name, exc)
                continue
            self._add(name, elapsed_ms)
        self._pending.clear()

    @contextmanager
    def scope(self, name: str):
        if not self._enabled:
            yield
            return

        start_evt = self._record_event()
        if start_evt is not None and hasattr(start_evt, "elapsed_time"):
            try:
                yield
            finally:
                end_evt = self._record_event()
                if end_evt is not None:
                    self._pending.append((name, start_evt, end_evt))
            return

        # Fallback path for platforms without working event elapsed_time.
        current_omni_platform.synchronize()
        start_t = time.perf_counter()
        try:
            yield
        finally:
            current_omni_platform.synchronize()
            self._add(name, (time.perf_counter() - start_t) * 1000.0)

    def summary(self) -> str:
        if not self._enabled:
            return ""
        self._resolve_pending()
        if not self._timers_ms:
            return ""

        total = sum(self._timers_ms.values())
        lines = [
            f"{'op':<28} | {'total_ms':>12} | {'count':>8} | {'avg_ms':>10} | {'pct':>7}",
            "-" * 78,
        ]
        for name, elapsed in sorted(self._timers_ms.items(), key=lambda item: item[1], reverse=True):
            count = self._counts[name]
            avg = elapsed / max(count, 1)
            pct = (elapsed / total * 100.0) if total > 0 else 0.0
            lines.append(f"{name:<28} | {elapsed:>12.3f} | {count:>8d} | {avg:>10.3f} | {pct:>6.2f}%")
        lines.append("-" * 78)
        lines.append(f"{'total':<28} | {total:>12.3f}")
        return "\n".join(lines)

    def _rows(self) -> list[dict[str, Any]]:
        total = sum(self._timers_ms.values())
        rows = []
        for name, elapsed in sorted(self._timers_ms.items(), key=lambda item: item[1], reverse=True):
            count = self._counts[name]
            rows.append(
                {
                    "op": name,
                    "total_ms": round(elapsed, 6),
                    "count": count,
                    "avg_ms": round(elapsed / max(count, 1), 6),
                    "pct": round((elapsed / total * 100.0) if total > 0 else 0.0, 3),
                }
            )
        return rows

    def dump(self) -> str | None:
        """Write the retained (last) step to the path in ``VLLM_OMNI_DEVICE_OP_TIMING_FILE``."""
        if not self._enabled:
            return None
        path_str = os.getenv(_TIMING_FILE_ENV)
        if not path_str:
            return None

        self._resolve_pending()
        if not self._timers_ms:
            return None

        path = Path(path_str).expanduser()
        rows = self._rows()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".json":
                payload = {
                    "step_index": self._step_index,
                    "total_ms": round(sum(self._timers_ms.values()), 6),
                    "ops": rows,
                }
                path.write_text(json.dumps(payload, indent=2))
            else:
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["step_index", "op", "total_ms", "count", "avg_ms", "pct"])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({"step_index": self._step_index, **row})
        except OSError as exc:
            logger.warning("Failed to write device op timing file %s: %s", path, exc)
            return None
        return str(path)


_DEVICE_OP_TIMER = _DeviceOpTimer()


def is_device_op_timing_enabled() -> bool:
    return _DEVICE_OP_TIMER.enabled


def reset_device_op_timing() -> None:
    _DEVICE_OP_TIMER.reset()


def begin_device_op_timing_step(step_index: int) -> None:
    _DEVICE_OP_TIMER.begin_step(step_index)


def dump_device_op_timing() -> str | None:
    return _DEVICE_OP_TIMER.dump()


def get_device_op_timing_summary() -> str:
    return _DEVICE_OP_TIMER.summary()


def device_op_timer(name: str):
    return _DEVICE_OP_TIMER.scope(name)
