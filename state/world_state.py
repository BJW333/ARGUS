"""
ARGUS WorldState — shared live context for the entire system.
=============================================================

Every module reads / writes here. Nobody holds private state that
other modules need.  The brain reads it before thinking. Embodiments
write sensor data into it. Services query it for context.

Thread-safe. Callback-driven. Version-tracked.

Usage:
    from state.world_state import WORLD

    WORLD.update("user_input", "what's the weather")
    val = WORLD.get("user_input")

    # React to changes
    def on_change(key, old_val, new_val):
        ...
    WORLD.register_callback(on_change)

    # Batch updates (single version bump)
    with WORLD.batch():
        WORLD.update("active_workspace", "coding")
        WORLD.update("task_state", "executing")
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


class WorldState:
    """Thread-safe key-value store with change callbacks and version counter."""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[Any, float]] = {}      # key → (value, timestamp)
        self._lock = threading.RLock()
        self._version: int = 0
        self._callbacks: List[Callable] = []
        self._batch_depth: int = 0
        self._batch_pending: List[Tuple[str, Any, Any]] = []

    # ── reads ────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return current value for *key*, or *default*."""
        with self._lock:
            entry = self._data.get(key)
            return entry[0] if entry is not None else default

    def get_timestamp(self, key: str) -> Optional[float]:
        """Return the unix timestamp of the last write to *key*."""
        with self._lock:
            entry = self._data.get(key)
            return entry[1] if entry is not None else None

    def snapshot(self) -> Tuple[int, Dict[str, Any]]:
        """
        Return (version, {key: value}) — a frozen copy safe to hand
        to the brain for reasoning without holding any lock.
        """
        with self._lock:
            data = {k: v for k, (v, _ts) in self._data.items()}
            return self._version, data

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    # ── writes ───────────────────────────────────────────

    def update(self, key: str, value: Any) -> None:
        """Set *key* to *value*.  No-op if value hasn't changed."""
        with self._lock:
            old_entry = self._data.get(key)
            old_val = old_entry[0] if old_entry is not None else None
            if old_val == value:
                return
            self._data[key] = (value, time.time())
            self._version += 1

            if self._batch_depth > 0:
                self._batch_pending.append((key, old_val, value))
            else:
                self._fire(key, old_val, value)

    def delete(self, key: str) -> None:
        """Remove a key entirely."""
        with self._lock:
            if key in self._data:
                old_val = self._data.pop(key)[0]
                self._version += 1
                self._fire(key, old_val, None)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """
        Group several updates — callbacks fire once at the end.
        
            with WORLD.batch():
                WORLD.update("key1", val1)
                WORLD.update("key2", val2)
            # callbacks fire here
        """
        with self._lock:
            self._batch_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    pending = self._batch_pending[:]
                    self._batch_pending.clear()
            # fire outside the lock to avoid deadlocks in callbacks
            for key, old, new in pending:
                self._fire(key, old, new)

    # ── callbacks ────────────────────────────────────────

    def register_callback(self, fn: Callable) -> None:
        """Register a function called as fn(key, old_value, new_value) on changes."""
        with self._lock:
            self._callbacks.append(fn)

    def unregister_callback(self, fn: Callable) -> None:
        with self._lock:
            try:
                self._callbacks.remove(fn)
            except ValueError:
                pass

    def _fire(self, key: str, old: Any, new: Any) -> None:
        for cb in list(self._callbacks):
            try:
                cb(key, old, new)
            except Exception:
                pass  # TODO: log in production

    # ── metadata ─────────────────────────────────────────

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def __repr__(self) -> str:
        with self._lock:
            return f"<WorldState v{self._version} keys={list(self._data.keys())}>"


# ── module-level singleton ──
WORLD = WorldState()
