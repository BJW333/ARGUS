"""
LLMBackend — abstract interface for all model backends.

Every backend implements two methods:
    generate(system, user, **opts) -> str          (full response, blocking)
    stream(system, user, on_chunk, **opts) -> str  (streamed, calls on_chunk per token)

Backends know nothing about ARGUS internals (memory prefix, GA, RAG).
They take system+user text in, return assistant text out. Routing
decisions (which backend, when to fall back) live in router.py.

Subclasses MUST set the `name` class attribute and SHOULD return ""
on failure rather than raising — the router decides what to do next.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        num_predict: int = 2048,
        timeout: float = 120.0,
        stop: Optional[List[str]] = None,
        **opts,
    ) -> str:
        """Generate a full response synchronously. Returns "" on failure."""
        ...

    @abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        on_chunk: Callable[[str], None],
        *,
        temperature: float = 0.7,
        num_predict: int = 2048,
        timeout: float = 600.0,
        stop: Optional[List[str]] = None,
        **opts,
    ) -> str:
        """Stream a response, calling on_chunk(token) for each token. Returns full text or ""."""
        ...
