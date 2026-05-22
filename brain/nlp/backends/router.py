"""
RouterBackend — auto-routes between primary (API) and fallback (local).

Modes:
    "auto"  : try primary first; fall back to local on error or refusal.
    "api"   : primary only; return its result regardless of refusal.
    "local" : fallback only; never hit the API.

Refusal handling in stream():
    Streamed tokens from the primary are buffered until we have ~300 chars
    or hit a paragraph break — enough to confidently classify as refusal
    or normal response. If refusal: drop the buffer (no flicker in GUI),
    re-stream from fallback. If normal: flush buffer to GUI, continue
    streaming from primary as usual.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from brain.nlp.backends.base import LLMBackend
from brain.nlp.backends.refusal import looks_like_refusal
from config_metrics.logging import log_debug


class RouterBackend(LLMBackend):
    name = "router"

    def __init__(
        self,
        primary: LLMBackend,
        fallback: LLMBackend,
        mode: str = "auto",
        on_event: Optional[Callable] = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self._mode = mode
        self.on_event = on_event or (lambda **kw: log_debug(f"[Router] {kw}"))

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("auto", "api", "local"):
            raise ValueError(f"Invalid mode: {mode!r}. Must be 'auto', 'api', or 'local'.")
        log_debug(f"[Router] mode change: {self._mode} -> {mode}")
        self._mode = mode

    @property
    def active_backend_name(self) -> str:
        """Which underlying backend would handle the next call (primary's name in auto/api, fallback's in local)."""
        if self._mode == "local":
            return self.fallback.name
        return self.primary.name

    def _select(self):
        """Returns (first_to_try, fallback_or_None)."""
        if self._mode == "api":
            return self.primary, None
        if self._mode == "local":
            return self.fallback, None
        return self.primary, self.fallback

    # ── generate ────────────────────────────────────────────
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
        first, fb = self._select()
        kw = dict(temperature=temperature, num_predict=num_predict,
                  timeout=timeout, stop=stop, **opts)

        try:
            text = first.generate(system, user, **kw)
        except Exception as e:
            self.on_event(event="error_fallback", backend=first.name, error=str(e))
            return fb.generate(system, user, **kw) if fb else ""

        if not text and fb is not None:
            self.on_event(event="empty_fallback", backend=first.name)
            return fb.generate(system, user, **kw)

        if fb is not None and looks_like_refusal(text):
            self.on_event(event="refusal_fallback", backend=first.name, head=text[:120])
            return fb.generate(system, user, **kw)

        return text

    # ── stream ──────────────────────────────────────────────
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
        first, fb = self._select()
        kw = dict(temperature=temperature, num_predict=num_predict,
                  timeout=timeout, stop=stop, **opts)

        SNIFF_LIMIT = 300
        buf: List[str] = []
        flushed = [False]
        refused = [False]

        def _sniff(token: str):
            if flushed[0]:
                on_chunk(token)
                return
            buf.append(token)
            joined = "".join(buf)
            # Decide once we have enough text or a clear sentence boundary.
            if len(joined) >= SNIFF_LIMIT or "\n\n" in joined or ". " in joined:
                if fb is not None and looks_like_refusal(joined):
                    refused[0] = True
                    # Don't flush — fallback will re-stream.
                else:
                    flushed[0] = True
                    on_chunk(joined)

        try:
            full = first.stream(system, user, _sniff, **kw)
        except Exception as e:
            self.on_event(event="error_fallback", backend=first.name, error=str(e))
            if fb is None:
                return ""
            return fb.stream(system, user, on_chunk, **kw)

        # Refusal → re-stream from fallback (sniff buffer was never flushed).
        if refused[0] and fb is not None:
            self.on_event(event="refusal_fallback", backend=first.name, head="".join(buf)[:120])
            return fb.stream(system, user, on_chunk, **kw)

        # Very short response that finished before sniff threshold: flush now.
        if not flushed[0] and not refused[0]:
            on_chunk("".join(buf))

        # Empty primary response → try fallback.
        if not full and fb is not None:
            self.on_event(event="empty_fallback", backend=first.name)
            return fb.stream(system, user, on_chunk, **kw)

        return full
