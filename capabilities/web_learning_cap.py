"""
Web Learning Capabilities
=========================

Registers web learning commands (start/stop scraper, update knowledge)
as capabilities in the registry.
"""
from __future__ import annotations

from capabilities.registry import REGISTRY, Capability
from config_metrics.logging import log_debug

_scraper_service = None
_web_rag = None


def init_web_learning(scraper_service, web_rag) -> None:
    """Store references and register capabilities.  Called at startup."""
    global _scraper_service, _web_rag
    _scraper_service = scraper_service
    _web_rag = web_rag

    REGISTRY.register(Capability(
        name="web.update_knowledge",
        description="Ingest new web scraper data into RAG",
        handler=_update_knowledge,
    ))
    REGISTRY.register(Capability(
        name="web.start_learning",
        description="Start the background web scraper",
        handler=_start_learning,
    ))
    REGISTRY.register(Capability(
        name="web.stop_learning",
        description="Stop the background web scraper",
        handler=_stop_learning,
    ))
    log_debug("[Capability] web learning capabilities registered")


def _update_knowledge(**kwargs) -> str:
    if _web_rag is None:
        return "Web RAG is not available."
    try:
        added = _web_rag.ingest_new()
        return f"Web knowledge updated. Added {added} new chunks."
    except Exception as e:
        return f"Web knowledge update failed: {e}"


def _start_learning(**kwargs) -> str:
    if _scraper_service is None:
        return "Web scraper service is not available."
    try:
        _scraper_service.enabled = True
        _scraper_service.start()
        return "Web learning started."
    except Exception as e:
        return f"Could not start web learning: {e}"


def _stop_learning(**kwargs) -> str:
    if _scraper_service is None:
        return "Web scraper service is not available."
    try:
        _scraper_service.stop()
        return "Web learning stopped."
    except Exception as e:
        return f"Could not stop web learning: {e}"
