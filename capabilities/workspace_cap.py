"""
Workspace Capability — bridges existing workspace system into the registry.
==========================================================================

The workspace system (workspace/) already takes speak_fn and listen_fn
as constructor args — that's good design from v1.  This capability
just wraps it so the brain can route workspace intents through the
capability registry like everything else.

The embodiment provides speak/listen when registering this capability.
"""
from __future__ import annotations

from typing import Callable, Optional

from capabilities.registry import REGISTRY, Capability
from state.world_state import WORLD
from config_metrics.logging import log_debug


_workspace_instance = None


def get_workspace(speak_fn: Callable, listen_fn: Callable, research_fn=None):
    """Lazy-init the workspace integration (same pattern as old orchestrator)."""
    global _workspace_instance
    if _workspace_instance is None:
        from services.workspace.workspace_integration import ArgusWorkspaceIntegration
        _workspace_instance = ArgusWorkspaceIntegration(
            speak_fn=speak_fn,
            listen_fn=listen_fn,
            sensory_hub=None,
            research_fn=research_fn,
        )
    return _workspace_instance


def handle_workspace_command(command: str = "", speak_fn=None,
                             listen_fn=None, research_fn=None, **kwargs) -> str:
    """
    Capability handler for workspace commands.

    Returns a string result (or empty string if workspace handled it
    internally via speak_fn).
    """
    if speak_fn is None or listen_fn is None:
        return "Workspace requires speech I/O — not available on this embodiment."

    ws = get_workspace(speak_fn, listen_fn, research_fn)

    if ws.handle_command(command):
        # Workspace handled it (spoke its own output)
        # Update world state
        status = ws.get_status()
        WORLD.update("active_workspace", status.get("profile") if status.get("active") else None)
        WORLD.update("active_project", status.get("project_name"))
        WORLD.update("workspace_session_id", status.get("session_id"))
        return ""  # workspace spoke its own response
    else:
        return "I couldn't run that workspace command."


def register_workspace_capability(speak_fn: Callable, listen_fn: Callable,
                                   research_fn=None) -> None:
    """
    Register workspace as a capability.  Called by the embodiment at startup.

    Args:
        speak_fn:    Embodiment's speak function.
        listen_fn:   Embodiment's listen function.
        research_fn: Optional research function (Wikipedia, etc.)
    """
    REGISTRY.register(Capability(
        name="workspace.handle_command",
        description="Handle workspace voice commands (start, notes, tasks, etc.)",
        handler=lambda command="", **kw: handle_workspace_command(
            command=command,
            speak_fn=speak_fn,
            listen_fn=listen_fn,
            research_fn=research_fn,
        ),
        keywords=["workspace"],
    ))
    log_debug("[Capability] workspace.handle_command registered")
