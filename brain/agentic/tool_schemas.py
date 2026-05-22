"""
Tool Schemas — Maps capabilities to Anthropic tool-use schemas.

Phase 2 introduces native tool-use. When the intent classifier doesn't match
a capability directly (compound commands, novel queries, agentic tasks),
the orchestrator hands control to Claude with this list of tools available.

Claude decides whether to call zero, one, or many tools and chains them
based on results. This replaces the brittle JSON-decomposition planner.

Why a separate schema layer:
    The capability registry has internal-facing params ("raw_text", "entities")
    designed for the intent-classifier pipeline. Claude's tool-use needs
    clean, structured params Claude can fill in directly. This module is the
    adapter — each tool wrapper translates Claude's structured input into
    the shape existing capability handlers expect.

To add a new tool:
    1. Add a TOOL_DEFINITIONS entry below with name, description, input_schema.
    2. Add the corresponding handler in TOOL_HANDLERS — a callable that takes
       the structured kwargs and returns a string result.
    3. That's it. The orchestrator picks it up automatically.

Tool name conventions (Anthropic constraint: ^[a-zA-Z0-9_-]{1,128}$):
    - lowercase_with_underscores, no dots
    - verb_noun pattern: get_weather, open_app, calculate
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List
from brain.skills.registry import SKILL_REGISTRY


# ════════════════════════════════════════════════════════════════════
# Tool definitions
# Each entry produces an Anthropic-format tool schema sent to Claude.
# ════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "ask_user",
        "description": (
            "Ask the user a clarifying question to get information you need "
            "to complete a task — current location, which app, which file, "
            "which time zone, etc. The user's spoken answer is returned as "
            "the tool result. "
            ""
            "USE THIS TOOL WHENEVER you need real-time, current, or "
            "user-specific information that another tool requires. Examples: "
            "weather requires current city → call ask_user. Calendar query "
            "needs which calendar → call ask_user. Mail send needs the "
            "recipient → call ask_user. "
            ""
            "Do NOT pull location, names, or other live values from prior "
            "conversation or memory — those values may be stale or wrong. "
            "Memory is for personality, preferences, and context, NOT for "
            "current state. When in doubt, ask. "
            ""
            "If the user is typing rather than speaking, this tool returns "
            "a hint instead of listening; in that case make a sensible "
            "assumption, state it explicitly in your final response, and "
            "proceed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user, phrased naturally.",
                },
            },
            "required": ["question"],
        },
    },
]


# ════════════════════════════════════════════════════════════════════
# Tool handlers
# Each handler takes the structured kwargs Claude provides and returns
# a string result. Handlers may call existing capability handlers, but
# they own the param translation.
# ════════════════════════════════════════════════════════════════════

def _make_handlers() -> Dict[str, Callable[..., str]]:
    """
    Lazy-build the handler map. Imports are inside the function so this
    module loads cheaply at startup; capability modules are only pulled
    when the orchestrator actually fires a tool.
    """
    from capabilities.registry import REGISTRY

    def _call_capability(cap_name: str, **kwargs) -> str:
        """Run a capability handler by name. Returns string result."""
        cap = REGISTRY.get(cap_name)
        if cap is None:
            return f"[error] capability '{cap_name}' is not registered"
        try:
            result = cap.handler(**kwargs)
        except Exception as e:
            return f"[error] {cap_name}: {e}"

        if result is None:
            return "Done."
        if isinstance(result, str):
            # Strip the __ASK__ pattern — convert to a clarifying message
            # Claude will see and relay to the user conversationally.
            if result.startswith("__ASK__:"):
                return f"[need_more_info] {result[8:]}"
            return result
        if isinstance(result, dict):
            return result.get("message") or str(result)
        return str(result)

    # ── Tool wrappers ──
    def h_ask_user(question: str, **_) -> str:
        """
        Ask the user a clarifying question.

        Voice in: speaks the question, listens for the voice answer, returns it.
        Text in: returns a hint telling Claude to make an assumption instead.
        """
        from brain.agentic import io_hooks
        try:
            from state.world_state import WORLD
            source = WORLD.get("input_source", "voice")
        except Exception:
            source = "voice"

        if source == "voice" and io_hooks.has_voice():
            io_hooks.speak(question)
            # Routes through the unified listener. If the user says
            # "argus ..." during the wait, that's treated as an interrupt
            # and request_voice_answer returns None — Claude then sees the
            # message below and proceeds with assumptions OR aborts cleanly
            # depending on context.
            answer = io_hooks.request_voice_answer(timeout=12.0)
            if answer:
                return answer
            return "[user did not respond or interrupted — proceed with best-effort assumption]"
        
        # Typed input mode (or no voice available): don't try to listen.
        # Tell Claude to make a sensible assumption and continue.
        return (
            "[user is typing, not speaking — do NOT relay this question. "
            "Make a reasonable assumption based on context and proceed with "
            "the task. State your assumption in your final response so the "
            "user can correct it if wrong.]"
        )
        
    return {
        "ask_user":          h_ask_user,
    }

# Lazy singleton — built on first access.
_HANDLERS_CACHE: Dict[str, Callable[..., str]] = {}

def get_tool_definitions(embodiment: str = "desktop") -> List[Dict[str, Any]]:
    base = list(TOOL_DEFINITIONS)
    skill_tools = SKILL_REGISTRY.tool_definitions(embodiment=embodiment)
    existing = {t["name"] for t in skill_tools}
    base = [t for t in base if t["name"] not in existing]
    return base + skill_tools

def get_tool_names() -> List[str]:
    """Names of all tools available."""
    return [t["name"] for t in TOOL_DEFINITIONS]

def execute_tool(name: str, **kwargs) -> str:
    if SKILL_REGISTRY.get(name) is not None:
        return SKILL_REGISTRY.execute(name, **kwargs)

    global _HANDLERS_CACHE
    if not _HANDLERS_CACHE:
        _HANDLERS_CACHE = _make_handlers()
    handler = _HANDLERS_CACHE.get(name)
    if handler is None:
        return f"[error] unknown tool: {name!r}"
    try:
        return handler(**kwargs)
    except TypeError as e:
        return f"[error] bad arguments for {name}: {e}"
    except Exception as e:
        return f"[error] {name} failed: {e}"
 