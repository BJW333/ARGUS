"""
brain.agentic — Native tool-use orchestration for ARGUS.

Phase 2: replaces the planner's brittle JSON decomposition with Claude's
native tool-use API. The orchestrator drives a streaming loop where Claude
can call tools, see results, and continue reasoning — all in a single turn
from the user's perspective.

Public API:
    AgenticOrchestrator  — the loop driver
    get_tool_definitions — full tool list for API calls
    execute_tool         — direct invocation if you need it
"""
from brain.agentic.tool_loop import ToolLoop
from brain.agentic.tool_schemas import (
    get_tool_definitions,
    get_tool_names,
    execute_tool,
)
from brain.agentic import io_hooks

__all__ = [
    "ToolLoop",
    "get_tool_definitions",
    "get_tool_names",
    "execute_tool",
    "io_hooks",
]
