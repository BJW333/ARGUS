"""
Agentic Orchestrator — The streaming tool-use loop.

This is the heart of Phase 2. Replaces the planner's brittle JSON-decomposition
with Claude's native tool-use API. Claude can call zero, one, or many tools
in a single turn, see results, and adapt — true agentic behavior.

Flow:
    1. User input arrives
    2. Send to Claude with full tool list
    3. Stream Claude's response. Three things can happen per "block":
        a) Text block      → stream tokens to GUI in real-time
        b) Tool-use block  → buffer tool name + input until block_stop
                              then execute via tool_schemas.execute_tool()
        c) Multiple of above
    4. If any tools were called, send tool_results back to Claude
       (this is "turn 2" — Claude sees results and continues)
    5. Repeat until Claude returns a final text-only turn (stop_reason="end_turn")
    6. Return the assembled final text

Why streaming matters:
    Without it: user waits 3-5 seconds in silence while tools execute.
    With it: "Let me check the weather..." appears instantly while the tool
    runs in the background. Massive perceived-latency improvement.

Safety rails:
    - max_iterations: cap on tool-use cycles (default 6) so a confused model
      can't infinite-loop us.
    - per-tool timeout enforced inside execute_tool itself.
    - Each iteration logged, easy to trace what Claude did.
"""
from __future__ import annotations

import json
from typing import Callable, List, Dict, Any, Optional

from brain.agentic.tool_schemas import get_tool_definitions, execute_tool
from config_metrics.logging import log_debug


class ToolLoop:
    """
    Drives the multi-turn tool-use loop.

    Owned by the ReasoningPipeline. Created once, reused per turn.

    Usage:
        orch = AgenticOrchestrator(backend=anthropic_backend)
        text = orch.run(
            system_prompt=identity + memory_prefix,
            user_input="open Chrome and tell me the weather",
            on_text_chunk=stream_chunk_to_gui,
            on_tool_event=print_to_gui,
        )
    """

    def __init__(
        self,
        backend,
        max_iterations: int = 6,
    ):
        """
        Args:
            backend: An AnthropicBackend instance with stream_with_tools support.
            max_iterations: Hard cap on tool-call rounds per user input.
                Six covers the vast majority of compound requests; if Claude
                hasn't finished by then it's likely confused — cut the loop.
        """
        self.backend = backend
        self.max_iterations = max_iterations

    def run(
        self,
        system_prompt: str,
        user_input: str,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_tool_event: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute the tool-use loop and return the final assembled text.

        Args:
            system_prompt: Combined identity + memory prefix.
            user_input:    The user's raw input.
            on_text_chunk: Callback per text token (for GUI streaming).
            on_tool_event: Callback for tool-call status messages
                           (e.g. "[Tool] get_weather(city='Syracuse')").
                           Useful for the debug pane; safe to leave None.

        Returns:
            The final assembled text response. May be empty if Claude
            chose to only call tools and report nothing.
        """
        tools = get_tool_definitions()
        log_debug(f"[Orchestrator] starting; {len(tools)} tools available")

        # Conversation tracker. Each entry is a {"role", "content"} dict
        # in Anthropic's expected format. We keep growing it across rounds.
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": user_input}
        ]

        final_text_parts: List[str] = []

        for iteration in range(1, self.max_iterations + 1):
            log_debug(f"[Orchestrator] iter {iteration}")

            # ── Stream Claude's response for this turn ──
            stream_result = self.backend.stream_with_tools(
                system=system_prompt,
                messages=messages,
                tools=tools,
                on_text_chunk=on_text_chunk,
            )

            # stream_result contains:
            #   "stop_reason"   : "end_turn" | "tool_use" | "max_tokens" | ...
            #   "content_blocks": List of either {"type": "text", "text": ...}
            #                     or {"type": "tool_use", "id": ..., "name": ..., "input": ...}
            #   "text"          : the assembled text portion (already streamed via on_text_chunk)

            stop_reason = stream_result.get("stop_reason", "")
            blocks = stream_result.get("content_blocks", [])
            text_part = stream_result.get("text", "")
            if text_part:
                final_text_parts.append(text_part)

            # Claude is done — no tool calls in this turn.
            if stop_reason != "tool_use":
                log_debug(f"[Orchestrator] done; stop_reason={stop_reason}")
                break

            # ── There were tool calls. Append assistant turn (with the
            #    tool_use blocks Claude produced), then execute and append
            #    a user turn with the matching tool_result blocks. ──
            messages.append({"role": "assistant", "content": blocks})

            tool_results: List[Dict[str, Any]] = []
            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                tool_name = block.get("name", "")
                tool_input = block.get("input", {}) or {}
                tool_id = block.get("id", "")

                log_debug(
                    f"[Orchestrator] tool_use: {tool_name}({json.dumps(tool_input)[:200]})"
                )
                if on_tool_event:
                    try:
                        preview = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
                        on_tool_event(f"[Tool] {tool_name}({preview})")
                    except Exception:
                        pass

                # Execute the tool. Errors are returned as strings — Claude
                # will see them and can decide to retry, ask for clarification,
                # or give up gracefully.
                result_text = execute_tool(tool_name, **tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            # for-else: ran the full max_iterations without breaking.
            log_debug(f"[Orchestrator] hit max_iterations={self.max_iterations}")
            final_text_parts.append(
                "\n\n[Note: I made several tool calls but didn't reach a "
                "complete answer. Try rephrasing.]"
            )

        return "\n".join(p for p in final_text_parts if p).strip()
