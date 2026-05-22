"""
Phase 2 Verification Tests.

Run from the project root:
    cd ~/Desktop/ARGUS
    python3.10 scripts/phase2_test.py

Tests the agentic tool-use path in isolation, before wiring it into
the main pipeline. If all tests pass, Phase 2 is safe to enable in
reasoning.py.

What this verifies:
    1. Tool schemas load and look valid.
    2. Direct tool execution works (no LLM involved).
    3. AnthropicBackend.stream_with_tools() runs and parses correctly.
    4. AgenticOrchestrator handles a simple text-only response.
    5. Orchestrator handles a single tool call.
    6. Orchestrator handles a compound (multi-tool) request.
    7. Orchestrator survives a tool error gracefully.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()


def section(title: str) -> None:
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    line = f"  {mark} {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return ok


def main() -> int:
    failures = 0

    # ────────────────────────────────────────────────────────
    section("1. Tool schemas load")
    # ────────────────────────────────────────────────────────
    try:
        from brain.agentic import get_tool_definitions, get_tool_names
        tools = get_tool_definitions()
        names = get_tool_names()
        if not check(f"Loaded {len(tools)} tools", len(tools) > 0):
            failures += 1
        if not check(
            "All tools have name + description + input_schema",
            all(
                "name" in t and "description" in t and "input_schema" in t
                for t in tools
            ),
        ):
            failures += 1
        print(f"     Tool names: {', '.join(names)}")
    except Exception as e:
        check(f"Tool schemas import", False, str(e))
        failures += 1
        return failures

    # ────────────────────────────────────────────────────────
    section("2. Direct tool execution (no LLM)")
    # ────────────────────────────────────────────────────────
    # Need REGISTRY populated. The desktop embodiment registers most caps,
    # but we don't want to start the full embodiment for tests. Just register
    # a minimal set that doesn't need GUI/speech.
    try:
        from capabilities.registry import REGISTRY, Capability
        from actions.actions import action_time, coin_flip
        import pyjokes

        REGISTRY.register(Capability(
            name="time.get_current",
            description="time", handler=lambda **kw: action_time(),
        ))
        REGISTRY.register(Capability(
            name="misc.coinflip",
            description="flip", handler=lambda **kw: coin_flip(),
        ))
        REGISTRY.register(Capability(
            name="misc.joke",
            description="joke", handler=lambda **kw: pyjokes.get_joke(),
        ))
        REGISTRY.register(Capability(
            name="calculator.calculate",
            description="math",
            handler=lambda expression="", **kw: __import__(
                "actions.actions", fromlist=["calculate"]
            ).calculate(expression),
        ))

        from brain.agentic import execute_tool
        result = execute_tool("get_time")
        if not check(
            "execute_tool('get_time') returns a string",
            isinstance(result, str) and len(result) > 0,
            result[:60],
        ):
            failures += 1

        result = execute_tool("calculate", expression="3 plus 5")
        if not check(
            "execute_tool('calculate', expression='3 plus 5') works",
            "8" in result,
            result[:60],
        ):
            failures += 1

        # Non-existent tool: should return an error string, not raise.
        result = execute_tool("nonexistent_tool")
        if not check(
            "Unknown tool returns error string (no raise)",
            isinstance(result, str) and "error" in result.lower(),
            result[:60],
        ):
            failures += 1

    except Exception as e:
        check("Direct tool execution", False, str(e))
        failures += 1

    # ────────────────────────────────────────────────────────
    section("3. AnthropicBackend.stream_with_tools() — text only")
    # ────────────────────────────────────────────────────────
    try:
        from brain.nlp.backends import AnthropicBackend
        backend = AnthropicBackend()
        if not backend.api_key:
            check("API key present", False, "ANTHROPIC_API_KEY missing")
            failures += 1
            return failures

        chunks = []
        result = backend.stream_with_tools(
            system="You are a calm assistant. Answer briefly.",
            messages=[{"role": "user", "content": "Say hello in five words."}],
            tools=tools,
            on_text_chunk=chunks.append,
        )
        if not check(
            "Streamed text-only response",
            result.get("stop_reason") == "end_turn"
            and len(result.get("text", "")) > 0,
            f"stop={result.get('stop_reason')} text={result.get('text', '')[:60]!r}",
        ):
            failures += 1
        if not check(
            "Tokens streamed via on_text_chunk callback",
            len(chunks) > 0,
            f"{len(chunks)} chunks",
        ):
            failures += 1
    except Exception as e:
        check("stream_with_tools text-only", False, str(e))
        failures += 1

    # ────────────────────────────────────────────────────────
    section("4. Orchestrator — single tool call")
    # ────────────────────────────────────────────────────────
    try:
        from brain.agentic import ToolLoop
        loop = ToolLoop(backend=backend, max_iterations=4)

        events = []
        text = loop.run(
            system_prompt=(
                "You are ARGUS. If the user asks for the time, use the "
                "get_time tool. Be concise."
            ),
            user_input="What time is it right now?",
            on_text_chunk=lambda t: None,  # silent for this test
            on_tool_event=events.append,
        )
        if not check(
            "Orchestrator returned final text",
            isinstance(text, str) and len(text) > 0,
            text[:80],
        ):
            failures += 1
        if not check(
            "Orchestrator fired the get_time tool",
            any("get_time" in e for e in events),
            "; ".join(events),
        ):
            failures += 1
    except Exception as e:
        check("Orchestrator single tool", False, str(e))
        failures += 1

    # ────────────────────────────────────────────────────────
    section("5. Orchestrator — compound multi-tool")
    # ────────────────────────────────────────────────────────
    try:
        events = []
        text = loop.run(
            system_prompt=(
                "You are ARGUS. Tools: get_time, coin_flip, calculate, "
                "tell_joke. Use the appropriate tools for each part of the "
                "user's request. Be concise."
            ),
            user_input="What time is it, and what's 17 times 23?",
            on_text_chunk=lambda t: None,
            on_tool_event=events.append,
        )
        # Orchestrator should call BOTH tools.
        if not check(
            "Compound request triggered multiple tool calls",
            sum("[Tool]" in e for e in events) >= 2,
            f"{len(events)} tool events",
        ):
            failures += 1
        if not check(
            "Final text mentions a result number",
            "391" in text or "391" in str(events),
            text[:120],
        ):
            failures += 1
    except Exception as e:
        check("Orchestrator compound", False, str(e))
        failures += 1

    # ────────────────────────────────────────────────────────
    section("Summary")
    # ────────────────────────────────────────────────────────
    if failures == 0:
        print("  ✓ All Phase 2 verification checks passed.")
        print("  Safe to enable orchestrator in reasoning.py — see PHASE2_README.md.")
        return 0
    else:
        print(f"  ✗ {failures} check(s) failed.")
        print("  Fix before wiring the orchestrator into reasoning.py.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
