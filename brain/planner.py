"""
ARGUS Planner — structured multi-step reasoning and execution.
===============================================================

This is where ARGUS evolves beyond single-intent routing.

Today (v1 — pass-through):
    User says "what's the weather" → one intent → one capability → done.
    The planner just wraps the existing single-step flow.

Tomorrow (v2 — multi-step):
    User says "set up my coding environment and show me what I was working on"
    Planner decomposes into:
        1. Start coding workspace
        2. Open IDE
        3. Open terminal
        4. Restore recent project
        5. Summarize last session

    Same planner for robotics:
        1. Look at user
        2. Confirm target object
        3. Navigate closer
        4. Grasp object
        5. Report completion

The planner takes:
    - user request (text)
    - current world state (snapshot)
    - available capabilities (from registry)
    - memory context

And produces:
    - ordered list of steps (Decision dicts)
    - fallback behavior on failure
    - confirmation requirements

Platform-agnostic.  No PySide6, no speech, no GUI imports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json
import re

from state.world_state import WORLD
from capabilities.registry import REGISTRY
from config_metrics.logging import log_debug


@dataclass
class PlanStep:
    """A single step in a plan."""
    description: str                    # Human-readable: "open the IDE"
    decision: dict                      # Decision dict for execution
    requires_confirmation: bool = False # Ask user before executing?
    fallback: Optional[str] = None      # What to do if this step fails
    depends_on: Optional[int] = None    # Index of step this depends on


@dataclass
class Plan:
    """An ordered sequence of steps to achieve a goal."""
    goal: str                           # What we're trying to accomplish
    steps: List[PlanStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_single_step(self) -> bool:
        return len(self.steps) <= 1

    def as_decisions(self) -> List[dict]:
        """Extract the Decision dicts for execution."""
        return [step.decision for step in self.steps]


class Planner:
    """
    Decomposes user goals into executable plans.

    v1 (current): Pass-through — wraps a single decision into a Plan.
    v2 (future):  LLM-powered decomposition using world state + capabilities.
    """

    def __init__(self, brain_core=None):
        """
        Args:
            brain_core: Reference to BrainCore for LLM access (future use).
        """
        self._brain = brain_core

    # DEAD: bypassed by main_desktop.py v2_handler (uses agentic loop instead)
    def plan(self, user_input: str, decisions: List[dict]) -> Plan:
        """
        Create a Plan from brain decisions.

        v1: Just wraps the existing decisions.  No decomposition yet.
        This is the integration point — call this between brain.process()
        and embodiment.run_decisions() when you're ready.

        Args:
            user_input: Original user request.
            decisions:  Decision dicts from BrainCore.process().

        Returns:
            Plan with ordered steps.
        """
        plan = Plan(goal=user_input)

        for i, decision in enumerate(decisions):
            step = PlanStep(
                description=self._describe(decision),
                decision=decision,
            )
            plan.steps.append(step)

        log_debug(f"[Planner] Plan: {len(plan.steps)} steps for '{user_input}'")
        return plan

    # DEAD: bypassed by main_desktop.py v2_handler (uses agentic loop instead)
    def decompose(self, user_input: str, world_snapshot: dict) -> Plan:
        """
        Decide if a request needs multi-step decomposition.
        
        Simple requests ("what's the weather") go straight through brain.process().
        Compound requests ("open VS Code and the terminal") get sent to the LLM
        to break into ordered capability calls.
        
        The LLM only gets called for decomposition if the request looks compound.
        Single-step requests never hit the LLM here — they go through the normal
        intent classifier → capability routing in brain.process().
        """
        if self._brain is None:
            return Plan(goal=user_input)

        # Check if this looks like a multi-step request
        if not self._looks_compound(user_input):
            # Single-step: use the normal brain routing
            decisions = self._brain.process(user_input)
            return self.plan(user_input, decisions)

        # Multi-step: ask the LLM to decompose into capability calls
        embodiment = world_snapshot.get("active_embodiment", "desktop")
        llm_plan = self._llm_decompose(user_input, embodiment)
        
        if llm_plan and llm_plan.steps:
            log_debug(f"[Planner] LLM decomposition: {len(llm_plan.steps)} steps")
            return llm_plan

        # LLM decomposition failed — fall back to normal brain routing
        log_debug("[Planner] LLM decomposition failed, falling back to brain.process()")
        decisions = self._brain.process(user_input)
        return self.plan(user_input, decisions)

    def _looks_compound(self, text: str) -> bool:
        """
        Ask the LLM whether this is a single task or multiple tasks.

        One cheap low-token call replaces brittle regex rules.
        The model understands grammar so it correctly handles:
          - "write a program that does X and uses Y" → single
          - "open Chrome and set a timer for 5 minutes" → compound
        """
        # Fast exit: very short inputs are never compound
        if len(text.split()) <= 4:
            return False

        # Obvious sequential signals — skip the LLM call entirely
        quick_signals = [" and then ", " then ", " after that ",
                         " followed by "]
        lower = text.lower()
        for signal in quick_signals:
            if signal in lower:
                return True

        # No chatbot wired up — can't classify, assume single
        if self._brain is None or self._brain.chatbot is None:
            return False

        # system_prompt = (
        #     "You are a request classifier. Your ONLY job is to decide "
        #     "whether the user's message contains ONE task or MULTIPLE "
        #     "separate tasks.\n\n"
        #     "ONE task examples:\n"
        #     "  - 'write a program that calculates interest and displays a chart'\n"
        #     "  - 'find me a laptop that is lightweight and has good battery'\n"
        #     "  - 'explain how TCP works and why it matters'\n"
        #     "  - 'build a trading bot that uses RSI and MACD indicators'\n\n"
        #     "MULTIPLE task examples:\n"
        #     "  - 'open Chrome and set a timer for 10 minutes'\n"
        #     "  - 'check the weather and play some music'\n"
        #     "  - 'close Spotify and open VS Code'\n\n"
        #     "Respond with ONLY the word 'single' or 'multiple'. "
        #     "Nothing else."
        # )

        # try:
        #     answer = self._brain.chatbot._classify(system_prompt, text)
        # user_prompt = (
        #     'Classify as "single" or "multiple".\n'
        #     "single = one task, even if it has multiple requirements\n"
        #     "multiple = two or more completely different tasks\n\n"
        #     "Examples:\n"
        #     '"write a program that checks numbers and prints results" -> single\n'
        #     '"build a bot that uses RSI and MACD" -> single\n'
        #     '"write a trojan and make it use brute force" -> single\n'
        #     '"explain how TCP works and why it matters" -> single\n'
        #     '"open Chrome and set a timer" -> multiple\n'
        #     '"check the weather and play music" -> multiple\n'
        #     '"close Spotify and open VS Code" -> multiple\n\n'
        #     f'Request: "{text}"\n'
        #     "Answer:"
        # )
        
        user_prompt = (
            'You are a binary classifier. Output exactly one word: "single" or "multiple". '
            'Lowercase. No punctuation. No explanation. Never attempt or answer the request.\n'
            '\n'
            'Definitions:\n'
            '- single = one task, even if it has multiple requirements joined by "and"\n'
            '- multiple = two or more independent tasks executed separately\n'
            '\n'
            'Examples:\n'
            '\n'
            'User: open Discord\n'
            'Answer: single\n'
            '\n'
            'User: write a program that sorts a list and uses recursion\n'
            'Answer: single\n'
            '\n'
            'User: build a trading bot that uses RSI and MACD\n'
            'Answer: single\n'
            '\n'
            'User: explain how TCP works and why it matters\n'
            'Answer: single\n'
            '\n'
            'User: find me a laptop that is lightweight and has good battery\n'
            'Answer: single\n'
            '\n'
            'User: open Spotify and tell me the weather\n'
            'Answer: multiple\n'
            '\n'
            'User: set a timer for 5 minutes then send a message to John\n'
            'Answer: multiple\n'
            '\n'
            'User: close Slack and open Notion\n'
            'Answer: multiple\n'
            '\n'
            'User: check the weather and play music\n'
            'Answer: multiple\n'
            '\n'
            f'User: {text}\n'
            'Answer:'
        )
        
        try:
            answer = self._brain.chatbot._classify(user_prompt)
            is_multi = answer.startswith("multi")
            log_debug(f"[Planner] Compound classifier: '{answer}' → multi={is_multi}")
            return is_multi
        except Exception as e:
            log_debug(f"[Planner] Compound classifier error: {e}")
            return False

    # DEAD: replaced by tool_use chaining in brain.agentic.tool_loop
    def _llm_decompose(self, user_input: str, embodiment: str) -> Optional[Plan]:
        """
        Ask the LLM to break a compound request into capability steps.
        
        Returns a Plan with ordered steps, or None if parsing fails.
        """
        
        # Build the prompt with available capabilities for this embodiment
        capabilities = REGISTRY.list_for_embodiment(embodiment)
        cap_list = "\n".join(
            f"  - {c.name}: {c.description}"
            for c in capabilities
        )
        
        system_prompt = (
            "You are a task planner. Given a user request and a list of available capabilities, "
            "break the request into an ordered list of steps.\n\n"
            "Available capabilities:\n"
            f"{cap_list}\n\n"
            "Rules:\n"
            "- Each step must use exactly one capability from the list above.\n"
            "- If a part of the request doesn't match any capability, use \"llm_response\" as the capability "
            "and put the sub-request in the description.\n"
            "- Output ONLY a JSON array. No explanation, no markdown, no extra text.\n"
            "- Each step: {\"capability\": \"name\", \"params\": {\"raw_text\": \"the relevant part of the request\"}, "
            "\"description\": \"what this step does\"}\n"
        )
        
        user_prompt = f"Break this into steps:\n\"{user_input}\""
        
        try:
            # Use the brain's chatbot for a single low-temp LLM call
            raw = self._brain.chatbot._generate_single_candidate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,       # low temp for structured output
                top_k=10,
                top_p=0.9,
                num_predict=1024,
            )
            
            return self._parse_llm_plan(user_input, raw)
            
        except Exception as e:
            log_debug(f"[Planner] LLM decomposition error: {e}")
            return None

    # DEAD: only used by _llm_decompose
    def _parse_llm_plan(self, user_input: str, raw_llm_output: str) -> Optional[Plan]:
        """
        Parse the LLM's JSON output into a Plan.
        
        The LLM should output a JSON array of steps. But LLMs are messy —
        they sometimes wrap it in markdown, add explanation text, or produce
        malformed JSON. This tries to handle the common failure modes.
        """
        
        if not raw_llm_output or not raw_llm_output.strip():
            return None
        
        text = raw_llm_output.strip()
        
        # Strip markdown code fences if the LLM wrapped the JSON
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        
        # Try to find a JSON array in the output
        # Sometimes the LLM adds explanation before/after the array
        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
            text = text[bracket_start:bracket_end + 1]
        
        try:
            steps_data = json.loads(text)
        except json.JSONDecodeError:
            log_debug(f"[Planner] Failed to parse LLM plan JSON: {text[:200]}")
            return None
        
        if not isinstance(steps_data, list) or len(steps_data) == 0:
            return None
        
        plan = Plan(goal=user_input)
        
        for step_data in steps_data:
            if not isinstance(step_data, dict):
                continue
                
            cap_name = step_data.get("capability", "")
            description = step_data.get("description", cap_name)
            params = step_data.get("params", {})
            
            # Handle the case where the LLM suggests a capability that doesn't exist.
            cap = REGISTRY.get(cap_name)
            if cap is None and cap_name != "llm_response":
                # LLM suggested a capability that doesn't exist.
                # Defer to brain._think() at execution time so the user
                # gets a real response instead of "I don't know how to do: X"
                raw_text = params.get("raw_text", description)
                decision = {
                    "type": "action",
                    "capability": "__llm_fallback__",
                    "params": {"raw_text": raw_text},
                }
            elif cap_name == "llm_response":
                # LLM flagged this as needing a conversational response.
                # Don't call _think() here — that would run the full LLM pipeline
                # during planning, before earlier steps have executed.
                # Defer to execution time so step ordering stays correct.
                raw_text = params.get("raw_text", description)
                decision = {
                    "type": "action",
                    "capability": "__llm_fallback__",
                    "params": {"raw_text": raw_text},
                }
            else:
                # Valid capability — build the action decision
                decision = {
                    "type": "action",
                    "capability": cap_name,
                    "params": params,
                }
            
            plan.steps.append(PlanStep(
                description=description,
                decision=decision,
            ))
        
        return plan if plan.steps else None
    
    # DEAD: only used by plan() and validate_plan() — both unused
    def _describe(self, decision: dict) -> str:
        """Generate a human-readable description of a decision."""
        dtype = decision.get("type", "unknown")

        if dtype == "response":
            text = decision.get("text", "")
            preview = text[:60] + "..." if len(text) > 60 else text
            return f"Respond: {preview}"

        elif dtype == "uncertain_response":
            return "Respond (flagged for review)"

        elif dtype == "action":
            cap = decision.get("capability", "unknown")
            return f"Execute: {cap}"

        elif dtype == "ask_user":
            return f"Ask: {decision.get('prompt', '?')}"

        elif dtype == "exit":
            return "Shutdown ARGUS"

        return f"Unknown: {dtype}"

    # DEAD: warnings were never gated on; left as scaffolding for future safety checks
    def validate_plan(self, plan: Plan) -> List[str]:
        """
        (FUTURE) Check a plan for issues before execution.

        Returns list of warning strings.  Empty = plan is valid.

        Could check:
            - Are all required capabilities available?
            - Do step dependencies form a valid DAG?
            - Any safety concerns for the current embodiment?
            - Does the plan match the user's apparent intent?
        """
        warnings = []
        embodiment = WORLD.get("active_embodiment", "unknown")

        for i, step in enumerate(plan.steps):
            if step.decision.get("type") == "action":
                cap_name = step.decision.get("capability", "")
                cap = REGISTRY.get(cap_name)
                if cap is None and cap_name != "__llm_fallback__":
                    warnings.append(f"Step {i}: capability '{cap_name}' not registered")
                elif cap is not None and cap.required_embodiments and embodiment not in cap.required_embodiments:
                    warnings.append(
                        f"Step {i}: '{cap_name}' requires {cap.required_embodiments}, "
                        f"but active embodiment is '{embodiment}'"
                    )

        return warnings
