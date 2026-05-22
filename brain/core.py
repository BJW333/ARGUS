"""
BrainCore — ARGUS's thinking layer.

Takes text in, produces Decision dicts out.
No platform imports allowed here — the brain doesn't know if it's running on a Mac, a robot, or anything else.
THe embodiment handles all I/O.

The embodiment calls:
    decisions = brain.process("what's the weather in Syracuse")

And gets back a list like:
    [{"type": "action", "capability": "weather.get_forecast",
      "params": {"raw_text": "...", "entities": {...}}}]

Or for an LLM response:
    [{"type": "response", "text": "The weather is...",
      "confidence": 0.82, "reward": 35}]

The embodiment then executes these decisions through its own I/O.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from state.world_state import WORLD
from capabilities.registry import REGISTRY
from brain.reasoning import ReasoningPipeline
from config_metrics.logging import log_debug, log_metrics
from datafunc.data_store import DataStore


class BrainCore:
    """
    Platform-agnostic brain. The single thinking center of ARGUS.

    Every embodiment (desktop, robot, simulation) uses the same BrainCore instance. 
    
    The brain never imports platform code it doesn't know what body it has.

    Flow:
        1. process(text) — entry point. Classifies intent, routes to capability or LLM.
        2. _route()      — maps intent label → capability action or LLM fallback.
                           Checks embodiment filtering (desktop can't move_arm, robot can't open_app).
        3. _think()      — full LLM pipeline when no capability matches.
                           Memory → RAG → candidates → GA rerank → refine → confidence check.

    Decision dict types returned by process():
        {"type": "response",           "text": "...", "confidence": 0.8, "reward": 35}
        {"type": "uncertain_response", "text": "...", "flagged": True, ...}
        {"type": "action",             "capability": "weather.get_forecast", "params": {...}}
        {"type": "ask_user",           "prompt": "Which city?"}
        {"type": "exit"}
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        web_db_path: Optional[str] = None,
        prompt_fn=None,
        output_fn=None,
    ):
        """
        Initialize the brain.

        Args:
            memory_dir:  Path to memory storage (personality.json, etc.)
            web_db_path: Path to web_learning.db for RAG.
            prompt_fn:   Callable(question) → answer.  Provided by embodiment
                         for interactive dialogs (name change, etc.)
            output_fn:   Callable(message) → None.  Provided by embodiment
                         for status messages.
        """
        # ── Memory ──
        from brain.memory.manager import MemoryManager

        if memory_dir is None:
            core_dir = Path(__file__).resolve().parent.parent / "core" / "Argus_memory_storage"
            memory_dir = str(
                Path(os.getenv("ARGUS_MEMORY_DIR", core_dir)).resolve()
            )
        os.makedirs(memory_dir, exist_ok=True)
        self.memory = MemoryManager(
            memory_path=memory_dir,
            prompt_fn=prompt_fn,
            output_fn=output_fn,
        )

        # ── Intent ──
        from brain.nlp.intent import intentrecognition
        self._intent_class = intentrecognition

        # ── LLM + Reward + RAG ──
        from brain.nlp.chatbot_init import initialize_chatbot_components
        chatbot, reward_system, data_store, conversation_history = (
            initialize_chatbot_components()
        )
        self.chatbot = chatbot
        self.reward_system = reward_system
        self.data_store = data_store
        self.conversation_history = conversation_history

        # RAG (optional)
        web_rag = None
        if web_db_path is None:
            web_db_path = os.path.join(memory_dir, "web_learning.db")
        try:
            from brain.rag.web_rag import WebRAG
            web_rag = WebRAG(scraper_db_path=web_db_path)
        except Exception as e:
            log_debug(f"WebRAG init skipped: {e}")

        self.reasoning = ReasoningPipeline(
            chatbot=chatbot,
            reward_system=reward_system,
            web_rag=web_rag,
        )

        log_debug("[BrainCore] Initialized.")

    # ── Main entry point ─────────────────────────────────

    def process(self, user_input: str) -> List[dict]:
        """
        Process user input and return a list of Decision dicts.

        Handles multi-intent splitting: "open Spotify and tell me the time"
        becomes two decisions.

        Each decision is one of:
            {"type": "response", "text": "...", "confidence": ..., "reward": ...}
            {"type": "uncertain_response", "text": "...", "flagged": True, ...}
            {"type": "action", "capability": "weather.get_forecast", "params": {...}}
            {"type": "ask_user", "prompt": "Which city?"}
            {"type": "exit"}
        """
        WORLD.update("user_input", user_input)
        WORLD.update("task_state", "thinking")

        # ── Read world context before routing ──
        _ver, snapshot = WORLD.snapshot()
        embodiment = snapshot.get("active_embodiment", "desktop")
        
        intent_recog = self._intent_class()
        intent_results = intent_recog.unified_intent_pipeline(user_input)

        #decisions = []
        #for intent_label, clause in intent_results:
        #    decision = self._route(clause, intent_label, intent_recog, embodiment)
        #    decisions.append(decision)
        decisions = []
        for intent_label, clause, confidence in intent_results:
            decision = self._route(
                clause, intent_label, intent_recog, embodiment,
                classifier_confidence=confidence,
            )
            decisions.append(decision)    

        WORLD.update("task_state", "idle")
        return decisions

    #Internal routing
    #def _route(self, text: str, intent: str, intent_recog, embodiment: str = "desktop") -> dict:
    def _route(
        self,
        text: str,
        intent: str,
        intent_recog,
        embodiment: str = "desktop",
        classifier_confidence: float = 1.0,
    ) -> dict:
        """
        Route a single intent to a capability or the LLM.

        Priority order:
            1. Exit command
            2. Web learning commands (exact string match, bypass classifier)
            3. Workspace commands (intent == "workspace")
            4. Capability registry lookup by intent label
               → with embodiment filtering (rejects if capability doesn't work on this body)
            5. Edge cases not covered by intent classifier (substring match)
            6. Math fallback (detects digits + math words)
            7. Notes fallback
            8. Default: full LLM pipeline via _think()

        Args:
            text:        The user's command text (may be a clause from multi-intent split).
            intent:      Intent label from the classifier (e.g. "weather_data", "open", "exit").
            intent_recog: The intent recognizer instance (for entity extraction).
            embodiment:  Name of the active embodiment ("desktop", "robot", etc.).

        Returns:
            A single Decision dict.
        """

        entities = intent_recog.extract_entities(text)
        _u = (text or "").strip().lower()

        # ── Exit ──
        if intent == "exit":
            return {"type": "exit"}

        # ── Web learning commands (bypass intent routing) ──
        if _u in (
            "update web knowledge", "update webknowledge",
            "update web memory", "update webmemory",
        ):
            return {"type": "action", "capability": "web.update_knowledge", "params": {}}

        if _u in (
            "start web learning", "start weblearning",
            "start web scraper", "start webscraper",
        ):
            return {"type": "action", "capability": "web.start_learning", "params": {}}

        if _u in (
            "stop web learning", "stop weblearning",
            "stop web scraper", "stop webscraper",
        ):
            return {"type": "action", "capability": "web.stop_learning", "params": {}}

        # ── Workspace commands ──
        if intent == "workspace":
            return {
                "type": "action",
                "capability": "workspace.handle_command",
                "params": {"command": _u},
            }

        # ── Check capability registry ──
        # cap = REGISTRY.find_by_intent(intent)
        # if cap:
        #     # ── Embodiment filtering ──
        #     if cap.required_embodiments and embodiment not in cap.required_embodiments:
        #         return {
        #             "type": "response",
        #             "text": f"I can't do that — {cap.description} "
        #                     f"is only available on {', '.join(cap.required_embodiments)}.",
        #             "confidence": 1.0, "reward": 0,
        #         }
        #     return {
        #         "type": "action",
        #         "capability": cap.name,
        #         "params": {
        #             "raw_text": text,
        #             "entities": entities,
        #             "user_input": text,
        #         },
        #     }
        
        # ── Check capability registry ──
        cap = REGISTRY.find_by_intent(intent)
        if cap:
            # ── Confidence gate ──
            # Brittle rule-based handlers run only when the classifier is
            # genuinely confident (driven by ML-vs-rule agreement upstream).
            # Anything ambiguous falls through to the agentic loop (Claude),
            # which handles natural-language variation reliably.
            FAST_PATH_THRESHOLD = 0.85
            if classifier_confidence < FAST_PATH_THRESHOLD:
                log_debug(
                    f"[ROUTE] Confidence {classifier_confidence:.2f} < "
                    f"{FAST_PATH_THRESHOLD} for intent={intent!r} on {text!r} "
                    f"— falling through to agentic loop"
                )
                # Skip fast path; fall through to _think() at end of method.
            else:
                # ── Embodiment filtering ──
                if cap.required_embodiments and embodiment not in cap.required_embodiments:
                    return {
                        "type": "response",
                        "text": f"I can't do that — {cap.description} "
                                f"is only available on {', '.join(cap.required_embodiments)}.",
                        "confidence": 1.0, "reward": 0,
                    }
                return {
                    "type": "action",
                    "capability": cap.name,
                    "params": {
                        "raw_text": text,
                        "entities": entities,
                        "user_input": text,
                    },
                }    

        #Hardcoded edge cases Phrases the intent classifier doesn't catch reliably
        if "are you there" in _u:
            from config_metrics.main_config import MASTER
            return {"type": "response", "text": f"I'm here {MASTER}", "confidence": 1.0, "reward": 0}

        if "i need to adjust the model" in _u:
            return {
                "type": "action",
                "capability": "model.adjust",
                "params": {"raw_text": text},
            }

        if "tell me a joke" in _u:
            return {"type": "action", "capability": "misc.joke", "params": {}}

        if "movie" in _u:
            return {"type": "action", "capability": "misc.movie", "params": {}}

        if "what are your skills" in _u:
            return {"type": "action", "capability": "misc.list_skills", "params": {}}

        # ── Math fallback (digits + math words) ──
        if re.search(r"\d", text):
            math_re = re.compile(
                r"\b(plus|minus|add|subtract|times|multiply|divide|over|power|mod)\b",
                re.I,
            )
            has_symbol = any(op in text for op in ("+", "-", "*", "/", "**", "%"))
            if has_symbol or math_re.search(text):
                return {
                    "type": "action",
                    "capability": "calculator.calculate",
                    "params": {"expression": text},
                }

        if "notes" in _u:
            return {"type": "action", "capability": "notes.take", "params": {}}

        # ── Default: full LLM pipeline ──
        return self._think(text)

    def _think(self, user_input: str) -> dict:
        """Run the full LLM reasoning pipeline and return a Decision dict."""
        mem = self.memory.get_all_memories(user_input)

        decision = self.reasoning.think(
            user_input=user_input,
            memory=mem,
        )

        # Update world state regardless
        WORLD.update("last_response", decision["text"])
        WORLD.update("last_confidence", decision["confidence"])
        WORLD.update("last_reward", decision["reward"])

        # Log metrics regardless
        log_metrics(
            user_input,
            decision["text"],
            decision.get("response_time", 0),
            decision["reward"],
        )

        # ONLY log to memory/history if NOT flagged for correction.
        # If flagged (uncertain_response), the embodiment will call
        # brain.log_final_response() AFTER the human correction flow,
        # so we log the corrected text — matching old ARGUS behavior.
        if not decision.get("flagged", False):
            self.log_final_response(
                user_input, decision["text"],
                decision["reward"], flagged_for_retraining=False,
            )

        return decision

    def log_final_response(self, user_input: str, response: str,
                           reward: int, flagged_for_retraining: bool = False) -> None:
        """
        Log a finalized response to memory, conversation history, and data store.
        
        Called by:
          - _think() for normal (confident) responses
          - the embodiment for uncertain responses AFTER human correction
        
        This matches old ARGUS behavior: if the user corrects a response,
        only the corrected text is logged — the original uncertain response
        is never saved to training data.
        """
        self.memory.update_memory_from_conversation([{
            "user_input": user_input,
            "bot_response": response,
            "reward": reward,
        }])
        self.conversation_history.append((user_input, response))
        self.data_store.save_data({
            "user_input": user_input,
            "bot_response": response,
            "reward": reward,
            "flagged_for_retraining": flagged_for_retraining,
        })
