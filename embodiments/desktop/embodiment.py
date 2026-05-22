"""
ARGUS Desktop Embodiment
========================

The first body.  Wraps the existing speech stack (Mimic3 TTS,
Google Speech Recognition), QML GUI, and macOS app control
around the platform-agnostic BrainCore.

Main loop:
    1. Listen for wake word ("Argus")
    2. Capture user command via speech recognition
    3. Pass text to BrainCore.process()
    4. Execute returned Decision dicts (speak, display, run action)

This file OWNS all PySide6 and speech imports.  The brain never
touches any of them.
"""
from __future__ import annotations

import re
import random
from typing import Any, Dict, List, Optional

from embodiments.base import Embodiment
from state.world_state import WORLD
from capabilities.registry import REGISTRY, Capability
from config_metrics.logging import log_debug

class DesktopEmbodiment(Embodiment):
    """
    Desktop body for ARGUS.
    Owns: microphone, TTS, QML GUI, macOS app control.
    """

    def __init__(self, brain):
        """
        Args:
            brain: A brain.core.BrainCore instance.
        """
        self.brain = brain
        self.running = False

        # Lazy-loaded platform components
        self._speak_fn = None
        self._listen_fn = None
        self._gui_print_fn = None
        self._speech_manager = None

    @property
    def name(self) -> str:
        return "desktop"

    # ── Lifecycle ────────────────────────────────────────
    def start(self) -> None:
        """Initialize speech, GUI bridge, register desktop capabilities."""
        # Import platform-specific modules HERE, not at module level
        from speech.speak import speak
        from speech.listen import generalvoiceinput, request_voice_answer
        from speech.speechmanager import speech_manager
        from core.input_bus import print_to_gui

        self._speak_fn = speak
        self._listen_fn = generalvoiceinput
        self._gui_print_fn = print_to_gui
        self._speech_manager = speech_manager
        
        # Register I/O hooks so the agentic ask_user tool can reach the
        # voice stack without the brain importing platform code.
        from brain.agentic import io_hooks
        io_hooks.register_speak(self._speak_fn)
        io_hooks.register_voice_answer(request_voice_answer)
        
        # Register desktop-only capabilities
        self._register_platform_capabilities()

        # Register common capabilities (weather, stocks, etc.)
        self._register_common_capabilities()
        
        # Load hot-discovered skills (after common capabilities so skills
        # that wrap them can find their dependencies in REGISTRY)
        from brain.skills import load_skills
        load_skills()
        
        # Wire brain's memory system to this embodiment's I/O
        # so interactive dialogs (name change, etc.) use voice on desktop.
        # _prompt: speaks the question via TTS then listens for voice answer
        # _output: speaks AND displays confirmations (old ARGUS did both
        #          print_to_gui + speak for messages like "I will now call you X")
        self.brain.memory._prompt = self.ask_user
        self.brain.memory._output = lambda msg: self.deliver_response(msg)

        # Register workspace capability (needs speak/listen from this embodiment)
        from capabilities.workspace_cap import register_workspace_capability
        from actions.actions import gatherinfofromknowledgebase
        register_workspace_capability(
            speak_fn=self._speak_fn,
            listen_fn=self._listen_fn,
            research_fn=gatherinfofromknowledgebase,
        )

        WORLD.update("active_embodiment", "desktop")
        WORLD.update("system_status", "ready")
        self.running = True
        log_debug("[DesktopEmbodiment] Started.")

    # def stop(self) -> None:
    #     """Clean shutdown."""
    #     self.running = False
    #     WORLD.update("system_status", "shutting_down")
    #     if self._speech_manager:
    #         try:
    #             self._speech_manager.shutdown_speech()
    #         except Exception as e:
    #             log_debug(f"Speech shutdown error: {e}")
    #     log_debug("[DesktopEmbodiment] Stopped.")

    def stop(self) -> None:
        """Clean shutdown."""
        self.running = False
        WORLD.update("system_status", "shutting_down")
        if self._speech_manager:
            try:
                self._speech_manager.shutdown_speech()
            except Exception as e:
                log_debug(f"Speech shutdown error: {e}")
        # Shut down RealtimeSTT recorder BEFORE Qt teardown
        # (the listen thread is daemon, its finally: never fires).
        try:
            from speech import listen as _listen_mod
            if getattr(_listen_mod, "_recorder", None) is not None:
                _listen_mod._recorder.shutdown()
                _listen_mod._recorder = None
        except Exception as e:
            log_debug(f"Recorder shutdown error: {e}")
        # Stop AEC engine too
        try:
            from speech.voice_engine import voice_engine
            voice_engine.stop()
        except Exception as e:
            log_debug(f"Voice engine stop error: {e}")
        log_debug("[DesktopEmbodiment] Stopped.")
        
    # ── Perception ───────────────────────────────────────

    def get_user_input(self) -> Optional[str]:
        """Block until wake word + command via speech recognition."""
        if self._listen_fn:
            return self._listen_fn()
        return None

    def ask_user(self, prompt: str) -> Optional[str]:
        """Speak a question and wait for voice answer."""
        if self._speak_fn:
            self._speak_fn(prompt)
        if self._listen_fn:
            return self._listen_fn()
        return None

    # ── Output ───────────────────────────────────────────

    # def deliver_response(self, text: str) -> None:
    #     """Speak + display in GUI."""
    #     if self._gui_print_fn:
    #         self._gui_print_fn(f"ARGUS: {text}")
    #     if self._speak_fn:
    #         self._speak_fn(f"Bot: {text}")
            
    def deliver_response(self, text: str, *, gui: bool = True) -> None:
        """Display in GUI; speak only if last input came via voice.

        Voice in → voice out. Typed input stays silent. The input source is
        set by core.input_bus.send() and read from WorldState here so every
        TTS-emitting code path (responses, uncertain responses, capability
        results) respects the same gate.

        Args:
            gui: If False, skip the GUI print. Use this when text has
                 already been streamed to the GUI via stream_chunk_to_gui.
        """
        if gui and self._gui_print_fn:
            self._gui_print_fn(f"ARGUS: {text}")
        source = WORLD.get("input_source", "voice")
        if source == "voice" and self._speak_fn:
            self._speak_fn(f"Bot: {text}")
            
    def show_debug(self, text: str) -> None:
        """Display debug info in GUI."""
        if self._gui_print_fn:
            self._gui_print_fn(text)

    # ── Platform Actions ─────────────────────────────────

    def execute_platform_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute desktop-specific action (open app, volume, etc.)."""
        cap = REGISTRY.get(action_name)
        if cap and cap.handler:
            try:
                result = cap.handler(**params)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": f"Unknown action: {action_name}"}

    def get_platform_capabilities(self) -> List[str]:
        return [
            "desktop.open_app", "desktop.close_app", "desktop.volume",
        ]

    # ── Decision execution ───────────────────────────────

    def run_decisions(self, decisions: List[dict]) -> None:
        """
        Execute a list of Decision dicts from the brain.
        This is the core loop that connects brain output to real-world I/O.
        """
        for decision in decisions:
            self._run_one(decision)

    def _run_one(self, decision: dict) -> None:
        """Execute a single Decision dict."""
        dtype = decision.get("type", "")

        # if dtype == "response":
        #     if not decision.get("streamed", False):
        #         self.deliver_response(decision["text"])
        #     self.show_debug(f"Confidence: {decision.get('confidence', '?')}")
        #     self.show_debug(f"Reward: {decision.get('reward', '?')}")
        
        if dtype == "response":
            # If streaming, text already in GUI; only deliver_response's
            # TTS branch runs (and only if input was voice).
            # If non-streaming, deliver_response handles both GUI print + TTS.
            already_in_gui = decision.get("streamed", False)
            self.deliver_response(decision["text"], gui=not already_in_gui)
            self.show_debug(f"Confidence: {decision.get('confidence', '?')}")
            self.show_debug(f"Reward: {decision.get('reward', '?')}")
            
        elif dtype == "uncertain_response":
            # Speak the uncertain response first (old ARGUS did this)
            self.deliver_response(decision["text"])
            self.deliver_response("This response is flagged for human review. Correct the response.")
            correction = self.ask_user("Please provide the correct response:")
            
            user_input = WORLD.get("user_input", "")
            
            if correction:
                # User corrected — log the CORRECTED text (old behavior:
                # response = corrected_response, then log that)
                self.brain.log_final_response(
                    user_input, correction,
                    reward=0, flagged_for_retraining=True,
                )
            else:
                # No correction provided — log the original uncertain response
                # (old behavior: flagged_for_retraining = False, log original)
                self.brain.log_final_response(
                    user_input, decision["text"],
                    reward=decision.get("reward", 0),
                    flagged_for_retraining=False,
                )

        elif dtype == "action":
            self._run_action(decision)

        elif dtype == "ask_user":
            answer = self.ask_user(decision.get("prompt", ""))
            if answer:
                followup = self.brain.process(answer)
                self.run_decisions(followup)

        elif dtype == "exit":
            self.stop()
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.quit()
            except Exception:
                pass

    def _run_action(self, decision: dict) -> None:
        """Execute a capability action."""
        cap_name = decision.get("capability", "")
        params = decision.get("params", {})

        # Planner deferred this step to the brain's LLM pipeline.
        # Happens when the LLM decomposition produced a step that doesn't
        # map to a registered capability, or flagged it as needing a
        # conversational response. Run _think() now at execution time
        # so step ordering stays correct.
        if cap_name == "__llm_fallback__":
            raw_text = params.get("raw_text", "")
            if raw_text and self.brain:
                decisions = self.brain.process(raw_text)
                self.run_decisions(decisions)
            return

        cap = REGISTRY.get(cap_name)
        if cap is None:
            self.deliver_response(f"I don't know how to do: {cap_name}")
            return

        try:
            result = cap.handler(**params)

            # Handle special return values
            if isinstance(result, str):
                if result.startswith("__ASK__:"):
                    # Capability needs more info from user
                    prompt = result[8:]
                    answer = self.ask_user(prompt)
                    if answer:
                        params["followup_answer"] = answer
                        result = cap.handler(**params)
                        if isinstance(result, str) and not result.startswith("__"):
                            self.deliver_response(result)
                    else:
                        self.deliver_response("I didn't catch that. Please try again.")
                else:
                    self.deliver_response(result)
            elif isinstance(result, dict):
                if "message" in result:
                    self.deliver_response(result["message"])
                if "gui_message" in result:
                    self.show_debug(result["gui_message"])
        except Exception as e:
            log_debug(f"Capability {cap_name} error: {e}")
            self.deliver_response(f"Sorry, I hit an error with {cap_name}: {e}")

    # ── Capability Registration ──────────────────────────

    def _register_platform_capabilities(self) -> None:
        """Register desktop-only capabilities."""
        from actions.actions import open_app, close_application, volumecontrol

        REGISTRY.register(Capability(
            name="desktop.open_app",
            description="Open an application on macOS",
            handler=lambda raw_text="", followup_answer=None, **kw: self._handle_open_app(raw_text, followup_answer),
            keywords=["open"],
            required_embodiments=["desktop"],
        ))
        REGISTRY.register(Capability(
            name="desktop.close_app",
            description="Close an application on macOS",
            handler=lambda raw_text="", followup_answer=None, **kw: self._handle_close_app(raw_text, followup_answer),
            keywords=["close"],
            required_embodiments=["desktop"],
        ))
        REGISTRY.register(Capability(
            name="desktop.volume",
            description="Control system volume",
            handler=lambda raw_text="", **kw: volumecontrol(raw_text),
            keywords=["volume_control"],
            required_embodiments=["desktop"],
        ))

    def _register_common_capabilities(self) -> None:
        """Register platform-agnostic capabilities that come from actions.py."""
        from actions.actions import (
            calculate, get_ticker, get_city_coordinates,
            identifynetworkconnect, get_the_news, action_time,
            coin_flip, cocktail, start_timer, gatherinfofromknowledgebase,
            parse_weather_query, format_day_name,
        )
        from datafunc import data_analysis
        import pyjokes

        REGISTRY.register(Capability(
            name="weather.get_forecast",
            description="Get weather for a city",
            handler=lambda raw_text="", entities=None, followup_answer=None, **kw: self._handle_weather(
                raw_text, entities or {}, followup_answer
            ),
            keywords=["weather_data"],
        ))
        REGISTRY.register(Capability(
            name="weather.get_for_city",
            description="Direct city-based weather lookup (no NL parsing). Used by agentic tools.",
            handler=lambda city="", day_offset=0, **kw: self._get_weather_for_city(
                city=city, day_offset=day_offset
            ),
            keywords=[],
        ))
        REGISTRY.register(Capability(
            name="stocks.get_price",
            description="Get stock price for a company",
            handler=lambda raw_text="", entities=None, followup_answer=None, **kw: self._handle_stocks(
                raw_text, entities or {}, followup_answer
            ),
            keywords=["stock_data"],
        ))
        REGISTRY.register(Capability(
            name="stocks.get_for_company",
            description="Direct company-name → stock-price lookup. Used by agentic tools.",
            handler=lambda company="", **kw: self._get_stock_for_company(company),
            keywords=[],
        ))
        REGISTRY.register(Capability(
            name="time.get_current",
            description="Get current time",
            handler=lambda **kw: action_time(),
            keywords=["time"],
        ))
        REGISTRY.register(Capability(
            name="news.get_headlines",
            description="Get recent news headlines",
            handler=lambda **kw: self._handle_news(),
            keywords=["news"],
        ))
        REGISTRY.register(Capability(
            name="network.check",
            description="Check internet connectivity",
            handler=lambda **kw: "Internet is connected" if identifynetworkconnect() else "Internet is not connected",
            keywords=["connectionwithinternet"],
        ))
        REGISTRY.register(Capability(
            name="timer.start",
            description="Start a countdown timer",
            handler=lambda raw_text="", **kw: start_timer(raw_text),
            keywords=["timer"],
        ))
        REGISTRY.register(Capability(
            name="misc.coinflip",
            description="Flip a coin",
            handler=lambda **kw: coin_flip(),
            keywords=["coinflip"],
        ))
        REGISTRY.register(Capability(
            name="calculator.calculate",
            description="Do arithmetic",
            handler=lambda expression="", **kw: calculate(expression),
        ))
        REGISTRY.register(Capability(
            name="misc.joke",
            description="Tell a joke",
            handler=lambda **kw: pyjokes.get_joke(),
        ))
        REGISTRY.register(Capability(
            name="misc.movie",
            description="Suggest a movie",
            handler=lambda **kw: self._handle_movie(),
        ))
        REGISTRY.register(Capability(
            name="misc.list_skills",
            description="List what ARGUS can do",
            handler=lambda **kw: self._handle_list_skills(),
        ))
        REGISTRY.register(Capability(
            name="model.adjust",
            description="Manual model adjustment (save json to text, feedback)",
            handler=lambda raw_text="", followup_answer=None, **kw: (
                self._run_model_adjust_followup(followup_answer)
                if followup_answer
                else "__ASK__:Please say the command: 1. save json to text, 2. feedback"
            ),
        ))
        REGISTRY.register(Capability(
            name="search.wikipedia",
            description="Search Wikipedia for information",
            handler=lambda raw_text="", entities=None, **kw: self._handle_search(
                raw_text, entities or {}
            ),
            keywords=["searchsomething"],
        ))
        REGISTRY.register(Capability(
            name="crypto.get_price",
            description="Get cryptocurrency price",
            handler=lambda raw_text="", followup_answer=None, **kw: self._handle_crypto(followup_answer),
            keywords=["crypto_data"],
        ))
        REGISTRY.register(Capability(
            name="crypto.get_for_coin",
            description="Direct coin → price lookup. Used by agentic tools.",
            handler=lambda coin="", **kw: self._get_crypto_for_coin(coin),
            keywords=[],
        ))
        REGISTRY.register(Capability(
            name="flights.get_status",
            description="Get flight status",
            handler=lambda raw_text="", followup_answer=None, **kw: self._handle_flight(followup_answer),
            keywords=["flight_data"],
        ))
        REGISTRY.register(Capability(
            name="flights.get_for_number",
            description="Direct flight number → status. Used by agentic tools.",
            handler=lambda flight_number="", **kw: self._get_flight_for_number(flight_number),
            keywords=[],
        ))
        REGISTRY.register(Capability(
            name="cocktail.get_recipe",
            description="Get a cocktail recipe",
            handler=lambda raw_text="", followup_answer=None, **kw: self._handle_cocktail(followup_answer),
            keywords=["cocktail_intent"],
        ))
        REGISTRY.register(Capability(
            name="notes.take",
            description="Take voice notes",
            handler=lambda **kw: self._handle_notes(),
        ))

        WORLD.update("capabilities_available", REGISTRY.list_names())

    #Action handlers (bridge old actions.py → capability interface)
    def _handle_open_app(self, raw_text: str, followup_answer=None) -> Optional[str]:
        from actions.actions import open_app
        app_name = followup_answer  # user answered "which app?"
        if not app_name:
            m = re.search(r"\bopen\b\s+(.*)", raw_text, re.I)
            app_name = m.group(1).strip().rstrip(".,!?;:\"'` ") if m else ""
        if not app_name:
            return "__ASK__:Which app should I open?"
        open_app(app_name)
        return None
    
    def _handle_close_app(self, raw_text: str, followup_answer=None) -> Optional[str]:
        from actions.actions import close_application
        app_name = followup_answer
        if not app_name:
            m = re.search(r"\bclose\b\s+(.*)", raw_text, re.I)
            app_name = m.group(1).strip().rstrip(".,!?;:\"'` ") if m else ""
        if not app_name:
            return "__ASK__:Which app should I close?"
        close_application(app_name)
        return None
    
    def _get_weather_for_city(
        self,
        city: str,
        day_offset: int = 0,
    ) -> str:
        """LAYER 1 (pure structured-args API). No NL parsing.
        Both the agentic tool path and the legacy intent path call this.
        """
        from actions.actions import get_city_coordinates, format_day_name
        from datafunc import data_analysis

        if not city or not city.strip():
            log_debug("[WEATHER] called with empty city")
            return "[error] city argument is empty"

        log_debug(f"[WEATHER] Lookup: city={city!r} day_offset={day_offset}")

        coords = get_city_coordinates(city)
        if not coords or coords == (None, None):
            log_debug(f"[WEATHER] Geocoder returned no result for {city!r}")
            return (
                f"[error] geocoder couldn't find '{city}'. "
                f"Try a more specific city + state/country."
            )

        lat, lon = coords
        log_debug(f"[WEATHER] Coords: {lat}, {lon}")

        stream = data_analysis.WeatherStream(city=city, lat=lat, lon=lon)
        result = stream.fetch(days=day_offset + 1)
        if not result.success:
            log_debug(f"[WEATHER] Fetch failed: {result.error}")
            return f"[error] weather fetch failed for '{city}': {result.error}"

        metrics = stream.analyze(result, day_offset=day_offset)
        if "error" in metrics:
            log_debug(f"[WEATHER] Analyze error: {metrics['error']}")
            return f"[error] weather analyze failed: {metrics['error']}"

        if metrics["is_forecast"]:
            day_name = format_day_name(day_offset)
            return (
                f"{day_name.capitalize()} in {metrics['city']}: "
                f"high of {metrics['high_f']} degrees, "
                f"low of {metrics['low_f']} degrees, "
                f"{metrics['condition']}."
            )
        else:
            temp = round(metrics["temp_f"]) if metrics["temp_f"] else "unknown"
            msg = f"Currently in {metrics['city']}: {temp} degrees and {metrics['condition']}."
            high = round(metrics["high_f"]) if metrics["high_f"] else None
            low = round(metrics["low_f"]) if metrics["low_f"] else None
            if high and low:
                msg += f" Today's high {high} degrees, low {low} degrees."
            return msg
    
    def _handle_weather(self, raw_text: str, entities: dict, followup_answer=None) -> str:
        """LAYER 2 (NL wrapper for legacy intent path).
        Parses text → delegates to _get_weather_for_city.
        Old intent system continues to work unchanged.
        """
        from actions.actions import parse_weather_query

        location, day_offset = parse_weather_query(raw_text, entities)
        if not location:
            location = followup_answer  # user answered "which city?"
        if not location:
            return "__ASK__:What city would you like the weather for?"

        return self._get_weather_for_city(city=location, day_offset=day_offset)

    def _get_stock_for_company(self, company_name: str) -> str:
        """LAYER 1 (pure). Direct company-name → stock-price lookup."""
        from actions.actions import get_ticker
        from datafunc import data_analysis

        if not company_name or not company_name.strip():
            log_debug("[STOCK] called with empty company name")
            return "[error] company name is empty"

        log_debug(f"[STOCK] Lookup: company={company_name!r}")

        ticker = get_ticker(company_name)
        if not ticker:
            log_debug(f"[STOCK] Ticker resolution failed for {company_name!r}")
            return f"[error] couldn't find ticker for '{company_name}'"

        log_debug(f"[STOCK] Ticker resolved: {ticker}")

        stream = data_analysis.StockStream(symbol=ticker, api_key="MJX8BVSA9W1WOEH4")
        result = stream.fetch()
        if not result.success:
            log_debug(f"[STOCK] Fetch failed: {result.error}")
            return f"[error] stock fetch failed for {ticker.upper()}: {result.error}"

        metrics = stream.analyze(result)
        msg = f"The latest price of {metrics['symbol']} is ${metrics['price']:.2f}"
        if metrics["significant"]:
            direction = "up" if metrics["change"] > 0 else "down"
            msg += f", {direction} {abs(metrics['percent_change']):.1f}%"
        return msg
    
    def _handle_stocks(self, raw_text: str, entities: dict, followup_answer=None) -> str:
        """LAYER 2 (NL wrapper). Parses entities → delegates."""
        company_name = followup_answer
        if not company_name and "ORG" in entities.values():
            for ent, label in entities.items():
                if label == "ORG":
                    company_name = ent
                    break
        if not company_name:
            return "__ASK__:Please tell me the company name or stock listing."
        return self._get_stock_for_company(company_name)
    
    def _handle_news(self) -> str:
        from actions.actions import get_the_news
        news = get_the_news()
        lines = ["Here's what's happening in the news:"]
        for category, headlines in news.items():
            lines.append(f"\n{category} News:")
            for i, headline in enumerate(headlines, 1):
                lines.append(f"{i}. {headline}")
        return "\n".join(lines)

    def _handle_search(self, raw_text: str, entities: dict) -> str:
        """
        Search handler matching old orchestrator searchsomething behavior:
        1. Try WebRAG first → if results, synthesize with LLM + GA
        2. Else try Wikipedia (PERSON entity or keyword extraction)
        3. Else fallback to Google
        """
        from actions.actions import gatherinfofromknowledgebase
        from keybert import KeyBERT
        import json
        from brain.reasoning import generate_candidates, build_memory_prefix, strip_thinking, cleanup_final

        # ── Step 1: Try WebRAG synthesis first (old lines 417-459) ──
        if self.brain.reasoning.web_rag:
            try:
                try:
                    web_memory = self.brain.reasoning.web_rag.retrieve(raw_text, top_k=4)
                except TypeError:
                    web_memory = self.brain.reasoning.web_rag.retrieve(raw_text)
            except Exception as e:
                log_debug(f"WebRAG retrieve error (searchsomething): {e}")
                web_memory = ""

            if web_memory:
                mem = self.brain.memory.get_all_memories(raw_text)
                personality = json.dumps(mem.get("personality", {}))
                short_term = json.dumps(mem.get("short_term", []))
                long_term = json.dumps(mem.get("long_term", []))
                memory_prefix = build_memory_prefix(personality, short_term, long_term)
                web_memory = web_memory[:8000]
                synth_prefix = memory_prefix + "\n\n[WEB_MEMORY]\n" + web_memory + "\n\n"

                cands = generate_candidates(
                    self.brain.chatbot,
                    user_input=raw_text,
                    memory_prefix=synth_prefix,
                    num_candidates=3,
                    temperature=0.65,
                )
                
                cands = [strip_thinking(c) for c in cands]
                cands = [c for c in cands if c.strip()]
                
                best, _, _ = self.brain.chatbot.ga_rerank_candidates(
                    user_input=raw_text,
                    candidates=cands,
                    pop_size=5,
                    generations=1,
                    crossover_rate=0.5,
                    mutation_rate=0.2,
                )
                
                best = cleanup_final(best)
                return best

        # ── Step 2: Wikipedia lookup ──
        if "PERSON" in entities.values():
            person_name = next(
                (ent for ent, label in entities.items() if label == "PERSON"),
                raw_text,
            )
            wiki = gatherinfofromknowledgebase(person_name)
            if wiki and "No results found." not in wiki:
                return wiki
            return f"I couldn't find info on {person_name}. I'll look it up on Google for you."
        else:
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(
                raw_text, keyphrase_ngram_range=(1, 2),
                stop_words="english", top_n=3,
            )
            term = keywords[0][0] if keywords else raw_text
            wiki = gatherinfofromknowledgebase(term)
            if wiki and "No results found." not in wiki:
                return wiki
            return f"I couldn't find information about {term}. I'll look it up on Google for you."

    def _handle_notes(self) -> str:
        from actions.actions import takenotes
        from config_metrics.main_config import MASTER
        note_text = self.ask_user(f"Just state what you want to document in your notes, {MASTER}.")
        if not note_text:
            return "No note captured."
        result = takenotes(note_text)
        return result or "Note saved"

    def _get_crypto_for_coin(self, coin: str) -> str:
        """LAYER 1 (pure). Direct coin-name → price lookup."""
        from datafunc import data_analysis

        if not coin or not coin.strip():
            log_debug("[CRYPTO] called with empty coin")
            return "[error] coin argument is empty"

        coin_id = coin.lower().strip()
        log_debug(f"[CRYPTO] Lookup: coin={coin_id!r}")

        stream = data_analysis.CryptoStream(coin_id=coin_id, vs_currency="usd")
        result = stream.fetch()
        if not result.success:
            log_debug(f"[CRYPTO] Fetch failed: {result.error}")
            return f"[error] crypto fetch failed for '{coin_id}': {result.error}"

        metrics = stream.analyze(result)
        msg = f"The current price of {metrics['name']} is ${metrics['price']:.2f}"
        if metrics["significant"]:
            direction = "up" if metrics["change_24h"] > 0 else "down"
            msg += f", {direction} {abs(metrics['change_24h']):.1f}% in the last 24 hours"
        return msg
    
    def _handle_crypto(self, followup_answer=None) -> str:
        """LAYER 2 (NL wrapper). Delegates to _get_crypto_for_coin."""
        if not followup_answer:
            return "__ASK__:Please tell me the cryptocurrency you're interested in."
        return self._get_crypto_for_coin(followup_answer)

    def _get_flight_for_number(self, flight_number: str) -> str:
        """LAYER 1 (pure). Direct flight-number → status lookup."""
        from datafunc import data_analysis

        if not flight_number or not flight_number.strip():
            log_debug("[FLIGHT] called with empty flight number")
            return "[error] flight number is empty"

        log_debug(f"[FLIGHT] Lookup: flight={flight_number!r}")

        stream = data_analysis.FlightStream(
            flight_number=flight_number,
            api_key="6087a4f837f7de0c30af184c6e886a9b",
        )
        result = stream.fetch()
        if not result.success:
            log_debug(f"[FLIGHT] Fetch failed: {result.error}")
            return f"[error] flight fetch failed for {flight_number.upper()}: {result.error}"

        metrics = stream.analyze(result)
        msg = f"Flight {metrics['flight']} from {metrics['from']} to {metrics['to']} is {metrics['status']}."
        if metrics["delayed"]:
            msg += " The flight is delayed."
        return msg
    
    def _handle_flight(self, followup_answer=None) -> str:
        """LAYER 2 (NL wrapper). Delegates to _get_flight_for_number."""
        if not followup_answer:
            return "__ASK__:Please tell me the flight number."
        return self._get_flight_for_number(followup_answer)

    def _handle_cocktail(self, followup_answer=None) -> str:
        if not followup_answer:
            return "__ASK__:Provide me with the name of the specific drink you want to make."
        from actions.actions import cocktail
        return cocktail(followup_answer)  # now returns string instead of speaking

    def _handle_movie(self) -> str:
        """Exact movie list from old orchestrator.py line 743."""
        from config_metrics.main_config import MASTER
        goodmovies = [
            "Star Wars", "Jurassic Park", "Clear and Present Danger",
            "War Dogs", "Wolf of Wall Street", "The Big Short",
            "Trading Places", "The Gentlemen", "Ferris Bueller's Day Off",
            "Goodfellas", "Lord of War", "Borat", "Marvel movies",
            "The Hurt Locker", "Hustle", "Forrest Gump", "Darkest Hour",
            "Coming to America", "Warren Miller movies", "The Dictator",
        ]
        return f"A good movie you could watch is {random.choice(goodmovies)}, {MASTER}"

    def _handle_list_skills(self) -> str:
        """Exact skills text from old orchestrator.py lines 748-766."""
        return (
            "-Hi, I am Argus. I can perform various tasks, including:\n"
            "- **General Chat**: I can have conversations and improve over time.\n"
            "- **Search the Web**: I can look up information on Google and Wikipedia.\n"
            "- **Open & Close Apps**: I can open and close apps on your computer.\n"
            "- **Check Time & Date**: I can tell you the current time and date.\n"
            "- **Stock & Crypto Data**: I can check stock prices and cryptocurrency trends.\n"
            "- **Weather Updates**: I can give you weather reports.\n"
            "- **Flight Tracking**: I can check flight statuses.\n"
            "- **News Updates**: I can tell you the latest headlines.\n"
            "- **Math**: I can do simple calculations.\n"
            "- **Custom Tools**: I can run custom tools like password checker, spider crawler, find peoples info, hide me and more"
            "- **Take Notes**: I can write and save notes for you.\n"
            "- **Find Cocktail Recipes**: I can look up drink recipes.\n"
            "- **Jokes & Fun**: I can tell jokes and suggest movies.\n"
            "- **Code Generation**: I can write and improve code based on your request.\n"
            "- **Improve Over Time**: I learn from interactions to get better.\n"
            "\nI'm always improving and adding new features!"
        )

    def _handle_model_adjust(self) -> str:
        """
        Manual model adjustment from old orchestrator.py lines 715-737.
        Returns __ASK__ to trigger the voice prompt flow.
        """
        return "__ASK__:Please say the command: 1. save json to text, 2. feedback"

    def _run_model_adjust_followup(self, answer: str) -> str:
        """Handle the model adjust sub-commands after user responds."""
        from datafunc.data_store import json_to_text
        from config_metrics.main_config import script_dir
        from core.feedback import collect_human_feedback

        if answer is None:
            return "There was an error adjusting the model, try again."
        lowered = answer.lower()
        if lowered == "save json to text":
            conversation_data = self.brain.data_store.load_data()
            text_data = json_to_text(conversation_data)
            file_path = script_dir / "data/conversation_datajsontotxt.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(text_data)
            return "JSON conversation data saved to text file."
        elif lowered == "feedback":
            collect_human_feedback(self.brain.conversation_history)
            return "Feedback collected."
        else:
            return "There was an error adjusting the model, try again."
