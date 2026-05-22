"""
ARGUS Desktop Entry Point (v2 architecture)
============================================

Uses the modular brain architecture:
    BrainCore (platform-agnostic) → Decision dicts
    DesktopEmbodiment (platform-specific) → speech + GUI

Uses the existing speech engine (speech/listen.py) for wake word
detection. No duplication — the same listen_for_wake_word() from
old ARGUS, just wired to the v2 brain pipeline.

Usage:
    python main_desktop.py
"""
import sys
import os
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

# Load .env file before anything else (so chatbot picks up API keys)
from dotenv import load_dotenv
load_dotenv()

from brain.core import BrainCore
from embodiments.desktop.embodiment import DesktopEmbodiment
from state.world_state import WORLD
from core.startup import wishme, print_banner
from gui_qml.backend_bridge import BackendBridge
from core.input_bus import print_to_gui, set_v2_handler
from config_metrics.logging import log_debug


def main():
    print_banner()
    wishme()

    # ── Initialize brain (platform-agnostic) ──
    brain = BrainCore()

    # ── Initialize web learning (optional background scraper) ──
    _mem_dir = Path(os.getenv("ARGUS_MEMORY_DIR",
                  Path(__file__).parent / "core" / "Argus_memory_storage")).resolve()
    _web_db = str(_mem_dir / "web_learning.db")
    _weblearn_on = os.getenv("ARGUS_WEBLEARN_ENABLED", "0").lower() in ("1", "true", "yes")

    from brain.web_learning.scraper_service import ScraperService
    scraper_svc = ScraperService(
        db_path=_web_db,
        cycle_seconds=int(os.getenv("ARGUS_WEBLEARN_CYCLE_SEC", "3600")),
        enabled=_weblearn_on,
    )
    if _weblearn_on:
        scraper_svc.start()

    from capabilities.web_learning_cap import init_web_learning
    init_web_learning(scraper_svc, brain.reasoning.web_rag)

    # ── Initialize planner ──
    from brain.planner import Planner
    planner = Planner(brain_core=brain)

    # ── Initialize desktop embodiment ──
    desktop = DesktopEmbodiment(brain)
    desktop.start()

    #old v2_handler with planner decomposition + validation steps (now bypassed in favor of agentic loop's native compound handling)
    # def v2_handler(command_text: str):
    #     """Process command through planner → brain → embodiment."""
    #     if not command_text:
    #         return
        
    #     def _process():
    #         # Planner decides: single-step (pass to brain.process) or 
    #         # multi-step (LLM decomposition into capability sequence)
    #         _ver, snapshot = WORLD.snapshot()
    #         plan = planner.decompose(command_text, snapshot)
            
    #         warnings = planner.validate_plan(plan)
    #         for w in warnings:
    #             desktop.show_debug(f"[Planner warning] {w}")
            
    #         desktop.run_decisions(plan.as_decisions())
        
    #     threading.Thread(target=_process, daemon=True, name="V2Handler").start()
    
    def v2_handler(command_text: str):
        """Route command: compound → agentic loop, single → normal brain routing."""
        if not command_text:
            return

        def _process():
            # Compound detection only. If compound, skip brain.process()'s
            # intent classifier fast path — the agentic loop (Claude with
            # tool_use) handles multi-step natively via tool chaining.
            # NOTE: planner._llm_decompose is intentionally bypassed here;
            # it remains in the codebase as a reference/fallback for
            # non-agentic backends but is no longer in the live path.
            if planner._looks_compound(command_text):
                log_debug("[v2] Compound detected — forcing agentic loop")
                decisions = [brain._think(command_text)]
            else:
                decisions = brain.process(command_text)

            desktop.run_decisions(decisions)

        threading.Thread(target=_process, daemon=True, name="V2Handler").start()
        
    set_v2_handler(v2_handler)

    # ── Set up Qt GUI ──
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QGuiApplication(sys.argv)
    app.setApplicationName("ARGUS")

    engine = QQmlApplicationEngine()
    backend = BackendBridge()
    backend.setObjectName("BackendBridge")
    app.backend = backend
    engine.rootContext().setContextProperty("Backend", backend)

    qml_file = Path(__file__).parent / "gui_qml" / "MainView.qml"
    engine.load(str(qml_file))

    if not engine.rootObjects():
        sys.exit(-1)

    # ── Start the existing wake word listener ──
    # This is the SAME listener from speech/listen.py that old ARGUS used.
    # It handles: wake word detection, ambient noise adjustment, backoff,
    # audio level feeding to BrainCanvas, pause/resume for workspace.
    # It calls input_bus.send() → our v2_handler above.
    from speech.listen import listen_for_wake_word
    wake_thread = threading.Thread(
        target=listen_for_wake_word,
        daemon=True,
        name="ListenWakeWordThread",
    )
    wake_thread.start()

    WORLD.update("system_status", "running")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
