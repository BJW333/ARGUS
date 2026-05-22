"""
ARGUS Embodiment — abstract base for any ARGUS body.
=====================================================

An embodiment provides perception (input) and action (output).
The brain calls these methods — it never knows HOW they work.

Desktop embodiment: speech recognition + Mimic3 TTS + QML GUI + macOS app control
Robot embodiment:   microphone + speaker + motors + sensors + camera
Simulated robot:    virtual sensors + console output + logged actions

Every embodiment must implement all abstract methods below.

Usage:
    class DesktopEmbodiment(Embodiment):
        @property
        def name(self) -> str:
            return "desktop"
        ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Embodiment(ABC):
    """Abstract base class for any ARGUS body."""

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique embodiment identifier.
        Used for capability filtering and world state.
        Examples: "desktop", "robot", "simulation"
        """
        ...

    # ── lifecycle ────────────────────────────────────────

    @abstractmethod
    def start(self) -> None:
        """
        Initialize hardware, start sensor threads, register
        platform-specific capabilities, etc.
        Called once at startup.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """
        Clean shutdown of all hardware, threads, and resources.
        Called once at exit.
        """
        ...

    # ── perception (input) ───────────────────────────────

    @abstractmethod
    def get_user_input(self) -> Optional[str]:
        """
        Block until the user provides input.  Returns text.

        Desktop: wake word detection + speech recognition.
        Robot:   speech recognition, gesture trigger, sensor event.
        Sim:     console input or scripted test sequence.

        Returns None if no input could be captured.
        """
        ...

    @abstractmethod
    def ask_user(self, prompt: str) -> Optional[str]:
        """
        Ask the user a follow-up question and wait for their answer.

        Used by: HRLF human correction loop, workspace voice prompts,
        memory name-change dialog, capability parameter collection.

        Desktop: speak the prompt via TTS, then listen for voice input.
        Robot:   speak + listen, or display on screen + wait.
        Sim:     print + console input.

        Returns the user's response text, or None on timeout/failure.
        """
        ...

    # ── action (output) ──────────────────────────────────

    @abstractmethod
    def deliver_response(self, text: str) -> None:
        """
        Present a response to the user.

        Desktop: TTS playback + display in QML chat panel.
        Robot:   TTS + LED/screen + optional gesture.
        Sim:     print to console + log.
        """
        ...

    @abstractmethod
    def show_debug(self, text: str) -> None:
        """
        Display debug / status information.

        Desktop: print_to_gui + console log.
        Robot:   log file + optional status LED.
        Sim:     console print.

        The brain uses this for confidence scores, reward values,
        timing info, etc.  Not shown to the user as conversation.
        """
        ...

    # ── platform actions ─────────────────────────────────

    @abstractmethod
    def execute_platform_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a platform-specific action that only this embodiment can do.

        Desktop examples: open_app, close_app, volume_control
        Robot examples:   move_arm, navigate_to, look_at

        Returns a result dict with at least:
            {"success": True/False, "message": "..."}
        """
        ...

    @abstractmethod
    def get_platform_capabilities(self) -> List[str]:
        """
        Return capability names this embodiment provides.

        Desktop: ["desktop.open_app", "desktop.close_app", "desktop.volume"]
        Robot:   ["robot.move_arm", "robot.navigate", "robot.look_at"]

        These get registered in the CapabilityRegistry at startup.
        """
        ...

    # ── optional hooks ───────────────────────────────────

    def on_brain_thinking(self) -> None:
        """Called when the brain starts processing.  Optional UI feedback."""
        pass

    def on_brain_done(self) -> None:
        """Called when the brain finishes processing.  Optional UI feedback."""
        pass

    def pause_perception(self) -> None:
        """
        Temporarily pause input capture (e.g., pause mic while TTS plays).
        Default: no-op.
        """
        pass

    def resume_perception(self) -> None:
        """Resume input capture after pause.  Default: no-op."""
        pass
