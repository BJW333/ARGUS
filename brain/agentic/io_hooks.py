"""
I/O hooks for tools that need to talk to the user.

The embodiment registers its speak/listen hooks here at startup.
Tools that need user interaction (ask_user) look them up at execution time.

Why this pattern:
    Tools are defined in brain.agentic, which can't import embodiment code
    (the brain stays platform-agnostic). The embodiment populates this
    registry at startup so tools can use voice/text without the brain
    needing to import it.
"""
from typing import Callable, Optional

_speak_fn:           Optional[Callable[[str], None]]            = None
_voice_answer_fn:    Optional[Callable[[float], Optional[str]]] = None


def register_speak(fn: Callable[[str], None]) -> None:
    """Embodiment calls this at startup with its TTS function."""
    global _speak_fn
    _speak_fn = fn


def register_voice_answer(fn: Callable[[float], Optional[str]]) -> None:
    """Embodiment registers a function that returns the user's next voice
    utterance (or None on timeout/interrupt). The function takes a single
    `timeout` arg in seconds.

    On the desktop, this maps to speech.listen.request_voice_answer which
    routes the next utterance from the unified wake-word listener into
    answer-mode."""
    global _voice_answer_fn
    _voice_answer_fn = fn


def speak(text: str) -> None:
    """Speak via the registered TTS. No-op if nothing registered."""
    if _speak_fn:
        try:
            _speak_fn(text)
        except Exception:
            pass


def request_voice_answer(timeout: float = 12.0) -> Optional[str]:
    """Block for the user's voice answer. Returns None on timeout or
    if the user interrupted with a wake-word command."""
    if _voice_answer_fn:
        try:
            return _voice_answer_fn(timeout)
        except Exception:
            return None
    return None


def has_voice() -> bool:
    """True if both speak and voice-answer are wired up."""
    return _speak_fn is not None and _voice_answer_fn is not None