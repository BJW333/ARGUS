"""
Production speech engine — Hybrid OpenWakeWord + RealtimeSTT.

THIS VERSION INCLUDES DIAGNOSTIC LOGGING. Every 5 seconds, prints:
  [DIAG] chunks/5s=X  max_score=0.YYY  armed=True/False  speaking=True/False
That tells you instantly whether:
  - Audio is flowing (chunks/5s should be ~150)
  - OWW is seeing your voice (max_score should hit ~0.5+ when you say "argus")
  - Wake state is firing/expiring as expected

Architecture (FINAL):

  We own the microphone stream via sounddevice. Audio flows
  continuously to BOTH engines:

  ┌────────────────────────────────────────────────────────────────┐
  │ sounddevice mic stream (16 kHz, mono, ~30 Hz chunk rate)       │
  │                                                                │
  │    audio chunk                                                 │
  │       │                                                        │
  │       ├──► RMS → smoothed → QML AnimatedCanvas pulse           │
  │       ├──► OpenWakeWord.predict() (continuous, every chunk)    │
  │       └──► RealtimeSTT.feed_audio() (continuous)               │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘

Public API preserved exactly. Nothing downstream needs changes.
"""
from __future__ import annotations

import os
import re
import time
import threading
import queue
from typing import Optional

import numpy as np
import sounddevice as sd
from RealtimeSTT import AudioToTextRecorder
from openwakeword.model import Model as OpenWakeWordModel

from speech.speechmanager import speech_manager
from speech.speak import speak
from config_metrics.logging import log_debug
from core.input_bus import send as submit_user_input, print_to_gui


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

SAMPLE_RATE        = 16000   # Hz
MIC_CHUNK_SAMPLES  = 512     # 32 ms blocks
OWW_CHUNK_SAMPLES  = 1280    # 80 ms — OpenWakeWord native window

WAKE_TIMEOUT_S     = 6.0     # Wake state expires this long after firing
                              # (was 2.5 — too short for longer commands)
WAKE_DEBOUNCE_S    = 1.0     # Min interval between wake events

DIAG_INTERVAL_S    = 5.0     # Diagnostic dump every N seconds


# ════════════════════════════════════════════════════════════════════
# DIAGNOSTIC COUNTERS
# ════════════════════════════════════════════════════════════════════

_diag_chunks_count    = 0
_diag_max_score       = 0.0
_diag_last_dump_time  = 0.0
_diag_lock            = threading.Lock()


def _diagnostic_dump_if_due() -> None:
    """Periodically log internal state so we can debug live."""
    global _diag_chunks_count, _diag_max_score, _diag_last_dump_time
    now = time.time()
    if now - _diag_last_dump_time < DIAG_INTERVAL_S:
        return
    with _diag_lock:
        chunks = _diag_chunks_count
        max_score = _diag_max_score
        _diag_chunks_count = 0
        _diag_max_score = 0.0
        _diag_last_dump_time = now
    armed = _wake_armed.is_set()
    speaking = _is_argus_speaking()
    log_debug(
        f"[DIAG] chunks/5s={chunks}  max_score={max_score:.3f}  "
        f"armed={armed}  argus_speaking={speaking}"
    )


# ════════════════════════════════════════════════════════════════════
# AUDIO LEVEL FEED (smoothed)
# ════════════════════════════════════════════════════════════════════

_smoothed_level   = 0.0
_LEVEL_SMOOTHING  = 0.35


def _feed_audio_level_to_gui(samples_int16: np.ndarray) -> None:
    global _smoothed_level
    try:
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if not app or not hasattr(app, "backend"):
            return
        if samples_int16.size == 0:
            return
        max_val = float(2 ** 15)
        rms = float(np.sqrt(np.mean(samples_int16.astype(np.float32) ** 2))) / max_val
        instant = min(1.0, rms * 12.0)
        _smoothed_level = (
            (1.0 - _LEVEL_SMOOTHING) * _smoothed_level
            + _LEVEL_SMOOTHING * instant
        )
        app.backend.setAudioLevel(_smoothed_level)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════
# PAUSE CONTROL
# ════════════════════════════════════════════════════════════════════

_listen_pause_event  = threading.Event()
_listen_paused_event = threading.Event()


def pause_wake_listener() -> None:
    log_debug("[LISTEN] Pause requested")
    _listen_pause_event.set()
    _listen_paused_event.wait(timeout=3.0)
    log_debug("[LISTEN] Listener paused")


def resume_wake_listener() -> None:
    log_debug("[LISTEN] Resume requested")
    _listen_paused_event.clear()
    _listen_pause_event.clear()
    log_debug("[LISTEN] Listener resumed")


# ════════════════════════════════════════════════════════════════════
# UNIFIED LISTENER STATE
# ════════════════════════════════════════════════════════════════════

_expecting_answer = threading.Event()
_answer_queue: "queue.Queue[Optional[str]]" = queue.Queue()


def request_voice_answer(timeout: float = 12.0) -> Optional[str]:
    while not _answer_queue.empty():
        try: _answer_queue.get_nowait()
        except queue.Empty: break
    log_debug("[LISTEN] Answer-mode ON")
    _expecting_answer.set()
    try:
        answer = _answer_queue.get(timeout=timeout)
        log_debug(f"[LISTEN] Answer received: {answer!r}")
        return answer
    except queue.Empty:
        log_debug("[LISTEN] Answer-mode timed out")
        return None
    finally:
        _expecting_answer.clear()
        log_debug("[LISTEN] Answer-mode OFF")


# ════════════════════════════════════════════════════════════════════
# WAKE-WORD ENGINE (direct OpenWakeWord)
# ════════════════════════════════════════════════════════════════════

_oww_model: Optional[OpenWakeWordModel] = None
_oww_buffer = np.empty(0, dtype=np.int16)
_oww_lock   = threading.Lock()

_wake_armed     = threading.Event()
_last_wake_time = 0.0


def _resolve_oww_model_path() -> str:
    env_path = os.getenv("OWW_MODEL_PATH", "").strip()
    candidates = [env_path] if env_path else []
    candidates += [
        "models/wakeword/argus.onnx",
        "models/wakeword/argus.tflite",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return candidates[0] if candidates else "models/wakeword/argus.onnx"


def _build_oww_model() -> OpenWakeWordModel:
    model_path = _resolve_oww_model_path()
    framework = "onnx" if model_path.endswith(".onnx") else "tflite"
    log_debug(f"[LISTEN] Loading OWW model: {model_path} ({framework})")
    model = OpenWakeWordModel(
        wakeword_models=[model_path],
        inference_framework=framework,
    )
    model.predict(np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16))
    return model


def _process_chunk_for_wake_word(chunk_int16: np.ndarray) -> None:
    global _oww_buffer, _last_wake_time, _diag_max_score

    if _oww_model is None or _listen_pause_event.is_set():
        return

    # Lower wake threshold during TTS — WebRTC AEC's nonlinear echo
    # suppressor over-attenuates the user's voice during double-talk,
    # so OWW scores are systematically lower while ARGUS is speaking.
    if _is_argus_speaking():
        threshold = float(os.getenv("WAKE_SENSITIVITY_SPEAKING", "0.35"))
    else:
        threshold = float(os.getenv("WAKE_SENSITIVITY", "0.5"))

    with _oww_lock:
        _oww_buffer = np.concatenate([_oww_buffer, chunk_int16])

        while len(_oww_buffer) >= OWW_CHUNK_SAMPLES:
            window = _oww_buffer[:OWW_CHUNK_SAMPLES]
            _oww_buffer = _oww_buffer[OWW_CHUNK_SAMPLES:]

            try:
                prediction = _oww_model.predict(window)
                score = max(prediction.values()) if prediction else 0.0
            except Exception as e:
                log_debug(f"[LISTEN] OWW predict error: {e}")
                continue

            # Track max score for diagnostics
            with _diag_lock:
                if score > _diag_max_score:
                    _diag_max_score = score

            if score >= threshold:
                now = time.time()
                if now - _last_wake_time < WAKE_DEBOUNCE_S:
                    continue
                _last_wake_time = now
                _wake_armed.set()
                _on_wake_word_event(score)


def _on_wake_word_event(score: float = 0.0) -> None:
    log_debug(f"[LISTEN] 🎯 Wake word detected (score={score:.3f}) — TTS interrupt")
    try:
        speech_manager.stop()
        log_debug("[LISTEN] speech_manager.stop() called successfully")
    except Exception as e:
        log_debug(f"[LISTEN] speech_manager.stop() error: {e}")


# ════════════════════════════════════════════════════════════════════
# RECORDER (RealtimeSTT — VAD + Whisper, fed via feed_audio)
# ════════════════════════════════════════════════════════════════════

_recorder: Optional[AudioToTextRecorder] = None
_recorder_lock = threading.Lock()


def _is_argus_speaking() -> bool:
    """Check speech_manager.speaking flag (we verified this attribute exists)."""
    try:
        if hasattr(speech_manager, "speaking"):
            return bool(speech_manager.speaking)
        if hasattr(speech_manager, "is_speaking"):
            return bool(speech_manager.is_speaking())
    except Exception:
        pass
    return False


def _build_recorder() -> AudioToTextRecorder:
    whisper_model = os.getenv("WHISPER_MODEL", "small.en")
    log_debug(f"[LISTEN] Building recorder (VAD+Whisper, fed audio): {whisper_model}")

    return AudioToTextRecorder(
        model=whisper_model,
        language="en",
        compute_type="int8", # "int8" also works but is less accurate; "float16" is a good sweet spot for speed + quality on modern GPUs

        silero_sensitivity=0.4,
        silero_use_onnx=True,
        silero_deactivity_detection=True,
        webrtc_sensitivity=3,

        post_speech_silence_duration=0.6,
        min_length_of_recording=0.3,
        min_gap_between_recordings=0.3,

        use_microphone=False,   # We feed via feed_audio()

        spinner=False,
        enable_realtime_transcription=False,
    )


# ════════════════════════════════════════════════════════════════════
# MIC STREAM CALLBACK (sounddevice)
# ════════════════════════════════════════════════════════════════════

def _audio_callback_from_aec(samples_int16: np.ndarray) -> None:
    """Receives CLEANED audio chunks from the AEC engine.
    
    The audio has already had TTS subtracted by WebRTC AEC, so
    wake-word detection and Whisper feed run on clean audio with
    no false positives from ARGUS's own voice.
    """
    global _diag_chunks_count
    
    try:
        # Diagnostic counter
        with _diag_lock:
            _diag_chunks_count += 1
        _diagnostic_dump_if_due()
        
        # GUI level pulse (use cleaned audio so meter only reacts to user)
        _feed_audio_level_to_gui(samples_int16)
        
        if _listen_pause_event.is_set():
            return
        
        # OWW wake detection on CLEAN audio
        _process_chunk_for_wake_word(samples_int16)
        
        # Whisper transcription pipeline on CLEAN audio
        if _recorder is not None:
            try:
                _recorder.feed_audio(samples_int16.tobytes())
            except Exception as e:
                log_debug(f"[LISTEN] feed_audio error: {e}")
    except Exception as e:
        log_debug(f"[LISTEN] aec callback error: {type(e).__name__}: {e}")
        
def _audio_callback(indata, frames, time_info, status) -> None:
    global _diag_chunks_count

    if status:
        log_debug(f"[LISTEN] sounddevice status: {status}")

    try:
        # float32 [-1, 1] → int16
        samples_int16 = (indata[:, 0] * 32767.0).astype(np.int16)

        # Diagnostic: count chunks
        with _diag_lock:
            _diag_chunks_count += 1
        _diagnostic_dump_if_due()

        # GUI level (continuous, smooth)
        _feed_audio_level_to_gui(samples_int16)

        if _listen_pause_event.is_set():
            return

        # OWW wake detection (continuous)
        _process_chunk_for_wake_word(samples_int16)

        # RealtimeSTT transcription pipeline (continuous feed)
        if _recorder is not None:
            try:
                _recorder.feed_audio(samples_int16.tobytes())
            except Exception as e:
                log_debug(f"[LISTEN] feed_audio error: {e}")
    except Exception as e:
        log_debug(f"[LISTEN] audio_callback error: {type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════
# TRANSCRIPTION ROUTING
# ════════════════════════════════════════════════════════════════════

def _on_transcription(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return

    if _listen_pause_event.is_set():
        log_debug(f"[STT] Dropped (paused): {text!r}")
        return

    armed = (
        _wake_armed.is_set()
        and (time.time() - _last_wake_time) < WAKE_TIMEOUT_S
    )
    expecting = _expecting_answer.is_set()
    speaking = _is_argus_speaking()

    log_debug(
        f"[STT] {text!r}  (armed={armed} expecting={expecting} speaking={speaking})"
    )

    # Route 1: wake word during answer-mode = INTERRUPT
    if expecting and armed:
        log_debug("[LISTEN] Answer-mode interrupted by wake word")
        _answer_queue.put(None)
        _expecting_answer.clear()
        cmd = re.sub(r"\b(argus)\b", "", text, flags=re.IGNORECASE).strip()
        cmd = re.sub(r"\s+", " ", cmd) or text
        print_to_gui("Me  --> ", cmd)
        _wake_armed.clear()
        handle_wake_word_detected(cmd)
        return

    # Route 2: wake-word command
    if armed:
        cmd = re.sub(r"\b(argus)\b", "", text, flags=re.IGNORECASE).strip()
        cmd = re.sub(r"\s+", " ", cmd) or text
        print(f"Me  -->  {text}")
        print_to_gui("Me  --> ", cmd)
        _wake_armed.clear()
        handle_wake_word_detected(cmd)
        return

    # Route 3: ask_user answer (no wake word)
    if expecting:
        if speaking:
            log_debug(f"[STT] Dropped during TTS (answer-mode): {text!r}")
            return
        print_to_gui("Me  --> ", text)
        _answer_queue.put(text)
        return

    # Drop
    if speaking:
        log_debug(f"[STT] Dropped (TTS echo): {text!r}")
    else:
        log_debug(f"[STT] Dropped (no wake word): {text!r}")


# ════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════

def listen_for_wake_word() -> None:
    global _oww_model, _recorder, _diag_last_dump_time

    print("listen function is running (Hybrid OWW + RealtimeSTT, owned mic) [DIAG MODE]")
    log_debug("[LISTEN] Starting hybrid speech engine with owned mic stream")
    log_debug(f"[LISTEN] Config: WAKE_TIMEOUT={WAKE_TIMEOUT_S}s "
              f"WAKE_SENSITIVITY={os.getenv('WAKE_SENSITIVITY', '0.5')} "
              f"WHISPER_MODEL={os.getenv('WHISPER_MODEL', 'small.en')}")

    _diag_last_dump_time = time.time()

    with _recorder_lock:
        if _oww_model is None:
            try:
                _oww_model = _build_oww_model()
                log_debug("[LISTEN] OpenWakeWord ready")
            except Exception as e:
                log_debug(f"[LISTEN] FATAL: OWW load failed: {e}")
                raise

        if _recorder is None:
            try:
                _recorder = _build_recorder()
                log_debug("[LISTEN] RealtimeSTT recorder ready")
            except Exception as e:
                log_debug(f"[LISTEN] FATAL: recorder build failed: {e}")
                raise

    try:
        # ── AEC engine owns the mic now ──
        # Register our consumer callback, then start the engine.
        # Engine opens full-duplex sounddevice stream; we receive
        # cleaned audio chunks via _audio_callback_from_aec.
        from speech.voice_engine import voice_engine as aec_engine
        aec_engine.set_audio_callback(_audio_callback_from_aec)
        aec_engine.start()
        log_debug("[LISTEN] AEC engine connected — TTS-free mic feed active")
        
        try:
            while True:
                try:
                    _recorder.text(_on_transcription)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log_debug(
                        f"[LISTEN] Transcription error: {type(e).__name__}: {e} "
                        f"— retry in 1s"
                    )
                    time.sleep(1)
        finally:
            try:
                if _recorder is not None:
                    _recorder.shutdown()
            except Exception as e:
                log_debug(f"[LISTEN] recorder shutdown error: {e}")
            aec_engine.stop()
    except Exception as e:
        log_debug(f"[LISTEN] FATAL audio engine error: {e}")
        raise
    
    
def generalvoiceinput() -> Optional[str]:
    return request_voice_answer(timeout=8.0)

def handle_wake_word_detected(maininputcommand: str) -> None:
    speech_manager.stop()
    if maininputcommand:
        submit_user_input(maininputcommand)
    else:
        log_debug("[LISTEN] Wake-word fired but command is empty")
        speak("Yes? What would you like me to do?")