"""
Production Acoustic Echo Cancellation engine for ARGUS.
Cross-platform (Mac / Windows / Linux) via sounddevice + speexdsp.

Owns the full-duplex audio stream (mic + speaker in one callback).
Speex DSP subtracts TTS audio from mic input in real time.
Result: ARGUS literally cannot hear itself.

Architecture:
  - sounddevice.Stream runs full-duplex at 16kHz, 10ms frames
  - Callback pulls TTS from playback queue → speaker
  - Same callback captures mic, AEC against the TTS frame
  - Cleaned mic chunks pushed to processing queue
  - Background thread feeds cleaned chunks to consumer (listen.py)

PLATFORM NOTES:
  macOS:   Built-in mic+speakers share one CoreAudio device. Just works.
  Windows: WASAPI default. Auto-falls back to blocksize=480 if needed.
  Linux:   PipeWire/Pulse work directly. ALSA-only may need device override.

Public API:
  aec_engine.start()
  aec_engine.stop()
  aec_engine.queue_playback(np.int16 mono 16kHz)
  aec_engine.set_audio_callback(fn)   # listen.py registers here
  aec_engine.is_playing                # bool — TTS currently playing
  aec_engine.clear_playback()          # interrupt support
"""
from __future__ import annotations

import os
import threading
import queue
import numpy as np
import sounddevice as sd
from speexdsp import EchoCanceller

from config_metrics.logging import log_debug


# ── Configuration ────────────────────────────────────────────────
SAMPLE_RATE          = 16000   # AEC + OWW + Whisper all at 16kHz
FRAME_SIZE           = 256     # 16ms @ 16kHz — voice-engine's tested frame size
FILTER_LENGTH        = 4096    # 256ms tail — covers macOS round-trip + reverb
PLAYBACK_QUEUE_MAX   = 200     # ~2 seconds buffered
PROCESS_QUEUE_MAX    = 200
DOWNSTREAM_CHUNK     = 512     # what listen.py expects (matches MIC_CHUNK_SAMPLES)

# Optional device override (for dev machines with multiple audio interfaces)
INPUT_DEVICE  = os.getenv("AEC_INPUT_DEVICE")
OUTPUT_DEVICE = os.getenv("AEC_OUTPUT_DEVICE")


class AECEngine:
    def __init__(self):
        self._echo_canceller = EchoCanceller.create(
            FRAME_SIZE,
            FILTER_LENGTH,
            SAMPLE_RATE,
        )
        self._playback_queue: queue.Queue = queue.Queue(maxsize=PLAYBACK_QUEUE_MAX)
        self._process_queue: queue.Queue = queue.Queue(maxsize=PROCESS_QUEUE_MAX)
        self._audio_callback = None
        self._stream = None
        self._process_thread = None
        self._running = threading.Event()
        self._playback_active = threading.Event()
        self._playback_lock = threading.Lock()
        
        # Reference delay buffer — fixes the temporal alignment bug
        # in full-duplex callbacks. Echo of TTS arrives at mic ~80-120ms
        # after we write to outdata. We feed AEC the historic reference
        # frame whose echo NOW appears in the mic capture.
        from collections import deque
        self._reference_delay = deque(maxlen=16)   # 16 frames × 16ms = 256ms history
        self._reference_delay_frames = 6           # ~96ms — typical Mac round-trip
        
        # Diagnostic state
        self._diag_counter = 0
        self._diag_last_log = 0.0
        self._diag_ref_energy = 0
        self._diag_mic_energy = 0
        self._diag_clean_energy = 0
        
    # ── Public API ─────────────────────────────────────────────────
    
    @property
    def is_playing(self) -> bool:
        return self._playback_active.is_set() or not self._playback_queue.empty()

    def set_audio_callback(self, fn) -> None:
        """fn(samples_int16: np.ndarray) → None — receives cleaned audio."""
        self._audio_callback = fn

    def queue_playback(self, audio_int16: np.ndarray) -> None:
        """Queue audio for speakers + AEC reference. Must be int16 mono 16kHz."""
        if audio_int16.dtype != np.int16:
            raise ValueError(f"Expected int16, got {audio_int16.dtype}")
        if audio_int16.ndim != 1:
            raise ValueError(f"Expected mono (1D), got shape {audio_int16.shape}")
        
        n = len(audio_int16)
        for i in range(0, n, FRAME_SIZE):
            chunk = audio_int16[i:i + FRAME_SIZE]
            if len(chunk) < FRAME_SIZE:
                chunk = np.concatenate([
                    chunk,
                    np.zeros(FRAME_SIZE - len(chunk), dtype=np.int16),
                ])
            try:
                self._playback_queue.put(chunk, timeout=2.0)
            except queue.Full:
                log_debug("[AEC] Playback queue full — dropping frame")
                return
        self._playback_active.set()

    def clear_playback(self) -> None:
        """Drop pending TTS playback (for interrupt)."""
        with self._playback_lock:
            while not self._playback_queue.empty():
                try:
                    self._playback_queue.get_nowait()
                except queue.Empty:
                    break
            self._playback_active.clear()

    def start(self) -> None:
        if self._stream is not None:
            log_debug("[AEC] Already started")
            return
        
        # Log devices for diagnosability
        try:
            devices = sd.query_devices()
            d_in  = sd.default.device[0]
            d_out = sd.default.device[1]
            log_debug(
                f"[AEC] Audio: {len(devices)} devices, "
                f"default in={d_in} ({devices[d_in]['name']!r}), "
                f"out={d_out} ({devices[d_out]['name']!r})"
            )
        except Exception as e:
            log_debug(f"[AEC] Device enum failed: {e}")
        
        self._running.set()
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="AECProcessor",
            daemon=True,
        )
        self._process_thread.start()
        
        device_arg = None
        if INPUT_DEVICE or OUTPUT_DEVICE:
            in_id  = INPUT_DEVICE  or sd.default.device[0]
            out_id = OUTPUT_DEVICE or sd.default.device[1]
            device_arg = (in_id, out_id)
            log_debug(f"[AEC] User-specified devices: in={in_id!r} out={out_id!r}")
        
        # Try preferred config (160-sample frames for tightest AEC sync)
        try:
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SIZE,
                channels=(1, 1),
                dtype='int16',
                callback=self._stream_callback,
                device=device_arg,
            )
            self._stream.start()
            log_debug(f"[AEC] Engine started — {SAMPLE_RATE}Hz, {FRAME_SIZE}-frame blocks")
            return
        except Exception as e:
            log_debug(f"[AEC] Preferred config failed: {type(e).__name__}: {e}")
        
        # Fallback: Windows WASAPI sometimes won't accept 160-sample blocks
        try:
            log_debug("[AEC] Retrying with blocksize=480 for OS compat")
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=480,
                channels=(1, 1),
                dtype='int16',
                callback=self._stream_callback,
                device=device_arg,
            )
            self._stream.start()
            log_debug("[AEC] Engine started with fallback (blocksize=480)")
            return
        except Exception as e:
            log_debug(f"[AEC] Fallback failed: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"Could not open audio stream. Set AEC_INPUT_DEVICE / "
                f"AEC_OUTPUT_DEVICE env vars to override.\n"
                f"Available:\n{sd.query_devices()}"
            ) from e

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log_debug("[AEC] Engine stopped")

    # ── sounddevice full-duplex callback ───────────────────────────
    
    def _stream_callback(self, indata, outdata, frames, time_info, status):
        """Runs every audio cycle. Must complete in <10ms.
        Handles variable frame size (Windows fallback may give 480 not 160).
        """
        if status:
            log_debug(f"[AEC] sounddevice status: {status}")
        
        try:
            mic_full     = indata[:, 0].astype(np.int16, copy=False)
            out_full     = np.zeros(frames, dtype=np.int16)
            cleaned_full = np.zeros(frames, dtype=np.int16)
            
            # Process in FRAME_SIZE chunks (speex requirement)
            for offset in range(0, frames, FRAME_SIZE):
                end = min(offset + FRAME_SIZE, frames)
                chunk_len = end - offset
                
                mic_chunk = mic_full[offset:end]
                if chunk_len < FRAME_SIZE:
                    mic_chunk = np.concatenate([
                        mic_chunk,
                        np.zeros(FRAME_SIZE - chunk_len, dtype=np.int16),
                    ])
                
                # Pull next TTS frame for FUTURE playback
                try:
                    tts_chunk = self._playback_queue.get_nowait()
                except queue.Empty:
                    tts_chunk = np.zeros(FRAME_SIZE, dtype=np.int16)
                    if self._playback_active.is_set() and self._playback_queue.empty():
                        self._playback_active.clear()
                
                # Push to delay line — this is the reference WE'LL use later,
                # when the audio actually echoes back to the mic.
                self._reference_delay.append(tts_chunk.copy())
                
                # Pull DELAYED reference whose echo arrives at the mic NOW.
                # If the delay line isn't full yet (first ~100ms), use silence.
                if len(self._reference_delay) > self._reference_delay_frames:
                    aec_reference = self._reference_delay[
                        len(self._reference_delay) - 1 - self._reference_delay_frames
                    ]
                else:
                    aec_reference = np.zeros(FRAME_SIZE, dtype=np.int16)
                
                # AEC math: subtract delayed reference from current mic capture
                cleaned_bytes = self._echo_canceller.process(
                    mic_chunk.tobytes(),
                    aec_reference.tobytes(),
                )
                cleaned_chunk = np.frombuffer(cleaned_bytes, dtype=np.int16)
                
                # Speaker gets the FRESH chunk (what should play next)
                # Cleaned audio gets the AEC output (echo of past audio removed)
                out_full[offset:end] = tts_chunk[:chunk_len]
                cleaned_full[offset:end] = cleaned_chunk[:chunk_len]
            
            outdata[:, 0] = out_full
            
            # ── Diagnostic logging: track signal energies once per second ──
            import time as _t
            self._diag_counter += 1
            self._diag_ref_energy   += int(np.abs(out_full).mean())
            self._diag_mic_energy   += int(np.abs(mic_full).mean())
            self._diag_clean_energy += int(np.abs(cleaned_full).mean())
            now = _t.time()
            if now - self._diag_last_log > 1.0:
                n = max(1, self._diag_counter)
                log_debug(
                    f"[AEC] cb/s={n}  "
                    f"ref={self._diag_ref_energy//n}  "
                    f"mic={self._diag_mic_energy//n}  "
                    f"clean={self._diag_clean_energy//n}  "
                    f"q={self._playback_queue.qsize()}"
                )
                self._diag_counter = 0
                self._diag_ref_energy = 0
                self._diag_mic_energy = 0
                self._diag_clean_energy = 0
                self._diag_last_log = now
                
            try:
                self._process_queue.put_nowait(cleaned_full.copy())
            except queue.Full:
                try:
                    self._process_queue.get_nowait()
                    self._process_queue.put_nowait(cleaned_full.copy())
                except queue.Empty:
                    pass
                log_debug("[AEC] Process queue overflow")
        except Exception as e:
            log_debug(f"[AEC] callback error: {type(e).__name__}: {e}")

    # ── Processing thread (consumes cleaned audio) ─────────────────
    
    def _process_loop(self):
        """Re-buffers speex 10ms frames into 32ms (512-sample) chunks
        that listen.py expects. Decoupled from audio callback to keep
        OWW/Whisper latency from causing audio dropouts."""
        accumulator = np.zeros(0, dtype=np.int16)
        
        while self._running.is_set():
            try:
                cleaned = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            accumulator = np.concatenate([accumulator, cleaned])
            
            while len(accumulator) >= DOWNSTREAM_CHUNK:
                chunk = accumulator[:DOWNSTREAM_CHUNK]
                accumulator = accumulator[DOWNSTREAM_CHUNK:]
                
                if self._audio_callback is not None:
                    try:
                        self._audio_callback(chunk)
                    except Exception as e:
                        log_debug(f"[AEC] consumer callback: {type(e).__name__}: {e}")


# Singleton
aec_engine = AECEngine()