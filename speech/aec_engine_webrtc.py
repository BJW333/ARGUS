"""
speech/aec_engine_webrtc.py

Production acoustic echo cancellation using LiveKit's WebRTC
Audio Processing Module — same algorithm Discord, Chrome, Zoom use.

Smoke test on synthesized 440Hz tone showed 94.7% reduction.
Real-world performance with TTS through built-in speakers expected
~70-85%, which is enough to break the self-transcription loop
reliably while preserving voice interrupt capability.

Architecture:
  sounddevice full-duplex Stream (16kHz, 10ms frames)
    callback(indata, outdata):
      outdata = pull next TTS frame from playback queue
      apm.process_reverse_stream(ref_frame)   # far-end (reference)
      apm.process_stream(mic_frame)           # near-end (cleaned in place)
      push cleaned mic_frame.data to processing queue

  Processing thread re-buffers cleaned audio into 512-sample chunks
  and forwards to listen.py's consumer callback.

KEY DIFFERENCE FROM SPEEX ENGINE:
  Speex used an external 6-frame reference delay buffer for alignment.
  WebRTC APM handles this internally via set_stream_delay_ms +
  adaptive delay estimation. Do NOT add an external delay buffer
  here — it would fight WebRTC's internal alignment.

Public API matches aec_engine_speex.py for drop-in compatibility.
"""
from __future__ import annotations

import os
import threading
import queue
import time
import numpy as np
import sounddevice as sd

from livekit.rtc.apm import AudioProcessingModule
from livekit.rtc import AudioFrame

from config_metrics.logging import log_debug


# ── Configuration ────────────────────────────────────────────────
SAMPLE_RATE          = 16000   # WebRTC APM operates at 16kHz natively
FRAME_SIZE           = 160     # 10ms @ 16kHz — APM REQUIRES exactly this
DOWNSTREAM_CHUNK     = 512     # what listen.py expects (matches MIC_CHUNK_SAMPLES)
PLAYBACK_QUEUE_MAX   = 200     # ~2 seconds buffered
PROCESS_QUEUE_MAX    = 200
SYSTEM_DELAY_MS      = 100     # initial round-trip estimate; APM adapts

# Optional device override (for dev machines with multiple audio interfaces)
INPUT_DEVICE  = os.getenv("AEC_INPUT_DEVICE")
OUTPUT_DEVICE = os.getenv("AEC_OUTPUT_DEVICE")


class WebRTCAECEngine:
    """Production AEC via LiveKit's WebRTC APM bindings."""

    def __init__(self):
        # WebRTC APM with AEC only — NS/HPF off for wake-word sensitivity.
        # AGC off — keep TTS at intended levels, no auto-leveling on mic side.
        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=False, # WebRTC NS is very aggressive and distorts voice too much. We rely on the mic's built-in NS (if any) instead.
            high_pass_filter=False, # WebRTC HPF is very aggressive and cuts off too much of the voice.
            auto_gain_control=False,
        )
        self._apm.set_stream_delay_ms(SYSTEM_DELAY_MS)
        log_debug(
            f"[WAEC] WebRTC APM initialized: AEC only (NS/HPF off), "
            f"delay={SYSTEM_DELAY_MS}ms"
        )

        # Queues
        self._playback_queue: queue.Queue = queue.Queue(maxsize=PLAYBACK_QUEUE_MAX)
        self._process_queue: queue.Queue = queue.Queue(maxsize=PROCESS_QUEUE_MAX)

        # Consumer
        self._audio_callback = None

        # Stream + thread
        self._stream = None
        self._process_thread = None
        self._running = threading.Event()
        self._playback_active = threading.Event()
        self._playback_lock = threading.Lock()
        self._interrupt = threading.Event()  # signal queue_playback to abort mid-feed
        
        # Frame re-buffering for sounddevice blocksize != FRAME_SIZE
        # (Mainly relevant on the 480-sample fallback path — at the
        # preferred blocksize=160, accumulator stays empty between calls.)
        self._mic_acc = np.zeros(0, dtype=np.int16)
        self._ref_acc = np.zeros(0, dtype=np.int16)

        # Diagnostics
        self._diag_calls = 0
        self._diag_last_log = 0.0
        self._diag_ref_energy = 0
        self._diag_mic_energy = 0
        self._diag_clean_energy = 0

    # ── Public API ────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._playback_active.is_set() or not self._playback_queue.empty()

    def set_audio_callback(self, fn) -> None:
        """fn(samples_int16: np.ndarray) → None — receives cleaned audio."""
        self._audio_callback = fn

    # def queue_playback(self, audio_int16: np.ndarray) -> None:
    #     """Queue audio for speakers + AEC reference. Must be int16 mono 16kHz."""
    #     if audio_int16.dtype != np.int16:
    #         raise ValueError(f"Expected int16, got {audio_int16.dtype}")
    #     if audio_int16.ndim != 1:
    #         raise ValueError(f"Expected mono (1D), got shape {audio_int16.shape}")

    #     n = len(audio_int16)
    #     for i in range(0, n, FRAME_SIZE):
    #         chunk = audio_int16[i:i + FRAME_SIZE]
    #         if len(chunk) < FRAME_SIZE:
    #             chunk = np.concatenate([
    #                 chunk,
    #                 np.zeros(FRAME_SIZE - len(chunk), dtype=np.int16),
    #             ])
    #         try:
    #             self._playback_queue.put(chunk, timeout=2.0)
    #         except queue.Full:
    #             log_debug("[WAEC] playback queue full — dropping frame")
    #             return
    #     self._playback_active.set()

    def queue_playback(self, audio_int16: np.ndarray) -> None:
        """Queue audio for speakers + AEC reference. Must be int16 mono 16kHz.
        Long utterances block here for their full duration as the queue
        drains in real time. _interrupt allows clear_playback() to abort
        mid-feed."""
        if audio_int16.dtype != np.int16:
            raise ValueError(f"Expected int16, got {audio_int16.dtype}")
        if audio_int16.ndim != 1:
            raise ValueError(f"Expected mono (1D), got shape {audio_int16.shape}")

        # Fresh playback — clear any stale abort signal from prior utterance
        self._interrupt.clear()

        n = len(audio_int16)
        total_frames = (n + FRAME_SIZE - 1) // FRAME_SIZE
        for i in range(0, n, FRAME_SIZE):
            # Abort if clear_playback() was called (wake word during TTS)
            if self._interrupt.is_set():
                log_debug(
                    f"[WAEC] queue_playback aborted at frame "
                    f"{i // FRAME_SIZE}/{total_frames}"
                )
                return
            chunk = audio_int16[i:i + FRAME_SIZE]
            if len(chunk) < FRAME_SIZE:
                chunk = np.concatenate([
                    chunk,
                    np.zeros(FRAME_SIZE - len(chunk), dtype=np.int16),
                ])
            try:
                self._playback_queue.put(chunk, timeout=2.0)
            except queue.Full:
                log_debug("[WAEC] playback queue full — dropping frame")
                return
        self._playback_active.set()
        
    # def clear_playback(self) -> None:
    #     """Drop pending TTS audio (interrupt support)."""
    #     with self._playback_lock:
    #         n_before = self._playback_queue.qsize()
    #         while not self._playback_queue.empty():
    #             try:
    #                 self._playback_queue.get_nowait()
    #             except queue.Empty:
    #                 break
    #         self._playback_active.clear()
    #         log_debug(f"[WAEC] clear_playback: drained {n_before} frames")
            
    def clear_playback(self) -> None:
        """Drop pending TTS audio (interrupt support).
        Signals queue_playback() to abort mid-feed so long utterances
        actually stop instead of refilling the queue we just drained."""
        # Signal abort BEFORE draining — queue_playback may be mid-iteration
        self._interrupt.set()
        with self._playback_lock:
            n_before = self._playback_queue.qsize()
            while not self._playback_queue.empty():
                try:
                    self._playback_queue.get_nowait()
                except queue.Empty:
                    break
            self._playback_active.clear()
            log_debug(f"[WAEC] clear_playback: drained {n_before} frames (interrupt set)")
            
    def start(self) -> None:
        if self._stream is not None:
            log_debug("[WAEC] already started")
            return

        # Log devices for diagnosability
        try:
            devs = sd.query_devices()
            d_in, d_out = sd.default.device[0], sd.default.device[1]
            log_debug(
                f"[WAEC] Audio: {len(devs)} devices, "
                f"in={d_in} ({devs[d_in]['name']!r}), "
                f"out={d_out} ({devs[d_out]['name']!r})"
            )
        except Exception as e:
            log_debug(f"[WAEC] device enum failed: {e}")

        self._running.set()

        # Processing thread
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="WebRTCAECProcessor",
            daemon=True,
        )
        self._process_thread.start()

        device_arg = None
        if INPUT_DEVICE or OUTPUT_DEVICE:
            in_id  = INPUT_DEVICE  or sd.default.device[0]
            out_id = OUTPUT_DEVICE or sd.default.device[1]
            device_arg = (in_id, out_id)
            log_debug(f"[WAEC] user devices: in={in_id!r} out={out_id!r}")

        # Try preferred config (frame-aligned blocksize for tightest AEC sync)
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
            log_debug(
                f"[WAEC] Engine started — {SAMPLE_RATE}Hz, "
                f"{FRAME_SIZE}-frame blocks (10ms)"
            )
            return
        except Exception as e:
            log_debug(f"[WAEC] preferred config failed: {type(e).__name__}: {e}")

        # Fallback: larger blocksize (some hardware won't open 160 reliably)
        try:
            log_debug("[WAEC] retrying with blocksize=480 (30ms)")
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=480,
                channels=(1, 1),
                dtype='int16',
                callback=self._stream_callback,
                device=device_arg,
            )
            self._stream.start()
            log_debug("[WAEC] Engine started with fallback blocksize=480 (30ms)")
            return
        except Exception as e:
            log_debug(f"[WAEC] fallback failed: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"Could not open audio stream: {e}. "
                f"Try setting AEC_INPUT_DEVICE / AEC_OUTPUT_DEVICE env vars.\n"
                f"Available:\n{sd.query_devices()}"
            ) from e

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log_debug("[WAEC] Engine stopped")

    # ── sounddevice full-duplex callback ──────────────────────────

    def _stream_callback(self, indata, outdata, frames, time_info, status):
        """Runs on every audio cycle. Must complete in <10ms.
        WebRTC APM requires EXACT 10ms (160-sample) frames, so we
        accumulate when sounddevice gives us a different blocksize
        and drain in 160-sample units."""
        if status:
            log_debug(f"[WAEC] sd status: {status}")

        try:
            # ── OUT: assemble TTS frames for this block ──
            tts_full = np.zeros(frames, dtype=np.int16)
            for offset in range(0, frames, FRAME_SIZE):
                end = min(offset + FRAME_SIZE, frames)
                chunk_len = end - offset
                try:
                    tts_chunk = self._playback_queue.get_nowait()
                except queue.Empty:
                    tts_chunk = np.zeros(FRAME_SIZE, dtype=np.int16)
                    if self._playback_active.is_set() and self._playback_queue.empty():
                        self._playback_active.clear()
                tts_full[offset:end] = tts_chunk[:chunk_len]
            outdata[:, 0] = tts_full

            # ── IN: copy mic capture out of CoreAudio's buffer ──
            mic_full = indata[:, 0].astype(np.int16, copy=True)

            # ── Accumulate for APM (which needs exact 10ms frames) ──
            self._mic_acc = np.concatenate([self._mic_acc, mic_full])
            self._ref_acc = np.concatenate([self._ref_acc, tts_full])

            cleaned_chunks = []
            while (len(self._mic_acc) >= FRAME_SIZE
                   and len(self._ref_acc) >= FRAME_SIZE):
                mic_frame_data = self._mic_acc[:FRAME_SIZE]
                ref_frame_data = self._ref_acc[:FRAME_SIZE]
                self._mic_acc = self._mic_acc[FRAME_SIZE:]
                self._ref_acc = self._ref_acc[FRAME_SIZE:]

                # Wrap in LiveKit AudioFrames
                ref_frame = AudioFrame(
                    data=ref_frame_data.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=FRAME_SIZE,
                )
                mic_frame = AudioFrame(
                    data=mic_frame_data.tobytes(),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=FRAME_SIZE,
                )

                # Far-end FIRST (TTS reference) — REQUIRED order per WebRTC spec
                self._apm.process_reverse_stream(ref_frame)
                # Then near-end (mic) — modified IN PLACE per LiveKit API
                self._apm.process_stream(mic_frame)

                cleaned = np.frombuffer(mic_frame.data, dtype=np.int16)
                cleaned_chunks.append(cleaned)

            if cleaned_chunks:
                cleaned_full = np.concatenate(cleaned_chunks)

                # Diagnostics — track signal energies once per second
                self._diag_calls += 1
                self._diag_ref_energy   += int(np.abs(tts_full).mean())
                self._diag_mic_energy   += int(np.abs(mic_full).mean())
                self._diag_clean_energy += int(np.abs(cleaned_full).mean())
                now = time.time()
                if now - self._diag_last_log > 1.0:
                    n = max(1, self._diag_calls)
                    log_debug(
                        f"[WAEC] cb/s={n}  "
                        f"ref={self._diag_ref_energy//n}  "
                        f"mic={self._diag_mic_energy//n}  "
                        f"clean={self._diag_clean_energy//n}  "
                        f"q={self._playback_queue.qsize()}"
                    )
                    self._diag_calls = 0
                    self._diag_ref_energy = 0
                    self._diag_mic_energy = 0
                    self._diag_clean_energy = 0
                    self._diag_last_log = now

                # Hand cleaned audio to processing thread
                try:
                    self._process_queue.put_nowait(cleaned_full.copy())
                except queue.Full:
                    try:
                        self._process_queue.get_nowait()
                        self._process_queue.put_nowait(cleaned_full.copy())
                    except queue.Empty:
                        pass
                    log_debug("[WAEC] process queue overflow")
        except Exception as e:
            log_debug(f"[WAEC] callback error: {type(e).__name__}: {e}")

    # ── Processing thread (consumes cleaned audio) ────────────────

    def _process_loop(self):
        """Re-buffer cleaned audio into DOWNSTREAM_CHUNK for listen.py.
        Decoupled from audio callback so OWW/Whisper latency can't
        cause audio dropouts."""
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
                        log_debug(f"[WAEC] consumer cb: {type(e).__name__}: {e}")


# Singleton (matches speex engine pattern)
aec_engine = WebRTCAECEngine()
