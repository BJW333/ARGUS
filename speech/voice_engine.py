"""
speech/voice_engine.py

Platform-agnostic voice engine factory.

Returns a singleton with a stable public API:
  start(), stop(), queue_playback(), set_audio_callback(),
  is_playing (property), clear_playback()

Implementations:
  - macOS:   WebRTC APM via LiveKit  (production AEC, ~94% reduction)
  - Linux:   speex                   (legacy fallback, ~30% reduction)
  - Windows: not yet implemented     (port WebRTC APM path; should work)

Consumers (speechmanager, listen) import this singleton instead of
the platform-specific engines directly:

    from speech.voice_engine import voice_engine as aec_engine
"""
import platform


def _make_voice_engine():
    system = platform.system()
    if system == "Darwin":
        from speech.aec_engine_webrtc import aec_engine
        return aec_engine
    elif system == "Linux":
        from speech.aec_engine_speex import aec_engine
        return aec_engine
    elif system == "Windows":
        # WebRTC APM via LiveKit should work on Windows too —
        # just need to verify WASAPI behavior with full-duplex
        # sounddevice + 160-sample blocks. If it fails, fall back
        # to speex which had a tested 480-sample fallback path.
        raise NotImplementedError(
            "Windows voice engine TBD — port WebRTC APM via LiveKit"
        )
    else:
        raise RuntimeError(f"No voice engine for platform: {system}")


voice_engine = _make_voice_engine()
