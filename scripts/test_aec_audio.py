"""
Smoke test: LiveKit AudioProcessingModule.
"""
import numpy as np
import sys

print("Importing livekit...")
try:
    from livekit.rtc.apm import AudioProcessingModule
    from livekit.rtc import AudioFrame
    print("✓ LiveKit APM imported\n")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Try: pip install livekit")
    sys.exit(1)

# Initialize APM with AEC + NS + HPF (no AGC — we want raw levels)
print("Creating APM with AEC + NS + HPF...")
apm = AudioProcessingModule(
    echo_cancellation=True,
    noise_suppression=True,
    high_pass_filter=True,
    auto_gain_control=False,
)
print("✓ APM constructed\n")

# Discover what methods are exposed
methods = sorted([m for m in dir(apm) if not m.startswith('_')])
print(f"APM methods ({len(methods)}):")
for m in methods:
    print(f"  - {m}")
print()

# === Synthesized AEC test ===
# Reference (TTS): 440Hz tone
# Mic: same tone (echo) + small noise
# Expected: AEC should subtract the tone, leaving mostly noise

SR = 16000
FRAME_MS = 10
FRAME_SAMPLES = SR * FRAME_MS // 1000  # 160

# 1 second of audio
n_samples = SR
t = np.linspace(0, 1, n_samples, endpoint=False)
reference = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
noise = (np.random.randn(n_samples) * 50).astype(np.int16)
mic = reference + noise   # mic hears the speaker echo + small ambient noise

ref_rms = np.sqrt((reference.astype(np.float64) ** 2).mean())
mic_rms = np.sqrt((mic.astype(np.float64) ** 2).mean())
print(f"Reference RMS: {ref_rms:.1f}")
print(f"Mic RMS:       {mic_rms:.1f}\n")

print("Processing through APM (10ms frames)...")

cleaned_chunks = []
for i in range(0, n_samples, FRAME_SAMPLES):
    ref_frame_data = reference[i:i + FRAME_SAMPLES]
    mic_frame_data = mic[i:i + FRAME_SAMPLES]
    if len(ref_frame_data) < FRAME_SAMPLES:
        break
    
    # Wrap in AudioFrame: AudioFrame(data, sample_rate, num_channels, samples_per_channel)
    ref_frame = AudioFrame(
        data=ref_frame_data.tobytes(),
        sample_rate=SR,
        num_channels=1,
        samples_per_channel=FRAME_SAMPLES,
    )
    mic_frame = AudioFrame(
        data=mic_frame_data.tobytes(),
        sample_rate=SR,
        num_channels=1,
        samples_per_channel=FRAME_SAMPLES,
    )
    
    # Far-end first (reference)
    apm.process_reverse_stream(ref_frame)
    # Near-end (mic) — modified IN PLACE per LiveKit docs
    apm.process_stream(mic_frame)
    
    # mic_frame.data now contains cleaned audio
    cleaned = np.frombuffer(mic_frame.data, dtype=np.int16)
    cleaned_chunks.append(cleaned)

cleaned_full = np.concatenate(cleaned_chunks)
cleaned_rms = np.sqrt((cleaned_full.astype(np.float64) ** 2).mean())
reduction_pct = 100 * (1 - cleaned_rms / mic_rms) if mic_rms > 0 else 0

print(f"Cleaned RMS:   {cleaned_rms:.1f}")
print(f"Reduction:     {reduction_pct:.1f}%\n")

if reduction_pct > 60:
    print("✓✓ EXCELLENT — AEC working as expected (>60% reduction)")
elif reduction_pct > 30:
    print("✓ AEC partially working (filter still adapting on synthesized data)")
elif reduction_pct > 5:
    print("⚠ Minor reduction — filter may need warmup")
else:
    print("✗ No reduction — AEC isn't engaging")

print("\nDone.")