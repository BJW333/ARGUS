"""
Test your custom 'argus' OpenWakeWord model in real-time.

Shows confidence per audio frame. Speak "argus" — the bar should spike
above your threshold. Speak random stuff — should stay flat.

Usage:
    cd ~/Desktop/ARGUS
    python3.10 scripts/test_wakeword.py /path/to/argus.onnx
    python3.10 scripts/test_wakeword.py /path/to/argus.onnx --threshold 0.5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


SAMPLE_RATE = 16000
CHUNK_MS    = 80
CHUNK_SIZE  = int(SAMPLE_RATE * CHUNK_MS / 1000)   # 1280 samples per frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to argus.onnx or argus.tflite")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold (default 0.5)")
    parser.add_argument("--key", default=None,
                        help="Key in prediction dict (defaults to model filename stem)")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    framework = "onnx" if model_path.suffix == ".onnx" else "tflite"
    key = args.key or model_path.stem  # e.g. "argus" if file is argus.onnx

    print(f"Loading model: {model_path}")
    print(f"Framework:     {framework}")
    print(f"Prediction key: {key!r}")
    print(f"Threshold:     {args.threshold}")

    oww = Model(
        wakeword_models=[str(model_path)],
        inference_framework=framework,
    )

    # Warmup so first inference isn't slow
    oww.predict(np.zeros(CHUNK_SIZE, dtype=np.int16))

    print("\n🎤 Listening — speak 'argus' to test. Ctrl+C to stop.\n")
    print(f"{'score':>7}  bar (50 chars wide)")
    print("─" * 70)

    last_hit = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal last_hit
        if status:
            print(status, file=sys.stderr)

        # Convert float32 [-1, 1] → int16 (what openwakeword wants)
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

        t0 = time.perf_counter()
        prediction = oww.predict(audio_int16)
        latency_ms = (time.perf_counter() - t0) * 1000

        score = float(prediction.get(key, 0.0))
        bar_width = int(score * 50)
        bar = "█" * bar_width
        color = "\033[92m" if score >= args.threshold else "\033[90m"
        reset = "\033[0m"

        # Live updating line
        sys.stdout.write(
            f"\r{color}{score:>7.3f}{reset}  {bar:<50}  {latency_ms:5.1f}ms"
        )
        sys.stdout.flush()

        # Mark detections
        now = time.time()
        if score >= args.threshold and now - last_hit > 1.0:
            last_hit = now
            sys.stdout.write(
                f"\n  \033[1;92m🎯 DETECTED  score={score:.3f}  "
                f"latency={latency_ms:.1f}ms\033[0m\n"
            )
            sys.stdout.flush()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=callback,
        ):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()