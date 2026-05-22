# import sounddevice as sd
# import numpy as np
# import time

# print("Recording 2 seconds — speak into mic now...")
# audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='float32')
# sd.wait()
# print(f"Captured {len(audio)} samples")
# print(f"max_amp = {float(np.abs(audio).max()):.4f}")
# print(f"mean_amp = {float(np.abs(audio).mean()):.4f}")
"""
Test 5: Voice processing requires the OUTPUT graph to be active too.
Attach a player, connect through main mixer, play silence to keep VP engaged.
"""
import time
import numpy as np
from AVFoundation import (
    AVAudioEngine, AVAudioPlayerNode, AVAudioFormat, AVAudioPCMBuffer,
)

engine = AVAudioEngine.alloc().init()
input_node = engine.inputNode()
output_node = engine.outputNode()

# Enable voice processing on input (auto-enables on output)
ok, err = input_node.setVoiceProcessingEnabled_error_(True, None)
print(f"VP on input: {ok}")
ok2, err2 = output_node.setVoiceProcessingEnabled_error_(True, None)
print(f"VP on output: {ok2}")

# Attach a player node and wire it to main mixer
# (mainMixerNode auto-connects to outputNode when first accessed)
player = AVAudioPlayerNode.alloc().init()
engine.attachNode_(player)
play_fmt = AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(
    48000.0, 1
)
engine.connect_to_format_(player, engine.mainMixerNode(), play_fmt)
print("Player attached, mainMixer connected to outputNode")

# Read input format AFTER everything's wired
fmt = input_node.outputFormatForBus_(0)
n_ch = int(fmt.channelCount())
print(f"Input format: sr={fmt.sampleRate()} ch={n_ch}\n")

channel_amps = [0.0] * n_ch
calls = [0]

def tap_block(buffer, when):
    n = int(buffer.frameLength())
    if n == 0:
        return
    ch_data = buffer.floatChannelData()
    if ch_data is None:
        return
    calls[0] += 1
    for ch in range(n_ch):
        try:
            arr = np.array(ch_data[ch].as_tuple(n), dtype=np.float32)
            # Filter NaN/Inf to avoid spurious overflow
            valid = arr[np.isfinite(arr)]
            if len(valid) > 0:
                amp = float(np.abs(valid).max())
                if amp > channel_amps[ch]:
                    channel_amps[ch] = amp
        except Exception:
            pass

input_node.installTapOnBus_bufferSize_format_block_(0, 1024, None, tap_block)
engine.prepare()
ok, err = engine.startAndReturnError_(None)
print(f"engine start: ok={ok}")

# Schedule a long silent buffer on the player to keep output side
# active for the duration of the test (silence so we don't hear it)
silence_frames = 48000 * 6  # 6s of silence
silence_buf = AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
    play_fmt, silence_frames
)
silence_buf.setFrameLength_(silence_frames)
player.scheduleBuffer_completionHandler_(silence_buf, None)
player.play()
print("Player playing silent buffer (engages VP output)\n")

print("Speak LOUDLY for 5 seconds...\n")
for i in range(5):
    time.sleep(1)
    print(f"[{i+1}s] calls={calls[0]:3d}")
    for ch in range(n_ch):
        bar = "█" * int(min(channel_amps[ch] * 30, 50))
        marker = " ★" if channel_amps[ch] > 0.01 else "  "
        print(f"  ch{ch}: max={channel_amps[ch]:.4f}{marker} {bar}")
    print()
    for ch in range(n_ch):
        channel_amps[ch] = 0.0
    calls[0] = 0

player.stop()
engine.stop()
try:
    input_node.removeTapOnBus_(0)
except Exception:
    pass