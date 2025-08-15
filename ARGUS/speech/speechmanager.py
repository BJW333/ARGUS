import threading
import queue
import tempfile
import os
import subprocess
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_sound


class SpeechManager:
    def __init__(self, pitch_semitones=0.5, speed_factor=1.05): #orginal speed was 1.06 and orginal pitch 0.5
        self.speech_queue = queue.Queue()
        self.current_playback = None
        self.speaking = False
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.thread.start()
        self.current_process = None  # For mimic3 subprocess
        #pitch and speed settings
        self.pitch_semitones = pitch_semitones
        self.speed_factor = speed_factor
        
    def _speak_loop(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self._speak_text(text)
            
    def clear_queue(self):
        # WARNING: This directly accesses the internal queue and mutex.
        # This is a fast way to clear all pending items, but it is not thread-safe if other threads
        # might add items to the queue immediately after clearing.
        # For this application (single speech manager, one producer), this is acceptable,
        # but be aware in multi-threaded or distributed systems.
        with self.speech_queue.mutex:
            self.speech_queue.queue.clear()

    def shutdown_speech(self):
        """Signal the speech thread to shut down cleanly."""
        self.speech_queue.put(None)
        self.thread.join()
        
    def _speak_text(self, text, voice="en_UK/apope_low"):
        self.speaking = True
        self.stop_flag.clear()
        #save output to a temp wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            wav_path = fp.name
        #call mimic3 to synthesize the audio, write output to the file
        with open(wav_path, "wb") as f:
            self.current_process = subprocess.Popen(
                ["mimic3", "--voice", voice, text],
                stdout=f
            )
            self.current_process.wait()
        if self.stop_flag.is_set():
            #interrupted during generation
            if os.path.exists(wav_path):
                os.remove(wav_path)
            self.speaking = False
            return
        
        #load the wav file using pydub
        audio = AudioSegment.from_file(wav_path)
        #apply pitch shift if needed
        if self.pitch_semitones:
            new_sr = int(audio.frame_rate * (2.0 ** (self.pitch_semitones / 12.0)))
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sr})
            audio = audio.set_frame_rate(44100)

        #apply speed change if needed
        if self.speed_factor and self.speed_factor != 1.0:
            audio = audio._spawn(audio.raw_data, overrides={
                'frame_rate': int(audio.frame_rate * self.speed_factor)
            }).set_frame_rate(44100)

        self.current_playback = play_sound(audio)
        
        print("Speaking started") #debugging line once comfortable comment out
        while self.current_playback.is_playing():
            if self.stop_flag.is_set():
                print("Speaking interrupted by stop()") #debugging line once comfortable comment out
                self.current_playback.stop()
                self.clear_queue()
                break
        self.speaking = False
        os.remove(wav_path)

    def speak(self, text):
        self.speech_queue.put(text)

    def stop(self):
        if self.speaking:
            self.stop_flag.set()
            #stop mimic3 subprocess if still running
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
            #also try to stop pydub playback if in progress
            if self.current_playback and self.current_playback.is_playing():
                self.current_playback.stop()

#singleton
speech_manager = SpeechManager()