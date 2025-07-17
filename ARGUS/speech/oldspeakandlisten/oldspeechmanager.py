#speechmanager.py
import threading
import queue
import tempfile
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_sound

class SpeechManager:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.current_playback = None
        self.speaking = False
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.thread.start()

    def _speak_loop(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self._speak_text(text)
            
    def clear_queue(self): # test function for speed purposes may cause issues in the future if so delete
        with self.speech_queue.mutex: # also clears the queue
            self.speech_queue.queue.clear() # also clears the queue
            
    def _speak_text(self, text, lang='en', tld='co.uk', pitch_semitones=-11.4, speed_factor=3.175):
        self.speaking = True
        self.stop_flag.clear()
        tts = gTTS(text=text, lang=lang, tld=tld)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio = AudioSegment.from_file(fp.name)

            # pitch shift
            new_sr = int(audio.frame_rate * (3.7 ** (pitch_semitones / 11.5)))
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sr})

            # speed
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': int(audio.frame_rate * speed_factor)})
            audio = audio.set_frame_rate(44100)

            self.current_playback = play_sound(audio)
            print("📢 Speaking started")
            while self.current_playback.is_playing():
                if self.stop_flag.is_set():
                    print("🛑 Speaking interrupted by stop()")
                    self.current_playback.stop()
                    speech_manager.clear_queue() # test line to clear the queue
                    break
        self.speaking = False
        os.remove(fp.name)

    def speak(self, text):
        self.speech_queue.put(text)

    def stop(self):
        if self.speaking:
            self.stop_flag.set()

#singleton
speech_manager = SpeechManager()