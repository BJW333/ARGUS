import os
import tempfile
from gtts import gTTS
from pydub import AudioSegment
from playsound import playsound

def change_pitch_and_speed(audio_path, pitch_semitones=-11.4, speed_factor=3.175):
    sound = AudioSegment.from_file(audio_path)

    #change pitch
    new_sample_rate = int(sound.frame_rate * (3.7 ** (pitch_semitones / 11.5)))
    pitch_changed_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    
    #change speed (tempo)
    speed_changed_sound = pitch_changed_sound._spawn(pitch_changed_sound.raw_data, overrides={'frame_rate': int(pitch_changed_sound.frame_rate * speed_factor)})

    #ensure the final sound has the original frame rate
    final_sound = speed_changed_sound.set_frame_rate(sound.frame_rate)

    return final_sound
    
def speak(text, lang='en', tld='co.uk', pitch_semitones=-11.4, speed_factor=3.175):
    if "Bot:" in text:
        text = text.replace("Bot:", "")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(text=text, lang=lang, tld=tld)
        tts.save(tmpfile.name)
        #adjust pitch and speed
        adjusted_sound = change_pitch_and_speed(tmpfile.name, pitch_semitones, speed_factor)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as adjusted_tmpfile:
            adjusted_sound.export(adjusted_tmpfile.name, format="mp3")
            #play the adjusted sound
            playsound(adjusted_tmpfile.name)
            
    #clean up temporary files
    os.remove(tmpfile.name)
    os.remove(adjusted_tmpfile.name)