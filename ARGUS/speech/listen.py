import speech_recognition as sr
import re
import time
#from core.orchestrator import handle_multi_intent_user_input #, process_user_input
from .speechmanager import speech_manager
from speech.speak import speak  
from gui.animated_canvas import mic_suspend, mic_resume, set_external_volume
import numpy as np
from config_metrics.logging import log_debug
from core.input_bus import send as submit_user_input #, debug_bus_id
#debug_bus_id("listen")

# Keep this sample rate in sync with AnimatedCanvas (44100 by default)
RATE = 44100

def _feed_canvas_volume_from_audio(audio):
    """Send RMS from SR audio to the canvas pulse (non-blocking UI proxy)."""
    try:
        sw = int(audio.sample_width)  # bytes per sample: 1,2, or 4
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
        samples = np.frombuffer(audio.get_raw_data(), dtype=dtype).astype(np.float32)
        max_val = float(2 ** (8*sw - 1))
        rms = float(np.sqrt(np.mean(samples * samples))) / max_val
        set_external_volume(min(1.0, rms * 12.0))   # same visual boost as canvas
    except Exception:
        pass
    
    
def generalvoiceinput(): 
    recognizer = sr.Recognizer()
    mic_suspend()  # suspend the canvas mic so SR can use the device
    try:
        with sr.Microphone(sample_rate=RATE) as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            #print("Listening...")  
            log_debug("Listening...")  
            try: 
                voiceinputaudio = recognizer.listen(source, timeout=4)
                _feed_canvas_volume_from_audio(voiceinputaudio)  # Feed volume to the canvas pulse
                inputbyvoice = recognizer.recognize_google(voiceinputaudio, language='en-us')
                print("Me  --> ", inputbyvoice)
                return inputbyvoice
            except sr.WaitTimeoutError:
                log_debug("No speech detected (timeout).")
                return None
            except sr.UnknownValueError:
                log_debug("Speech not understood.")
                return None
            except sr.RequestError as e:
                log_debug(f"Speech service request error: {e}")
                return None
            except Exception:
                print("Me  -->  ERROR")     
                return None            
    finally:
        # give the mic back to the canvas 
        mic_resume() 


def listen_for_wake_word():
    print("listen fuction is running")
    recognizer = sr.Recognizer()
    
    backoff = 1.0  # exponential backoff for network hiccups

    mic_suspend() # suspend the canvas mic so SR can use the device
    try:
        with sr.Microphone(sample_rate=RATE) as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)
            while True:
                #print("Listening for wake word and input...")
                log_debug("Listening for wake word and input...")
                try:
                    audio = recognizer.listen(source)
                    _feed_canvas_volume_from_audio(audio)
                    maininput = recognizer.recognize_google(audio, language='en-us')
                    print("Me  --> ", maininput)
                    if "argus" in maininput.lower():
                        #remove the wake word and extra spaces
                        maincommand = re.sub(r'\b(argus)\b', '', maininput, flags=re.IGNORECASE).strip()
                        maincommand = re.sub(r'\s+', ' ', maincommand)
                        handle_wake_word_detected(maincommand)
                except sr.WaitTimeoutError:
                    log_debug("No audio heard")  # No audio during window; just loop
                    continue
                except sr.UnknownValueError:
                    log_debug("Could not understand audio")  # heard something but couldn't decode
                    continue                    
                except sr.RequestError as e:
                    log_debug(f"Speech service request error: {e} — retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 8.0)
                except Exception as e:
                    log_debug(f"Speech loop error: {type(e).__name__}: {e} — retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 8.0)
    finally:
        # when the loop ends let the canvas reclaim the mic 
        mic_resume()

def handle_wake_word_detected(maininputcommand):
    speech_manager.stop()  # immediately stop any ongoing speech
    if maininputcommand:
        submit_user_input(maininputcommand)  # unified path 
    #DO NOT call listen_for_wake_word()
    else:
        log_debug("Theres a error with handle_wake_word_detected function its going to its else statement")
        speak("Yes? What would you like me to do?")