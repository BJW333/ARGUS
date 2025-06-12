import speech_recognition as sr
import re
from core.orchestrator import handle_multi_intent_user_input #, process_user_input
from .speechmanager import speech_manager
from speech.speak import speak  

def generalvoiceinput(): 
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")  
        try: 
            voiceinputaudio = recognizer.listen(source, timeout=4)
            inputbyvoice = recognizer.recognize_google(voiceinputaudio, language='en-us')
            print("Me  --> ", inputbyvoice)
            return inputbyvoice
        except Exception:
            print("Me  -->  ERROR")     
            return None     

def listen_for_wake_word():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        while True:
            print("Listening for wake word and input...")
            try:
                audio = recognizer.listen(source)
                maininput = recognizer.recognize_google(audio, language='en-us')
                print("Me  --> ", maininput)
                if "argus" in maininput.lower():
                    #remove the wake word and extra spaces
                    maincommand = re.sub(r'\b(argus)\b', '', maininput, flags=re.IGNORECASE).strip()
                    maincommand = re.sub(r'\s+', ' ', maincommand)
                    handle_wake_word_detected(maincommand)
            except sr.WaitTimeoutError:
                print("No audio heard")
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Error with the request: {e}")

def handle_wake_word_detected(maininputcommand):
    speech_manager.stop()  # immediately stop any ongoing speech
    if maininputcommand:
        #process_user_input(maininputcommand) # old function
        handle_multi_intent_user_input(maininputcommand) #now using the new multi intent function
    #DO NOT call listen_for_wake_word()
    else:
        speak("Yes? What would you like me to do?")