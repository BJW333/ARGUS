import speech_recognition as sr
import re
from core.orchestrator import process_user_input


def generalvoiceinput(): 
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")  
        try: 
            voiceinputauido = recognizer.listen(source, timeout=4)
            inputbyvoice = recognizer.recognize_google(voiceinputauido, language='en-us')
            print("Me  --> ", inputbyvoice)
            return inputbyvoice
        except:
            print("Me  -->  ERROR")     
            return None     
        
def listen_for_wake_word():
    global is_thinking
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        #print("Listening for wake word and input...")
        while True:
            print("Listening for wake word and input...")
            audio = recognizer.listen(source)
            is_thinking = True
            try:
                maininput = recognizer.recognize_google(audio, language='en-us')
                is_thinking = False
                print("Me  --> ", maininput)
                if "argus" in maininput.lower():
                    #use regex to remove the wake word and any surrounding spaces
                    maincommand = re.sub(r'\b(argus)\b', '', maininput, flags=re.IGNORECASE).strip()
                    #remove any double spaces created by removing the wake word
                    maincommand = re.sub(r'\s+', ' ', maincommand)
                    #print("this the command:", command)
                    handle_wake_word_detected(maincommand)
            except sr.WaitTimeoutError:
                is_thinking = False
                print("No audio heard")
            except sr.UnknownValueError:
                is_thinking = False
            except sr.RequestError as e:
                is_thinking = False
                print(f"Error with the request; {e}")
                
                
def handle_wake_word_detected(maininputcommand):
    if maininputcommand:
        #print("Processing input") debugging
        process_user_input(maininputcommand)
    #process_user_input(maininputcommand)
    
    listen_for_wake_word()