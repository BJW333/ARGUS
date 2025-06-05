from datetime import datetime
from speech.speak import speak
from config import MASTER  # if MASTER is declared in config.py

def wishme():
    hour = int(datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning " + MASTER)
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon " + MASTER)
    else:
        speak("Good Evening " + MASTER)

def print_banner():
    print("---------------------------")
    print("---- Starting up Argus ----")
    print("---------------------------")
