from datetime import datetime
from speech.speak import speak
from config import MASTER

def wishme():
    hour = int(datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning " + MASTER)
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon " + MASTER)
    else:
        speak("Good Evening " + MASTER)

def print_banner():
    #would like to put the wake word sound here when it starts up 
    #sound would be played here but if its inefficient and increases cpu and latency remove it 
    
    print("---------------------------")
    print("---- Starting up Argus ----")
    print("---------------------------")
    
    
