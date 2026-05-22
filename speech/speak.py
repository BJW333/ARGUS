from .speechmanager import speech_manager

def speak(text):
    if "Bot:" in text:
        text = text.replace("Bot:", "").strip()
    speech_manager.speak(text)
    
    
