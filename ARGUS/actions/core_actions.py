import os
import time
import platform
import re
import requests
from bs4 import BeautifulSoup
from pytickersymbols import PyTickerSymbols
import subprocess

from utilsfunctions import calculate, gatherinfofromknowledgebase, identifynetworkconnect, start_timer, coin_flip, cocktail
from config import script_dir, MASTER
from speech.speak import speak
from datafunc.data_store import DataStore

def action_time():
    current_time = time.strftime("%I:%M %p")
    return f"Bot: The current time is {current_time}"

def volumecontrol(command):
    """
    Adjust system volume based on a command string.
    Acceptable examples:
      - "volume up 20"        (increase volume by 20 on macOS; on Linux/Windows, sets volume to 20)
      - "volume down 20"      (decrease volume by 20 on macOS; on Linux/Windows, sets volume to 80)
      - "set volume 70" or "volume 70"  (set volume absolutely to 70)
    """
    sys_os = platform.system()
    cmd = command.lower()

    #match a relative command: "volume up 20" or "volume down 20"
    rel_match = re.search(r"volume\s+(up|down)\s+(\d+)", cmd)
    if rel_match:
        direction = rel_match.group(1)
        adjustment = int(rel_match.group(2))
        if sys_os == "Darwin":
            try:
                current = int(os.popen("osascript -e 'output volume of (get volume settings)'").read().strip())
            except Exception:
                current = 50  #fallback value
            new_volume = current + adjustment if direction == "up" else current - adjustment
        else:
            #For Linux/Windows treat relative commands as absolute
            #"volume up 20" sets volume to 20, and "volume down 20" sets volume to 80
            new_volume = adjustment if direction == "up" else 100 - adjustment

    else:
        #if not a relative command try to match an absolute command: "set volume 70" or "volume 70"
        abs_match = re.search(r"(?:set\s+)?volume\s+(\d+)", cmd)
        if abs_match:
            new_volume = int(abs_match.group(1))
        else:
            print("No valid volume command found.")
            return

    #clamp new_volume to the 0-100 range.
    new_volume = max(0, min(new_volume, 100))

    #apply the new volume based on the operating system.
    if sys_os == "Darwin":
        os.system(f'osascript -e "set volume output volume {new_volume}"')
        print(f"Volume adjusted to {new_volume} (macOS).")
    elif sys_os == "Linux":
        os.system(f"amixer -D pulse sset Master {new_volume}%")
        print(f"Volume set to {new_volume}% on Linux.")
    elif sys_os == "Windows":
        vol_val = int(new_volume * 65535 / 100)
        os.system(f"nircmd.exe setsysvolume {vol_val}")
        print(f"Volume set to {new_volume} on Windows.")
        
        
def get_the_news():
    url = 'https://www.bbc.com/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    headlines_links = soup.find('body').find_all('a')
    unwanted_bs = set(['BBC World News TV', 'BBC World Service Radio',
                       'News daily newsletter', 'Mobile app', 'Get in touch'])

    headlines_by_category = {}

    for link in headlines_links:
        headline_text = link.text.strip()
        if headline_text and headline_text not in unwanted_bs:
            href = link.get('href')
            if href and '/' in href:
                url_parts = href.split('/')
                if len(url_parts) > 2:
                    category = url_parts[2].replace('-', ' ').title()
                    headlines_by_category.setdefault(category, []).append(headline_text)
    
    return headlines_by_category

def get_ticker(company_name):
    stock_data = PyTickerSymbols()
    all_stocks = stock_data.get_all_stocks()
    matches = [stock for stock in all_stocks if company_name.lower() in stock['name'].lower()]
    if matches:
        return matches[0]['symbol']
    else:
        return None
        
def takenotes():
    notes_dir = script_dir / 'NOTES'  #ensure the directory is set
    notes_dir.mkdir(parents=True, exist_ok=True)  #create directory if it doesn't exist
    filepath = notes_dir / 'notes.txt'  #default notes file
    
    speak(f"Just state what you want to document in your notes, {MASTER}.")
    notesinput = generalvoiceinput()
    
    if not notesinput or notesinput.strip() == "":
        print("No input received.")
        return
    
    #define the phrase that indicates the end of note-taking
    end_phrase = "end note taking"
    
    #truncate the input if "end note taking" is found
    if end_phrase in notesinput:
        notesinput = notesinput.split(end_phrase, 1)[0].strip()
    
    #define replacements for spoken formatting
    replacements = {
        "new line": "\n",
        "period": ".",
        "question mark": "?",
        "exclamation point": "!",
        "slash": "/",
        "comma": ","
    }
    
    #perform replacements
    for key, value in replacements.items():
        notesinput = notesinput.replace(key, value)

    #determine the file where notes will be written
    if filepath.exists():
        #if the default file exists, find the next available numbered file
        newfilenum = 1
        while (notes_dir / f"notes{newfilenum}.txt").exists():
            newfilenum += 1
        filepath = notes_dir / f"notes{newfilenum}.txt"
    
    #write the notes to the file
    try:
        with open(filepath, "w") as filenotes:
            filenotes.write(notesinput)
        print(f"Notes saved at: {filepath}")
        speak("Note saved")
    except Exception as e:
        print(f"An error occurred while saving notes: {e}")

def get_city_coordinates(city_name):
    api_key = 'd206144f1b4440f7b00206c8667c8ba7'
    endpoint = 'https://api.opencagedata.com/geocode/v1/json'
    params = {
        'q': city_name,
        'key': api_key,
        'limit': 1
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    if data['results']:
        latitude = data['results'][0]['geometry']['lat']
        longitude = data['results'][0]['geometry']['lng']
        return latitude, longitude
    else:
        return None, None


def open_app(app_name):
    try:
        subprocess.Popen(["open", "-a", app_name])
        speak(f"Opening {app_name}...")
    except FileNotFoundError:
        speak(f"Sorry, I could not find the {app_name} application.")

def close_application(app_name):
    script = f'tell application "{app_name}" to quit'
    try:
        subprocess.run(['osascript', '-e', script])
        speak(f"Closing {app_name}...")
    except Exception as e:
        print(f'Error closing {app_name}: {e}')