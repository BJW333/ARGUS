import re
import os
import time
import json
import random
import wikipedia
import requests
import threading
import subprocess
import platform
from playsound import playsound
from bs4 import BeautifulSoup
from pytickersymbols import PyTickerSymbols
from config_metrics.main_config import script_dir, MASTER
from speech.speak import speak
from datetime import datetime, timedelta
from core.input_bus import print_to_gui #, debug_bus_id
from speech.listen import generalvoiceinput

def calculate(command):
    command = command.replace("plus", "+")
    command = command.replace("add", "+")
    command = command.replace("minus", "-")
    command = command.replace("subtract", "-")
    command = command.replace("multiply", "*")
    command = command.replace("times", "*")
    command = command.replace("divide", "/")
    command = command.replace("divided", "/")
    command = command.replace("multiplied", "*")  
    command = command.replace("over", "/")
    command = command.replace("mod", "%")
    
    tokens = re.findall(r'\d+|[+*/-]', command)

    expression = ' '.join(tokens)

    try:
        result = eval(expression)
    except ZeroDivisionError:
        return "numbers can't be divided by 0"
    except Exception as e:
        return str(e)

    return f"Bot: The answer to your question is {result}"


def identifynetworkconnect():
    url = ("https://www.google.com/")
    timeout = 10
    
    try:
        request = requests.get(url, timeout=timeout) 
        
        #print("Internet is on")
        internet = True
        internet = str(internet)
        return internet
    except (requests.ConnectionError,
        requests.Timeout) as exception:
        
        #print("Internet is off")
        internet = False
        internet = str(internet)
        return internet
  
  
def gatherinfofromknowledgebase(query):
    if query:
        print("Finding info related to that topic...")
        search_results = wikipedia.search(query, results=1)
        if search_results:
            try:
                summary = wikipedia.summary(search_results[0], sentences=5)
                #print(summary)
                return summary
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Disambiguation error: {e.options}")
                return "No results found."
            except wikipedia.exceptions.PageError:
                print("Page not found.")
                return "No results found."
            except Exception as e:
                print(f"An error occurred: {e}")
                return "No results found."
        else:
            print("No results found.")
            return "No results found."
    
        
def timer(userinput):
    soundfile = script_dir / 'audiofiles/timesup.mp3'
    if userinput:
        hours, mins, seconds = 0, 0, 0
        matches = re.findall(r'(\d+)\s*(hour|minute|second)', userinput)
        for match in matches:
            numoftime = int(match[0])
            unitoftime = match[1]

            if "hour" in unitoftime:
                hours += numoftime
            elif "minute" in unitoftime:
                mins += numoftime
            elif "second" in unitoftime:
                seconds += numoftime
        
        totalseconds = hours * 3600 + mins * 60 + seconds

        endtimertime = datetime.datetime.now() + datetime.timedelta(seconds=totalseconds)
        
        print_to_gui(f"Timer set for {str(datetime.timedelta(seconds=totalseconds))}")

        while totalseconds > 0:
            currenttime = datetime.datetime.now()
            totalseconds = int((endtimertime - currenttime).total_seconds())

            #print(str(datetime.timedelta(seconds=totalseconds)), end="\r")
            
            if totalseconds > 0:
                threading.Event().wait(1)
                
        print_to_gui("\nThe timer is up")
        playsound(soundfile)
   
        
def start_timer(userinput):
    timer_thread = threading.Thread(target=timer, args=(userinput,))
    timer_thread.start()


def coin_flip():
    options = ('Heads', 'Tails')
    rand_value = options[random.randint(0, 1)]
    #print(rand_value) #debugging line
    return rand_value


def cocktail(specificcocktailname):
    #speak("Please say a specifc drink name that you would like to know how to make.")
    #print("Please say a specifc drink name that you would like to know how to make.")
    cocktail = specificcocktailname
    cocktail.strip()
    #print(cocktail) #debugging make sure the code works
    url = 'https://www.thecocktaildb.com/api/json/v1/1/search.php?s=' + cocktail
    r = requests.get(url)
    json_data = json.loads(r.content)
    try:
        # collects the neccasary data from the JSON and saves in 
        cocktail_name = json_data['drinks'][0]['strDrink']
        ingredients_str = 'Ingredients: \n'
        i=1
        temp = json_data['drinks'][0]['strIngredient' + str(i)]
        while True:
            temp = json_data['drinks'][0]['strIngredient' + str(i)]
            if not temp:
                break
            ingredients_str += temp + '\n'
            i+=1
        # prints all the info in the command prompt in a pretty format
        seperator = '-----------------'
        print_to_gui(seperator)
        print_to_gui(cocktail_name)
        print_to_gui(seperator)
        print_to_gui(ingredients_str + seperator)
        print_to_gui(json_data['drinks'][0]['strInstructions'])
        print_to_gui(seperator)
        speak("The information to make the drink is displayed on the screen")
        #speak("The information to make the drink is displayed on the screen") #this is used temporaly until I can see what the output looks like then this will be adjusted
    except:
        #wrong user input 
        print_to_gui("Drink not found. Please try again.")
        speak("Drink not found. Please try again.")


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
            print_to_gui("No valid volume command found.")
            return

    #clamp new_volume to the 0-100 range.
    new_volume = max(0, min(new_volume, 100))

    #apply the new volume based on the operating system.
    if sys_os == "Darwin":
        os.system(f'osascript -e "set volume output volume {new_volume}"')
        print_to_gui(f"Volume adjusted to {new_volume} (macOS).")
    elif sys_os == "Linux":
        os.system(f"amixer -D pulse sset Master {new_volume}%")
        print_to_gui(f"Volume set to {new_volume}% on Linux.")
    elif sys_os == "Windows":
        vol_val = int(new_volume * 65535 / 100)
        os.system(f"nircmd.exe setsysvolume {vol_val}")
        print_to_gui(f"Volume set to {new_volume} on Windows.")
        
        
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
        print_to_gui(f"Notes saved at: {filepath}")
        speak("Note saved")
    except Exception as e:
        print(f"An error occurred while saving notes: {e}")

def get_city_coordinates(location: str) -> tuple[float, float] | tuple[None, None]:
    """Use Open-Meteo's free geocoding API."""
    if not location:
        return None, None
    try:
        resp = requests.get(
            'https://geocoding-api.open-meteo.com/v1/search',
            params={'name': location, 'count': 1},
            timeout=5
        )
        results = resp.json().get('results', [])
        if results:
            print(f"[DEBUG] Geocode found: {results[0].get('name')}, {results[0].get('admin1')}")
            return results[0]['latitude'], results[0]['longitude']
    except Exception as e:
        print(f"[DEBUG] Geocode error: {e}")
    return None, None


app_cache = {}

def find_app_path(app_name):
    if app_name in app_cache:
        return app_cache[app_name]

    system = platform.system()
    if system == "Darwin":
        search_dirs = ["/Applications", "/System/Applications"]
        for directory in search_dirs:
            for root, dirs, _ in os.walk(directory):
                for d in dirs:
                    if app_name.lower() in d.lower() and d.endswith(".app"):
                        full_path = os.path.join(root, d)
                        app_cache[app_name] = full_path
                        return full_path
                    
    elif system == "Windows":
        search_dirs = [r"C:\Program Files", r"C:\Program Files (x86)"]
        for directory in search_dirs:
            for root, _, files in os.walk(directory):
                for f in files:
                    if app_name.lower() in f.lower() and f.endswith(".exe"):
                        full_path = os.path.join(root, f)
                        app_cache[app_name] = full_path
                        return full_path
    return None


def open_app(app_name):
    system = platform.system()
    
    try:
        if system == "Darwin":  #macOS
            subprocess.Popen(["open", "-a", app_name])
        elif system == "Windows": #windows
            subprocess.Popen(["start", "", app_name], shell=True)
        elif system == "Linux": #linux
            subprocess.Popen([app_name])
        else:
            raise OSError("Unsupported OS")
        #speak the app opening message
        speak(f"Opening {app_name}...")
    except Exception:
        #attempt to search and open manually
        path = find_app_path(app_name)
        if path:
            if system == "Darwin":
                subprocess.Popen(["open", path])
            elif system == "Windows":
                subprocess.Popen([path])
            elif system == "Linux":
                subprocess.Popen([path])
            else:
                raise OSError("Unsupported OS")
            #speak the app opening message
            speak(f"Opening {app_name}...")
        else:
            speak(f"Sorry, I could not find the {app_name} application.")
    
    
def close_application(app_name):
    system = platform.system()
    
    try:
        if system == "Darwin":  #macOS
            script = f'tell application "{app_name}" to quit'
            subprocess.run(['osascript', '-e', script],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Windows":  #windows
            if not app_name.lower().endswith(".exe"):
                app_name += ".exe"
            subprocess.run(["taskkill", "/IM", app_name, "/F"], shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":  #linux
            subprocess.run(["pkill", "-f", app_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise OSError("Unsupported OS")
        #speak the app closing message
        speak(f"Closing {app_name}...")
    except Exception as e:
        speak(f'There was an error closing {app_name}: {e}')
        
        
def parse_weather_query(user_input: str, entities: dict) -> tuple[str | None, int]:
    """
    Extract location and day offset from weather query.
    Returns (location, day_offset) where day_offset is 0=today, 1=tomorrow, etc.
    """
    text = user_input.lower()
    
    #determine day offset
    day_offset = 0
    if 'tomorrow' in text or 'tmrw' in text:
        day_offset = 1
    elif 'day after tomorrow' in text:
        day_offset = 2
    elif match := re.search(r'in (\d+) days?', text):
        day_offset = int(match.group(1))
    elif 'this weekend' in text:
        today = datetime.now().weekday()
        days_until_saturday = (5 - today) % 7
        day_offset = days_until_saturday if days_until_saturday > 0 else 7
    elif any(day in text for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        today = datetime.now().weekday()
        for i, day in enumerate(days):
            if day in text:
                day_offset = (i - today) % 7
                if day_offset == 0:
                    day_offset = 7  #next weeks instance
                break
    
    #extract location from entities first
    location = None
    for ent, label in entities.items():
        if label in ('GPE', 'LOC'):  #GPE = geopolitical entity (city, state, country)
            location = ent
            break
    
    #fallback: regex patterns for "in <location>" or "for <location>"
    if not location:
        patterns = [
            r'weather (?:in|for|at) ([a-zA-Z\s]+?)(?:\s+(?:today|tomorrow|tmrw|this|next|on)|\?|$)',
            r'(?:in|for|at) ([a-zA-Z\s]+?)(?:\s+(?:today|tomorrow|tmrw)|\?|$)',
            r"what's it like in ([a-zA-Z\s]+)",
        ]
        for pattern in patterns:
            if match := re.search(pattern, text):
                location = match.group(1).strip()
                break
    
    #clean up location remove trailing time words
    if location:
        location = re.sub(r'\s+(tomorrow|tmrw|today|this weekend).*$', '', location, flags=re.IGNORECASE).strip()
    
    return location, day_offset


def format_day_name(day_offset: int) -> str:
    """Convert day offset to readable name."""
    if day_offset == 0:
        return "today"
    elif day_offset == 1:
        return "tomorrow"
    else:
        target = datetime.now() + timedelta(days=day_offset)
        return target.strftime("%A")  #e.g, "Saturday"