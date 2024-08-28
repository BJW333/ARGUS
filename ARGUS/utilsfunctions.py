import re
import wikipedia
import requests
import datetime
import threading
from playsound import playsound
import random

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
        return internet
    except (requests.ConnectionError,
        requests.Timeout) as exception:
        
        #print("Internet is off")
        internet = False
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
    soundfile = '/Users/blakeweiss/Desktop/ARGUS/audiofiles/timesup.mp3'
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
        
        print(f"Timer set for {str(datetime.timedelta(seconds=totalseconds))}")

        while totalseconds > 0:
            currenttime = datetime.datetime.now()
            totalseconds = int((endtimertime - currenttime).total_seconds())

            #print(str(datetime.timedelta(seconds=totalseconds)), end="\r")
            
            if totalseconds > 0:
                threading.Event().wait(1)
                
        print("\nThe timer is up")
        playsound(soundfile)
        
def start_timer(userinput):
    timer_thread = threading.Thread(target=timer, args=(userinput,))
    timer_thread.start()

def coin_flip():
    options = ('Heads', 'Tails')
    rand_value = options[random.randint(0, 1)]
    #print(rand_value)
    return rand_value