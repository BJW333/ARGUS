import re
import wikipedia
import requests
import datetime
import threading
from playsound import playsound
import random
from pathlib import Path
import json

script_dir = Path(__file__).parent

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
        request = requests.get(url, timeout=timeout) #requests.get(url, timeout=timeout)
        
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
    #soundfile = '/Users/blakeweiss/Desktop/ARGUS/audiofiles/timesup.mp3' #old version of the last line just incase its needed
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
        print(seperator)
        print(cocktail_name)
        print(seperator)
        print(ingredients_str + seperator)
        print(json_data['drinks'][0]['strInstructions'])
        print(seperator)
        print("The information to make the drink is displayed on the screen")
        #speak("The information to make the drink is displayed on the screen") #this is used temporaly until I can see what the output looks like then this will be adjusted
    except:
        # wrong user input
        #speak("Drink not found. Please try again.")
        print("Drink not found. Please try again.")

