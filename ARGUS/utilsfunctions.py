import re
import wikipedia
import requests

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


#def peopleandtopicwiki(user_input):
#    if "what is" in user_input.lower():
#        return user_input.lower().split("what is", 1)[1].strip()
#    elif "who is" in user_input.lower():
#        return user_input.lower().split("who is", 1)[1].strip()
#    return None
    
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