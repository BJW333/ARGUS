import os
import webbrowser
from keybert import KeyBERT
from PyQt5.QtWidgets import QApplication
from datetime import datetime
import random
import pickle
import pyjokes


from config import script_dir, MASTER, password_checker_path, hide_me_script_path, spider_crawler_path, wake_word_sound_path

from metrics.logging import log_metrics

from utilsfunctions import calculate, gatherinfofromknowledgebase, identifynetworkconnect, start_timer, coin_flip, cocktail
from actions.core_actions import get_ticker, get_city_coordinates, open_app, close_application, takenotes, volumecontrol, get_the_news, action_time

from argus_obj_person_recog import objectrecognitionrun 
from arguscode_model import argus_code_generation
#from gui.animated_canvas import audio_setup
#from nlp.chatbot import Chatbot 
#from nlp.reward import DynamicRewardSystem
from datafunc.data_store import json_to_text  # , DataStore
from datafunc import data_analysis

from core.feedback import train_with_feedback, collect_human_feedback
from nlp.intent import intentrecognition

from speech.speak import speak
#from speech.listen import generalvoiceinput
from nlp.chatbot_init import initialize_chatbot_components

from speech.speechmanager import speech_manager


chatbot, reward_system, data_store, conversation_history, sampled_dataset, train_dataset, vocab_size, start_token, end_token = initialize_chatbot_components()


def handle_multi_intent_user_input(user_input):
    """
    Splits user input into separate intent clauses using NLP.
    - If the input contains multiple commands (e.g. "open Spotify and tell me the time"),
        returns a list of clauses (["open Spotify", "tell me the time"]).
    - If input contains only one command, returns a single-element list with the full input.
        returns a single of clause aka the raw user_input (["open Spotify"]).
    For each detected (intent, clause) pair, calls process_user_input() individually.
    Ensures all user requests are handled, whether input contains one or several intents.
    """
    intent_recog = intentrecognition()
    intent_results = intent_recog.unified_intent_pipeline(user_input)
    for intent, clause in intent_results:
        process_user_input(clause, intent)
        

def process_user_input(user_input, intent):
    """
    Executes the action corresponding to the detected intent.
    Receives the intent clause as user_input 
    (if multi-intent, this is just the clause; if single-intent, it's the raw user_input).
    Extracts entities and runs the correct action or response logic.
    """
    from speech.listen import generalvoiceinput
    #audio_setup() 
    intent_recog = intentrecognition()
    
    #extract entities and recognize intent
    entities = intent_recog.extract_entities(user_input)
    #intent = intent_recog.unified_intent_pipeline(user_input) #used to be intent_recog.intentunderstand(user_input) orginally
    #intent now gets passed through multi intent function
        
    if intent == "exit":
        chatbot.save_model()
        print("Vocabulary Size:", vocab_size)
        try:
            speech_manager.shutdown_speech() # this is meant to be used to shutdown the speech manager there is a func that is used within the speech manager
            print("Speech manager shut down successfully.") #debugging print
        except Exception as e:
            print(f"Error shutting down speech manager: {e}")
        print("Exiting Argus") #debugging print
        QApplication.quit()
    
    elif intent == "volume_control":
        volumecontrol(user_input)
        
    elif intent == "searchsomething":
    #if the intent is to search Wikipedia and the entity detected is a PERSON
        if 'PERSON' in entities.values():
            # Perform a search for information about the person
            wiki = gatherinfofromknowledgebase(entities)
            if wiki and "No results found." not in wiki:
                responsewhoiswhatis = f"Bot: {wiki}"
            else:
                responsewhoiswhatis = "Bot: I couldn't find any information regarding that person on Wikipedia."
                speak(responsewhoiswhatis)
                #fallback to Google Search
                responsewhoiswhatis = ("I'll look it up on Google for you.") #responsewhoiswhatis may go here orginal is speak(then the message)
                google_search_url = f"https://www.google.com/search?q={entities}"
                webbrowser.open(google_search_url)
                return

        else:
            # Extract keywords from the user input
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(user_input, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)

            # Use the most probable keyword for the Wikipedia search
            if keywords:
                most_probable_keyword = keywords[0][0]  # Get the first (most probable) keyword
                print("Most Probable Key Term:", most_probable_keyword)
            else:
                most_probable_keyword = user_input  # Use the whole input as a fallback keyword if none found
                print("No keywords found, using full query as keyword.")

            # Perform the Wikipedia search
            wiki = gatherinfofromknowledgebase(most_probable_keyword)
            
            # If information is found on Wikipedia
            if wiki and "No results found." not in wiki:
                responsewhoiswhatis = f"Bot: {wiki}"
            else:
                responsewhoiswhatis = f"Bot: I couldn't find any information about {most_probable_keyword} on Wikipedia."
                speak(responsewhoiswhatis)
                # Fallback to Google Search
                responsewhoiswhatis = ("I'll look it up on Google for you.") #responsewhoiswhatis may go here orginal is speak(then the message)
                google_search_url = f"https://www.google.com/search?q={most_probable_keyword}"
                webbrowser.open(google_search_url)
                return
        
        # Output the response from Wikipedia
        response = responsewhoiswhatis
        print(response)
        speak(response)        
    
    
    elif intent == "codemodel":
        speak("What would you like me to create for you.")    
        print("What would you like me to create for you.")    
        prompt = generalvoiceinput()
        if prompt:
            response = argus_code_generation(prompt, max_length=1024)
            if response.strip():  #ensures a valid response was generated
                speak(f"{MASTER}, I have printed the code on your screen. Let me know if you need anything else.")
                print("\n" + response)
            else:
                speak("I generated an empty response. Would you like me to try again with a different prompt?")
        else:
            speak("I didn't catch that. Can you please repeat your request?")
            
    
    elif intent == "objrecog":
        results = objectrecognitionrun()
        print(results) #debugging print
        
        resultsformatted = [r for r in results if r.lower() != 'person']        
        #filters for readablity and output are the below two lines
        peoples = [r.replace('_', ' ').title() for r in resultsformatted if "_" in r]
        objects = [r.replace('_', ' ').title() for r in resultsformatted if "_" not in r]

        resultparts = []
        if peoples:
            if len(peoples) == 1:
                #print(f"I see {peoples[0]}") #debugging print
                resultparts.append(f"I see {peoples[0]}")
            else:
                people_list = "I see " + ", ".join(peoples[:-1]) + f" and {peoples[-1]}."      
                #print("Final output of people_list:", people_list) #debugging print
                resultparts.append(people_list)
                
        if objects:
            if len(objects) == 1:
                #print(f"I see a {objects[0]}") #debugging print
                resultparts.append(f"I see a {objects[0]}")
            else:
                objects_list = "I see a " + ", ".join(objects[:-1]) + f" and a {objects[-1]}."      
                #print("Final output of objects_list:", objects_list) #debugging print
                resultparts.append(objects_list)

        final_sentence = " Also ".join(resultparts)
        print(f"This is the output of final sentence: {final_sentence}") #debugging print
        speak(final_sentence)
        
        
    elif intent == "connectionwithinternet":
        connectionstatus = identifynetworkconnect()
        #print(connectionstatus)
        if connectionstatus:
            speak("Internet is connected")
        else: 
            speak("Internet is not connected")
            
            
    elif intent == "stock_data":
        if 'ORG' in entities.values():
        #extract the organization (company name)
            for ent, label in entities.items():
                if label == 'ORG':
                    company_name = ent
                    break
            print(f"Extracted company name: {company_name}")
        else:
            #if no organization entity is found ask the user for the company name
            print("Please tell me the company name and or stock listing you're interested in.")
            speak("Please tell me the company name and or stock listing you're interested in.")
            company_name = generalvoiceinput()

        #retrieve the stock ticker using the company name
        stock_ticker = get_ticker(company_name)
        if stock_ticker:
            # Initialize the stock data stream
            stock_stream = data_analysis.StockDataStream(symbol=stock_ticker.upper(), api_key='MJX8BVSA9W1WOEH4')
            data = stock_stream.fetch_data()
            if not data.empty:
                stock_stream.analyze_data(data)
                # Convert analysis to speech
                latest_price = data['Close'].iloc[-1]
                speak(f"The latest price of {stock_ticker.upper()} is ${latest_price:.2f}")
                
                #visualize data among the stream data
                #stock_stream.visualize_data(data)
            else:
                speak(f"Sorry, I couldn't fetch data for {stock_ticker.upper()}.")
        else:
            speak("I didn't catch the stock symbol. Please try again.")

    elif intent == "weather_data":
        # Ask the user for the city
        speak("Please tell me the city you're interested in.")
        city = generalvoiceinput()
        if city:
            # You might need to get latitude and longitude for the city
            # For simplicity, let's assume we have a function to get them
            latitude, longitude = get_city_coordinates(city)
            weather_stream = data_analysis.WeatherDataStream(city=city, latitude=latitude, longitude=longitude)
            data = weather_stream.fetch_data()
            if data:
                weather_stream.analyze_data(data)
                # Convert analysis to speech
                temp_celsius = data['Temperature']
                temp_fahrenheit = (temp_celsius * 9/5) + 32
                speak(f"The current temperature in {city} is {temp_fahrenheit:.1f} degrees Fahrenheit.")
            else:
                speak(f"Sorry, I couldn't fetch weather data for {city}.")
        else:
            speak("I didn't catch the city name. Please try again.")


    elif intent == "flight_data":
        # Ask the user for the flight number
        speak("Please tell me the flight number.")
        flight_number = generalvoiceinput()
        if flight_number:
            flight_stream = data_analysis.FlightDataStream(flight_number=flight_number.upper(), api_key='6087a4f837f7de0c30af184c6e886a9b')
            data = flight_stream.fetch_data()
            if data:
                flight_stream.analyze_data(data)
                # Convert analysis to speech
                status = data['flight_status']
                speak(f"The status of flight {flight_number.upper()} is {status}.")
            else:
                speak(f"Sorry, I couldn't fetch data for flight {flight_number.upper()}.")
        else:
            speak("I didn't catch the flight number. Please try again.")

    
    elif intent == "crypto_data":
        # Ask the user for the cryptocurrency
        speak("Please tell me the cryptocurrency you're interested in.")
        crypto_name = generalvoiceinput()
        if crypto_name:
            # Normalize the name (e.g., "Bitcoin" -> "bitcoin")
            crypto_id = crypto_name.lower()
            crypto_stream = data_analysis.CryptoDataStream(coin_id=crypto_id, vs_currency='usd')
            data = crypto_stream.fetch_data()
            if data:
                crypto_stream.analyze_data(data)
                # Convert analysis to speech
                current_price = data['current_price']
                speak(f"The current price of {crypto_name} is ${current_price:.2f}.")
            else:
                speak(f"Sorry, I couldn't fetch data for {crypto_name}.")
        else:
            speak("I didn't catch the cryptocurrency name. Please try again.")
    
    elif intent == "timer":
        start_timer(user_input)   
    
    elif intent == "coinflip":
        coinfliped = coin_flip()
        speak(coinfliped)
         
    elif intent == "cocktail_intent":
        speak("Provide me with the name of the specific drink you want to make")
        print("Provide me with the name of the specific drink you want to make")
        cocktailname = generalvoiceinput()
        if cocktailname:
            result = cocktail(cocktailname)
            speak("I will provide the information to make the drink it will be displayed on the screen") #this is used temporaly until I can see what the output looks like then this will be adjusted
        else:
            speak("I didn't catch the name of that drink. Please try again.")
        #print("I will provide the information to make the drink it will be displayed on the screen") #this is used temporaly until I can see what the output looks like then this will be adjusted
        #print(result)
            
    elif intent == "open":
        app_name = user_input.split("open ")[1]
        print(f"Opening app: {app_name}")
        open_app(app_name)
        
    elif intent == "close":
        app_name = user_input.split("close ")[1]
        print(f"Closing app: {app_name}")
        close_application(app_name)   
         
    elif intent == "news":
        print("Bot: Here's what's happening in the news:")
        speak("Bot: Here's what's happening in the news:")
        news_today_recent = get_the_news()
        for category, headlines in news_today_recent.items():
            print(f"\n{category} News:")
            for i, headline in enumerate(headlines, 1):
                print(f"{i}. {headline}")
        
    elif intent == "time":
        timeofday = action_time()    
        print(timeofday)
        speak(timeofday)
        
    elif "are you there" in user_input.lower():
        speak(f"I'm here {MASTER}")
    
    elif "hide me" in user_input.lower():
        os.system(f"bash {hide_me_script_path}")
        speak("Hiding your IP with Tor")
    
    #elif user_input.lower() in ["i need to manually adjust some things with the model", "adjust the model", "manually mantain the model", "I need to adjust the model"]:  
    elif "i need to adjust the model" in user_input.lower():
        print('Please say the command/word of whatever needs to be done in regards to the model')
        speak('Please say the command/word of whatever needs to be done in regards to the model')
        print('1. feedback')
        print('2. train')
        print('3. save') 
        print('4. save json to text')           
        manualinput = generalvoiceinput()
    #usage of training with feedback and historical data
        if manualinput == None:
            speak("There was a error adjusting the model try again")
            
        #used be if statement below
        elif manualinput.lower() == 'feedback':
            feedback_data = collect_human_feedback(conversation_history)
            epochs3 = 10 #orginal value 10
            train_with_feedback(chatbot, sampled_dataset, feedback_data, epochs3)
            
            chatbot.save_model() #save model after training
            
        elif manualinput.lower() == 'train':
            epochtotrain = 5
            #train the model
            chatbot.modeltrain(train_dataset, epochtotrain)  
            
            chatbot.save_model() #save model after training
            
        elif manualinput.lower() == 'save':
            chatbot.save_model()
            with open(script_dir / 'tokenizer.pickle', 'wb') as handle:
                pickle.dump(chatbot.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif manualinput.lower() == 'save json to text':
        #This gets the input and response from json file and puts into a txt so its possible to go into input txt and output txt
            conversation_jsontotxt = data_store.load_data()
            text_data = json_to_text(conversation_jsontotxt)
            file_pathforjsontotxt = (script_dir / 'data/conversation_datajsontotxt.txt')

            with open(file_pathforjsontotxt, 'w') as file:
                file.write(text_data)
        #below else is new
        else:
            manualinput == None
            speak("There was a error adjusting the model try again")
            
    elif "tell me a joke" in user_input.lower():
            speak(pyjokes.get_joke())     
                
    elif "movie" in user_input.lower():
            goodmovies = ["Star Wars", "Jurassic Park", "Clear and Present Danger", "War Dogs", "Wolf of Wall Street", "The Big Short", "Trading Places", "The Gentlemen", "Ferris Bueller's Day Off", "Goodfellas", "Lord of War", "Borat", "Marvel movies", "The Hurt Locker", "Hustle", "Forrest Gump", "Darkest Hour", "Coming to America", "Warren Miller movies", "The Dictator"]
            moviechoice = random.choice(goodmovies)
            speak(f"A good movie you could watch is {moviechoice}, {MASTER}")
            
    elif "what are your skills" in user_input.lower():
        skillsforuse = (
            "-Hi, I am Argus. I can perform various tasks, including:\n"
            "- **General Chat**: I can have conversations and improve over time.\n"
            "- **Search the Web**: I can look up information on Google and Wikipedia.\n"
            "- **Open & Close Apps**: I can open and close apps on your computer.\n"
            "- **Check Time & Date**: I can tell you the current time and date.\n"
            "- **Stock & Crypto Data**: I can check stock prices and cryptocurrency trends.\n"
            "- **Weather Updates**: I can give you weather reports.\n"
            "- **Flight Tracking**: I can check flight statuses.\n"
            "- **News Updates**: I can tell you the latest headlines.\n"
            "- **Math**: I can do simple calculations.\n"
            "- **Custom Tools**: I can run custom tools like password checker, spider crawler, find peoples info, hide me and more"
            "- **Take Notes**: I can write and save notes for you.\n"
            "- **Find Cocktail Recipes**: I can look up drink recipes.\n"
            "- **Jokes & Fun**: I can tell jokes and suggest movies.\n"
            "- **Code Generation**: I can write and improve code based on your request.\n"
            "- **Improve Over Time**: I learn from interactions to get better.\n"
            "\nI'm always improving and adding new features!"
        )
        speak(skillsforuse)
  
    elif any(op in user_input for op in ("+", "-", "*", "/")):
        result = calculate(user_input)
        print(result)
        speak(result)
        
    elif "notes" in user_input.lower():
        takenotes()
        speak("Notes have been taken and saved")
        
    elif "spider crawler" in user_input.lower():
        print("Starting up SpiderCrawler")
        speak("Starting up SpiderCrawler")
        os.system(f"python3.10 {spider_crawler_path}")
        
    elif user_input.lower() in ["password checker", "can you check if my password has been breached", "I think my password is comprimised"]:
        print("lets check if your password is compromised")
        speak("lets check if your password is compromised")
        os.system(f"python3.10 {password_checker_path}")


    else:
        start_time = datetime.now()
        
        candidates = chatbot.generate_seq2seqresponse(
            input_sentence=user_input,
            num_candidates=5,  # Adjust the number of candidates as needed was 5 num canidates
            temperature=0.6
        )  # temp was 0.5
        
        best_candidate, best_score = chatbot.ga_rerank_candidates(
            user_input=user_input,
            candidates=candidates,
            pop_size=10,
            generations=3, #oringal value was 3 generations
            crossover_rate=0.5,
            mutation_rate=0.3
        )
        
        #print("This is the best score:", best_score)
        
        #response = best_candidate
        if isinstance(best_candidate, str):  # If it's text, tokenize first
            best_candidate_tokens = chatbot.tokenizer.encode(best_candidate)
        else:
            best_candidate_tokens = best_candidate  # Assume it's already tokenized

        response = chatbot.postprocess_sentence(best_candidate_tokens)

        confidence = chatbot.calculate_confidencevalue_response(user_input, best_candidate, best_score, candidates)

        reward = reward_system.evaluate_response(user_input, response)
        print("Reward before normailzation:", reward)
        
        normalized_reward = (reward + 30) / 60  # Now ranges from 0 (bad) to 1 (good)
        
        #normalized_reward = reward / 60
        #In that case, a reward of 0 would yield 0, and a reward of 60 would yield 1.

        
        print("Normalized_reward value range(0-1): ", normalized_reward)
        
        """
        Our reward system ranges form -30 to 30 and maybe it be worth it to range our normalization from -1 to 1 ended up just rangeing from 0 to 1
        
        
        trying a new way to normalize the reward into a value between 0 and 1 
        
        #print("reward before normalization", reward)
        normalized_reward = (reward + 30) / 60  # Now ranges from 0 (bad) to 1 (good)
        #print("reward after normalization should return a value in the range of 0(bad) to 1(good)", normalized_reward)
        
        
        #the below lines have a gap there is a inbetween with the good or bad decison making off the reward threshold
        if normalized_reward >= 0.5:   # Response is "good"
            responseisgoodorbad = True  
        elif normalized_reward <= 0.2:  # Response is "bad"
            responseisgoodorbad = False
        else:
            responseisgoodorbad = None or responseisgoodorbad = False #false prompts human feedback decide wether that is good or bad thing
            
        
        #now with the below lines there is no gap decide whats better to be used
        if normalized_reward >= 0.5:
            responseisgoodorbad = True  # Good
        else:
            responseisgoodorbad = False # Bad
            
        # Dynamic confidence threshold (adjusts based on reward)
        confidence_threshold = max(0.4, min(0.8, 1 - (reward / 30)))  
        
        If the reward is high, require higher confidence (0.8).
        If the reward is low, allow lower confidence (0.4) but still flag it.

        
        debugging information
        print(f"Reward: {reward}, Normalized: {normalized_reward}, Confidence: {confidence}")
        """
        
        
        
        
        #print("this is new reward eval resp:", reward)
        response_printed = False
        
        #confidence_threshold = min(0.8, max(0.4, normalized_reward)) #old line right here new line below that is nondyanmic
        confidence_threshold = 0.5 #old 0.6 was the value
        
        print("Confidence_threshold value:", confidence_threshold)
        
        corrected_response = None
        responseisgoodorbad = False
        
        if normalized_reward >= 0.6: #old value was 0.5 due to range being (0 to 1) for the normalized reward
            responseisgoodorbad = True  # Good
            #print("True")  
        else:
            responseisgoodorbad = False # Bad
            #print("False")
            
            
        print("Confidence value fron confidence funciton:", confidence)    
        print("Value of True/False of responseisgoodorbad:", responseisgoodorbad)
        
        #check if the response's confidence is below the threshold and if response is bad
        if confidence < confidence_threshold or not responseisgoodorbad:
            print("\nBot (Uncertain):", response)
            
            #new line below testing see what it does
            speak(f"Bot: {response}")

            speak("This response is flagged for human review correct the response.")
            print("This response is flagged for human review correct the response.")
            corrected_response = generalvoiceinput()
                    
            if corrected_response:
                response = corrected_response  #use the corrected response for logging and feedback
                flagged_for_retraining = True  #flag this interaction for retraining
            else:
                flagged_for_retraining = False
            response_printed = True    
        else:
            flagged_for_retraining = False
            print("Bot:", response)
            speak(f"Bot: {response}")
            response_printed = True
        
        
        #is it good to double evalualte the changed responses unsure as if thats the correct approach
        #reward = reward_system.evaluate_response(user_input, response) 
        
                
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        #log information
        log_metrics(user_input, response, response_time, reward)
                    
        if not response_printed:
            print("Bot:", response.replace(start_token, '').replace(end_token, ''))
            speak("Bot:", response.replace(start_token, '').replace(end_token, ''))
        
        
        print("\nReward for this response:", reward)
        #print("Total reward:", reward_system.get_total_reward())   
        print("Response time:", response_time) #amount of time it takes for bot to respond
        
        
        conversation_history.append((user_input, response))
        
        conversation_data = {
            'user_input': user_input,
            'bot_response': response,
            'reward': reward,  #reward mechanism
            'flagged_for_retraining': flagged_for_retraining  #flag for retraining based on checking system
        }
        data_store.save_data(conversation_data)  

        
        
        
        
        