import os
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from speech.speak import speak
#from speech.listen import generalvoiceinput
from config import script_dir, nlp


class IntentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(-1).item()
        return predictions
    
    
class intentrecognition():
    def __init__(self, history_file=(script_dir / 'intent_history.csv')):
        # memory to store user inputs and correct intents
        self.history_file = history_file
        self.intent_history = self.load_intent_history()
        
    def load_intent_history(self):
        # Load intent history from a CSV file
        intent_history = {}
        if os.path.exists(self.history_file):
            with open(self.history_file, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    user_input, intent = row
                    intent_history[user_input] = intent
        return intent_history

    def save_intent_history(self):
        # Save intent history to a CSV file
        with open(self.history_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for user_input, intent in self.intent_history.items():
                writer.writerow([user_input, intent])  
                  
    def extract_entities(self, user_input):
        doc = nlp(user_input)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        return entities
    
    def calculate_confidence_intent(self, user_input, intent_keywords):
        user_input_lower = user_input.lower()
        keywords_set = set(intent_keywords)  # Convert list to set for O(1) lookup
        matched_keywords = sum(1 for keyword in keywords_set if keyword in user_input_lower)
        return matched_keywords / len(keywords_set) if keywords_set else 0.0
    
    def intentunderstand(self, user_input):
        if user_input in self.intent_history:
            return self.intent_history[user_input], 1.0  # Return full confidence for previously learned intents
        
        #intent keyword lists
        searchinfo = ['find me information', 'what is', 'who is', 'search', 'tell me about', 'google', 'web', 'look up', 'find']
        timelist = ['what time', 'what time is it', 'whats the time']
        openapp = ['open']
        closeapp = ['close']
        newsget = ['news']
        exitprogram = ['exit']
        internetstatus = ['is the internet connected', 'current status of the internet connection']
        timerprogram = ['timer', 'set a timer']
        coinflip = ['flip a coin', 'coin flip', 'do a coin flip']
        stock_intents = ['stock price', 'im trying to find the stock price', 'stock', 'share price', 'stock market']
        weather_intents = ['weather', 'temperature', 'forecast']
        flight_intents = ['flight status', 'flight', 'plane']
        crypto_intents = ['cryptocurrency', 'crypto price', 'crypto', 'krypto']
        drink_cocktail = ["I want to make a cocktail", "I want to make a drink", "cocktail"]
        arguscodemodel = ["I need you to write some code for me", "code"]
        volume_control_intent = ['set volume', 'volume up', 'volume down', 'volume']
        object_person_detection_intent = ['what do you see', 'what are you seeing right now']

        #intent mapping
        intent_map = {
            "searchsomething": searchinfo,
            "time": timelist,
            "open": openapp,
            "close": closeapp,
            "news": newsget,
            "exit": exitprogram,
            "connectionwithinternet": internetstatus,
            "timer": timerprogram,
            "coinflip": coinflip,
            "stock_data": stock_intents,
            "weather_data": weather_intents,
            "flight_data": flight_intents,
            "crypto_data": crypto_intents,
            "cocktail_intent": drink_cocktail,
            "codemodel": arguscodemodel,
            "volume_control": volume_control_intent,
            "objrecog":object_person_detection_intent
        }
        
        #user_input_lower = user_input.lower()
        best_intent = "unknown"
        highest_confidence = 0.0
        
        for intent, keywords in intent_map.items():
            confidence = self.calculate_confidence_intent(user_input, keywords)
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_intent = intent
        
        #set a threshold for confidence, below which the system will request human feedback
        if highest_confidence == 0.0:
            return "ai_response", 0.0
        
        return best_intent, highest_confidence
    
    def learnfromfeedback(self, user_input, predicted_intent, correct_intent):
        # If the user provides feedback, store the correct intent
        if correct_intent != predicted_intent:
            self.intent_history[user_input] = correct_intent
            self.save_intent_history()  # Save the updated intent history
        print(f"Learning: Corrected '{predicted_intent}' to '{correct_intent}' for input: '{user_input}'")
        
    def interactiveintentrecognition(self, user_input):
        from speech.listen import generalvoiceinput
        # Predict intent and get confidence
        predicted_intent, confidence = self.intentunderstand(user_input)
       
        #maybe include if response is ai response then skip over the checked intent human feedback
        if predicted_intent == "ai_response":
            confidence = 1.0

        if confidence < 0.2:
            print(f"Confidence is low ({confidence*100:.1f}%). Predicted intent: {predicted_intent}. Is this correct? (yes/no)")
            user_feedback = generalvoiceinput()

            if user_feedback == "no":
                # Ask the user for the correct intent
                print("Please provide the correct intent:")
                correct_intent = generalvoiceinput()
                self.learnfromfeedback(user_input, predicted_intent, correct_intent)
                return correct_intent
            elif user_feedback == "yes":
                return predicted_intent
        else:
            print(f"Predicted intent: {predicted_intent} with confidence {confidence*100:.1f}%.")

        return predicted_intent