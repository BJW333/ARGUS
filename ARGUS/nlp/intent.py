from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import string
from huggingface_hub import snapshot_download
from config import nlp
from metrics.logging import log_debug, log_metrics
from core.input_bus import print_to_gui

intent_repo_id = "bjw333/intent_model_argus"
intent_model_path = snapshot_download(repo_id=intent_repo_id, repo_type="model")
# Load model and tokenizer
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path)

#Load label mapping 
id2label = {
    0: "searchsomething",
    1: "time",
    2: "open",
    3: "close",
    4: "news",
    5: "exit",
    6: "connectionwithinternet",
    7: "timer",
    8: "coinflip",
    9: "stock_data",
    10: "weather_data",
    11: "flight_data",
    12: "crypto_data",
    13: "cocktail_intent",
    14: "codemodel",
    15: "volume_control",
    16: "objrecog"
}


def clean_text(text):
    #lowercase and remove punctuation (except for splitting)
    return re.sub(r'[^\w\s]', '', text.lower())

#this splits the user input into clauses based on coordinating conjunctions for multi intent commands
def split_intent_commands(user_input):
    """
    splitter: handles verbs and question-words after 'and'/'then', trims punctuation.
    """
    question_starters = {"who", "what", "when", "where", "why", "how", "which"}
    doc = nlp(user_input)
    clauses = []
    start = 0
    for i, token in enumerate(doc):
        # Split at "and"/"then" if followed by a verb or imperative
        if token.text.lower() in ("and", "then") and token.dep_ == "cc":
            j = i + 1
            while j < len(doc) and doc[j].is_space:
                j += 1
            #check for: verb, aux, modal, or question-word after the split word
            if j < len(doc) and (
                doc[j].pos_ in ("VERB", "AUX")
                or doc[j].tag_ in ("VB", "VBZ", "VBP", "MD")
                or doc[j].text.lower() in question_starters
            ):
                #extract and clean
                clause = doc[start:i].text.strip().strip(string.punctuation)
                if clause:
                    clauses.append(clause)
                start = j
    #add the final clause and clean
    last = doc[start:].text.strip().strip(string.punctuation)
    if last:
        clauses.append(last)
    return clauses if clauses else [user_input.strip().strip(string.punctuation)]


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
    def __init__(self):
        self.tokenizer = intent_tokenizer
        self.model = intent_model
        
    def extract_entities(self, user_input):
        doc = nlp(user_input)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        return entities    
    
    def calculate_confidence_intent(self, user_input, intent_keywords):
        user_input_lower = user_input.lower()
        keywords_set = set(intent_keywords)  #convert list to set for O(1) lookup
        matched_keywords = sum(1 for keyword in keywords_set if keyword in user_input_lower)
        return matched_keywords / len(keywords_set) if keywords_set else 0.0
    
    def intentunderstand(self, user_input):
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
        best_intent_rule = "unknown"
        highest_confidence_rule = 0.0
        
        for intent, keywords in intent_map.items():
            confidence = self.calculate_confidence_intent(user_input, keywords)
            if confidence > highest_confidence_rule:
                highest_confidence_rule = confidence
                best_intent_rule = intent
        
        #set a threshold for confidence, below which the system will request human feedback
        if highest_confidence_rule == 0.0:
            return "ai_response", 0.0
        
        return best_intent_rule, highest_confidence_rule   
        
        
    def rulebased_intentrecognition(self, user_input):
        #predict intent and get confidence also clean text before enter rulebased intent recog so no issues hopefully
        predicted_intent, confidence = self.intentunderstand(clean_text(user_input))
       
        #maybe include if response is ai response then skip over the checked intent human feedback
        if predicted_intent == "ai_response":
            confidence = 1.0
        
        return predicted_intent, confidence
        
    
    def ml_predict_intent(self, text, threshold=0.5):
        if not isinstance(text, str):
            raise TypeError(f"Expected `text` to be str but got {type(text).__name__}")
        
        inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = intent_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probs).item()
            confidence = probs[0][predicted_class_id].item()
        
        intent = id2label[predicted_class_id]
        #return intent if confidence >= threshold else "ai_response" #only returns intent
        return (intent, confidence) if confidence >= threshold else ("ai_response", confidence) #returns both confidence and predicted_intent
    
    def unified_intent_pipeline(self, user_input):
        clauses = split_intent_commands(user_input)
        results = []
        for clause in clauses:
            cleaned = clause.strip()
            #print("Processing clause:", cleaned) #debugging line
            ml_predicted_intent, ml_confidence = self.ml_predict_intent(cleaned)
            rulebased_predicted_intent, rulebased_confidence = self.rulebased_intentrecognition(cleaned)
        
            #below two prints could be removed or commented out once absoutely confident in intent system
            print_to_gui("ML intent:", ml_predicted_intent, "| Confidence:", ml_confidence) #debugging line
            print_to_gui("Rule-based intent:", rulebased_predicted_intent, "| Confidence:", rulebased_confidence) #debugging line

            #Both agree = confident decision
            if ml_predicted_intent == rulebased_predicted_intent:
                final_predicted_intent = ml_predicted_intent
            #Rule-based wants fallback
            elif rulebased_predicted_intent == "ai_response":
                final_predicted_intent = "ai_response"  #override even a confident ML guess
            #ML not confident
            elif ml_confidence < 0.5:
                final_predicted_intent = "ai_response"
            else:
                final_predicted_intent = ml_predicted_intent
                
            #append the result for this clause
            results.append((final_predicted_intent, clause))
            
        #print("results:", results)  #debugging line
        return results
