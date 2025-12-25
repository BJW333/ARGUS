from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import string
import os
import time
import json
from config_metrics.main_config import nlp
from config_metrics.logging import log_debug #, log_metrics
from core.input_bus import print_to_gui
from filelock import FileLock
from pathlib import Path

#repo id in hugging face
intent_repo_id = "bjw333/intent_model_argus"

#load model and tokenizer directly from the hub
intent_tokenizer = AutoTokenizer.from_pretrained(intent_repo_id)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_repo_id)
log_debug(f"Intent model loaded from {intent_repo_id}")

DATASET_PATH = os.path.expanduser("~/Desktop/ARGUS/trainingdata/intent_hlrf.jsonl")

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
    #14: "codemodel",
    15: "volume_control",
    16: "objrecog"
}

id2label[17] = "ai_response"          # add your fallback class
label2id = {v: k for k, v in id2label.items()}
NUM_LABELS = len(id2label)

def log_intent_example(text: str, final_intent: str, meta: dict | None = None, path: str = DATASET_PATH):
    rec = {"text": text, "label": final_intent, "ts": int(time.time())}
    if meta:
        rec.update(meta)
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        lock_path = str(p) + ".lock"
        #all of these HLRF prints can be changed to print_to_gui if wanted
        print(f"[HLRF] Target file: {p}")
        print(f"[HLRF] Lock file:   {lock_path}")

        with FileLock(lock_path, timeout=5):
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[HLRF] Write OK. Size now: {p.stat().st_size} bytes")
    except Exception as e:
        # Surface the error loudly so you can fix it fast
        print(f"[HLRF][ERROR] {type(e).__name__}: {e}")
        try:
            log_debug(f"HLRF log failed: {e}")
        except Exception:
            pass
        raise
    
def clean_text(text: str) -> str: 
    #lowercase, trim whitespace, and collapse multiple spaces 
    t = text.lower().strip()
    return re.sub(r"\s+", " ", t)

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
        #keywords_set = set(intent_keywords)  #convert list to set for O(1) lookup #old line
        keywords_set = {k.lower() for k in intent_keywords} #case insensitive
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
        #arguscodemodel = ["I need you to write some code for me", "code"]
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
            #"codemodel": arguscodemodel,
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
        
        #inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True) #old line
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            #logits = intent_model(**inputs).logits #old line
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probs).item()
            confidence = probs[0][predicted_class_id].item()
        
        #intent = id2label[predicted_class_id] #old line
        intent = id2label.get(predicted_class_id, "ai_response") #default to ai_response as fallback if id not found

        #return intent if confidence >= threshold else "ai_response" #only returns intent
        return (intent, confidence) if confidence >= threshold else ("ai_response", confidence) #returns both confidence and predicted_intent
    
    def unified_intent_pipeline(self, user_input):
        """
        Combines ML and rule-based intent recognition to determine final intent.

        Args:
            user_input (_type_): _description_

        Returns:
            list of tuples: Each tuple contains (final_intent, clause).
        """
        clauses = split_intent_commands(user_input)
        results = []
        for clause in clauses:
            cleaned = clause.strip()
            #print("Processing clause:", cleaned) #debugging line
            ml_predicted_intent, ml_confidence = self.ml_predict_intent(cleaned)
            rulebased_predicted_intent, rulebased_confidence = self.rulebased_intentrecognition(cleaned)
        
            #below two prints could be removed or commented out once absoutely confident in intent system
            #print_to_gui(f"ML intent: {ml_predicted_intent} | Confidence: {ml_confidence:.2f}") #debugging line
            #print_to_gui(f"Rule-based intent: {rulebased_predicted_intent} | Confidence: {rulebased_confidence:.2f}") #debugging line
            
            #Decision logic:
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
            
            
            #print_to_gui(f"""| Final Intent: {final_predicted_intent} | 
            #            | Rule-based intent: {rulebased_predicted_intent}, {rulebased_confidence:.2f} | 
            #            | ML intent: {ml_predicted_intent}, {ml_confidence:.2f} |
            #            """)
            print_to_gui(f"| Final Intent: {final_predicted_intent} |")
            
            #log the final predicted intent along with metadata into the jsonl file for future training of new intent model
            log_intent_example(
                cleaned,
                final_predicted_intent,
                meta={
                    "ml_pred": ml_predicted_intent,
                    "ml_conf": round(ml_confidence, 4),
                    "rule_pred": rulebased_predicted_intent,
                    "rule_conf": round(rulebased_confidence, 4)
                }
            )
            
            #append the result for this clause
            results.append((final_predicted_intent, clause))
            
        #print("results:", results)  #debugging line
        return results
