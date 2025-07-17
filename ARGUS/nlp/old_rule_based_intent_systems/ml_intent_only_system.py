from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from huggingface_hub import snapshot_download
from config import nlp


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
    
    def predict_intent(self, text, threshold=0.5):
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
        print(user_input)
        cleaned = (user_input)
        print(cleaned)
        predicted_intent, confidence = self.predict_intent(cleaned)
        print(confidence)
        if confidence <= 0.5:
            return "ai_response"

        return predicted_intent