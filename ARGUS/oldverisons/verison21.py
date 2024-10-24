import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random
import zipfile
import io
import json
import string
from transformers import GPT2LMHeadModel, AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import logging
from datetime import datetime
import time
import sounddevice as sd
import tkinter as tk
import threading
import subprocess
import webbrowser
import speech_recognition as sr
import requests
from gtts import gTTS
from bs4 import BeautifulSoup
import pyjokes
from pathlib import Path
from playsound import playsound
from pydub import AudioSegment
#from pydub.playback import play
#from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
from keybert import KeyBERT
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#file tool imports
from utilsfunctions import calculate, gatherinfofromknowledgebase, identifynetworkconnect, start_timer, coin_flip
#from objectrecognitionARGUSnewfacerecogmesh import objectrecognitionrun
import data_analysis 

nlp = spacy.load('en_core_web_sm')

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

script_dir = Path(__file__).parent
config_path = script_dir / 'Metrics/config.json'
#load config
with open(config_path) as config_file:
    config = json.load(config_file)

base_dir = Path.home()
wake_word_sound_path = base_dir / config['wake_word_sound']
hide_me_script_path = base_dir / config['hide_me_script']
password_checker_path = base_dir / config['password_checker']
spider_crawler_path = base_dir / config['spider_crawler']

MASTER = "Blake"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#forget if used or not
#sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

#not sure if open ai key needed anymore if it is then add it
#openai.api_key = 'apikey'

def change_pitch_and_speed(audio_path, pitch_semitones=-11.4, speed_factor=3.175):
    sound = AudioSegment.from_file(audio_path)

    #change pitch
    new_sample_rate = int(sound.frame_rate * (3.7 ** (pitch_semitones / 11.5)))
    pitch_changed_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    
    #change speed (tempo)
    speed_changed_sound = pitch_changed_sound._spawn(pitch_changed_sound.raw_data, overrides={'frame_rate': int(pitch_changed_sound.frame_rate * speed_factor)})

    #ensure the final sound has the original frame rate
    final_sound = speed_changed_sound.set_frame_rate(sound.frame_rate)

    return final_sound
    
def speak(text, lang='en', tld='co.uk', pitch_semitones=-11.4, speed_factor=3.175):
    if "Bot:" in text:
        text = text.replace("Bot:", "")
    tts = gTTS(text=text, lang=lang, tld=tld)
    filename = "audio.mp3"
    tts.save(filename)
    
    adjusted_sound = change_pitch_and_speed(filename, pitch_semitones, speed_factor)
    adjusted_filename = "adjusted_audio.mp3"
    adjusted_sound.export(adjusted_filename, format="mp3")
    
    playsound(adjusted_filename)
    os.remove(filename)
    os.remove(adjusted_filename)

master = None
canvas = None
circle_base_radius = 100
circle_center = (300, 300)
circles = []
CHUNK = 1024
RATE = 44100
p = None
stream = None
is_thinking = False
is_listening = False
thinking_animation_step = 0

def create_circle():
    global canvas, circle_base_radius, circle_center, circles
    for i in range(5):
        outline_color = f"#{hex(255 - i*40)[2:]:0>2}{hex(255 - i*50)[2:]:0>2}ff"
        circle = canvas.create_oval(circle_center[0] - circle_base_radius,
                                    circle_center[1] - circle_base_radius,
                                    circle_center[0] + circle_base_radius,
                                    circle_center[1] + circle_base_radius,
                                    outline=outline_color, width=5 - i)
        circles.append(circle)

def audio_setup():
    global stream, CHUNK, RATE
    CHUNK = 1024  
    RATE = 44100  
    stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=CHUNK, dtype='int16')
    stream.start()


def update_circle():
    global stream, CHUNK, canvas, circle_center, circles, is_listening
    data, _ = stream.read(CHUNK)
    data = np.frombuffer(data, dtype=np.int16)
    volume = np.linalg.norm(data) / 10
    radius = min(max(volume / 3000, 50), 200)

    for i, circle in enumerate(circles):
        radius_offset = i * 10
        x0, y0, x1, y1 = (circle_center[0] - radius - radius_offset,
                          circle_center[1] - radius - radius_offset,
                          circle_center[0] + radius + radius_offset,
                          circle_center[1] + radius + radius_offset)
        canvas.coords(circle, x0, y0, x1, y1)

    is_listening = volume > 300

    master.after(20, update_circle)


def thinking_animation():
    global is_thinking, thinking_animation_step, circles, canvas, circle_base_radius, circle_center, is_listening
    if is_thinking:
        thinking_animation_step += 1
        scale_factor = 1.05 + 0.05 * np.sin(thinking_animation_step / 10.0)
        for i, circle in enumerate(circles):
            radius_offset = i * 10
            x0, y0, x1, y1 = (circle_center[0] - circle_base_radius * scale_factor - radius_offset,
                              circle_center[1] - circle_base_radius * scale_factor - radius_offset,
                              circle_center[0] + circle_base_radius * scale_factor + radius_offset,
                              circle_center[1] + circle_base_radius * scale_factor + radius_offset)
            canvas.coords(circle, x0, y0, x1, y1)
    else:
        thinking_animation_step = 0

    if is_listening:
        for i, circle in enumerate(circles):
            outline_color = f"#{hex(255 - i*20)[2:]:0>2}{hex(255 - i*30)[2:]:0>2}ff"
            canvas.itemconfig(circle, outline=outline_color, width=5 - i)
    else:
        for i, circle in enumerate(circles):
            outline_color = f"#{hex(255 - i*40)[2:]:0>2}{hex(255 - i*50)[2:]:0>2}ff"
            canvas.itemconfig(circle, outline=outline_color, width=5 - i)

    master.after(20, thinking_animation)

def on_closing():
    global stream, master
    stream.stop()
    stream.close()
    master.destroy()

def wishme():
    hour = int(datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning " + MASTER)
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon " + MASTER)
    else:
        speak("Good Evening " + MASTER)

print("---------------------------")
print("---- Starting up Argus ----")
print("---------------------------")
wishme()

class DataStore:
    def __init__(self, filepath):
        self.filepath = filepath

    def save_data(self, data):
        try:
            with open(self.filepath, 'r') as file:
                existing_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        existing_data.append(data)

        with open(self.filepath, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def load_data(self):
        try:
            with open(self.filepath, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        
data_store = DataStore(script_dir / 'conversation_history.json')
conversation_history = data_store.load_data()


#takes the json data and processes it into txt file
def json_to_text(data):
    text_lines = []
    for entry in data:
        user_input = entry["user_input"]
        bot_response = entry["bot_response"]
        text_lines.append(f"User: {user_input}\nBot: {bot_response}\n")
    return "\n".join(text_lines)


#logging fuction
logging.basicConfig(level=logging.INFO, filename = script_dir / 'Metrics/chatbot_metrics.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s') 
                    
def log_metrics(user_input, bot_response, response_time, reward):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp}, User Input: {user_input}, Bot Response: {bot_response}, Response Time: {response_time}, Reward: {reward}")
    
def collect_human_feedback(conversation_history):
    feedback_data = []
    for conversation in conversation_history:
        if isinstance(conversation, dict) and conversation.get('flagged_for_retraining') == True:
            user_input = conversation.get('user_input', '')
            bot_response = conversation.get('bot_response', '')
            print("User:", user_input)
            print("Bot:", bot_response)
            print("Feedback Data:", feedback_data)
            feedback_data.append((user_input, bot_response))
    return feedback_data


def train_with_feedback(chatbot, sampled_dataset, feedback_data, epochs): #could include sampled_dataset as a variable so that the main dataset is also in the feedback data
    #combine the original dataset feedback data and conversation history
    combined_dataset = sampled_dataset + feedback_data #could include sampled_dataset so that the main dataset is also in the feedback data you would replace orginal dataset with that variable
    print(combined_dataset)
    #preprocess the combined dataset as you did with the original dataset
    input_texts, target_texts = zip(*combined_dataset)
    input_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=chatbot.max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=chatbot.max_length, padding='post')
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32)

    #train the model
    chatbot.modeltrain(train_dataset, epochs)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    

class IntentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(-1).item()
        return predictions

class DynamicRewardSystem:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.intent_classifier = IntentClassifier()  
        self.reward_score = 0

    def evaluate_response(self, user_input, bot_response):
        # Process the texts
        user_doc = self.nlp(user_input)
        bot_doc = self.nlp(bot_response)
        user_intent = self.intent_classifier.predict_intent(user_input)
        bot_intent = self.intent_classifier.predict_intent(bot_response)

        #Semantic and Contextual Analysis
        relevance, similarity = self.check_relevance(user_doc, bot_doc, user_input, bot_response)
        intent_match = user_intent == bot_intent  #intents match?

        #sentiment analysis
        sentiment_score = self.analyze_sentiment(user_input, bot_response)

        #update Reward
        self.update_reward(relevance, similarity, sentiment_score, intent_match)

        print(f"\nUpdated Reward Score: {self.reward_score}")
        return self.reward_score
    
    def check_relevance(self, user_doc, bot_doc, user_input, bot_response):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        bot_embedding = self.model.encode(bot_response, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, bot_embedding).item()

        contextual_match = len(set([chunk.text for chunk in user_doc.noun_chunks]) & set([chunk.text for chunk in bot_doc.noun_chunks])) > 0
        dependency_match = any(token.dep_ == bot_token.dep_ for token in user_doc for bot_token in bot_doc)

        relevance = similarity > 0.3 and (contextual_match or dependency_match)
        return relevance, similarity

    def analyze_sentiment(self, user_input, bot_response):
        user_sentiment = self.sentiment_analyzer.polarity_scores(user_input)['compound']
        bot_sentiment = self.sentiment_analyzer.polarity_scores(bot_response)['compound']
        return abs(user_sentiment - bot_sentiment)
    
    def update_reward(self, relevance, similarity, sentiment_score, intent_match):
        #positive rewards
        if relevance:
            self.reward_score += 10  #increase reward if the response is relevant
        if similarity > 0.5:
            self.reward_score += 5  #additional reward for high similarity
        if sentiment_score < 0.1:
            self.reward_score += 5  #reward alignment in sentiment
        if intent_match:
            self.reward_score += 10  #reward for matching intents

        #penalties
        if not relevance:
            self.reward_score -= 10  #penalty for irrelevant response
        if similarity < 0.3:
            self.reward_score -= 5  #penalty for low similarity
        if sentiment_score > 0.5:
            self.reward_score -= 5  #penalty for poor sentiment alignment
        if not intent_match:
            self.reward_score -= 10  #penalty for mismatched intents


    def get_total_reward(self):
        return self.reward_score
        
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Seq2SeqModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.encoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.5)
        self.attention = BahdanauAttention(hidden_units)
        self.decoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.5)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_embeddings = self.embedding(encoder_inputs)
        encoder_outputs, state_h, state_c = self.encoder(encoder_embeddings)
        context_vector, _ = self.attention(state_h, encoder_outputs)
        decoder_embeddings = self.embedding(decoder_inputs)

        #repeat the context vector across the sequence length
        repeated_context_vector = tf.repeat(tf.expand_dims(context_vector, 1), repeats=decoder_inputs.shape[1], axis=1)

        #decoder embeddings with the repeated context vector
        decoder_input_with_context = tf.concat([decoder_embeddings, repeated_context_vector], axis=-1)

        #pass the concatenated input to the decoder
        decoder_outputs, _, _ = self.decoder(decoder_input_with_context, initial_state=[state_h, state_c])
        logits = self.fc(decoder_outputs)
        return logits


class Chatbot:
    def __init__(self, vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.optimizer = tf.keras.optimizers.Adam()
    
    def preprocess_sentence(self, sentence):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  #remove punctuation
        encoded_sentence = [self.tokenizer.encode(self.start_token)[0]] + self.tokenizer.encode(sentence) + [self.tokenizer.encode(self.end_token)[0]]
        #ensure the sentence does not exceed max_length
        encoded_sentence = encoded_sentence[:self.max_length]
        #pad the sentence to max_length
        encoded_sentence = encoded_sentence + [0] * (self.max_length - len(encoded_sentence))
        return encoded_sentence

#deletes start and end tokens new mothod delte if not working
    def postprocess_sentence(self, sentence):
        if isinstance(sentence, int):
            sentence = [sentence]
        elif isinstance(sentence, list) and all(isinstance(i, int) for i in sentence):
            pass
        elif isinstance(sentence, list) and all(isinstance(i, list) for i in sentence):
            sentence = [item for sublist in sentence for item in sublist]
        else:
            sentence = sentence

        decoded_sentence = self.tokenizer.decode(sentence)
        #remove <start> and <end> tokens
        decoded_sentence = decoded_sentence.replace('start', '').replace('end', '').strip()
        return decoded_sentence

    def generate_seq2seqresponse(self, input_sentence, num_candidates=5, temperature=0.6, top_k=30):   #orginal top k = 50 #orginal num_canadiates = 10
        input_sequence = self.preprocess_sentence(input_sentence)
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=self.max_length, padding='post')
        input_tensor = tf.convert_to_tensor(input_sequence)
        start_token = self.tokenizer.encode(self.start_token)[0]
        end_token = self.tokenizer.encode(self.end_token)[0]

        candidates = []
        for _ in range(num_candidates):
            decoder_input = tf.expand_dims([start_token], 0)
            response = []
            for _ in range(self.max_length):
                predictions = self.model([input_tensor, decoder_input])

            #temperature scaling
                predictions = predictions / temperature

            #convert logits to probabilities
                predicted_probabilities = tf.nn.softmax(predictions[:, -1, :], axis=-1).numpy()[0]

            #select top-k tokens
                top_k_indices = np.argsort(predicted_probabilities)[-top_k:]
                top_k_probs = predicted_probabilities[top_k_indices]

            #normalize probabilities
                top_k_probs /= np.sum(top_k_probs)

            #sample from the top k tokens
                predicted_id = np.random.choice(top_k_indices, p=top_k_probs)

                if predicted_id == end_token:
                    break
                response.append(predicted_id)
                decoder_input = tf.concat([decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1)

            candidates.append(response)

        #rerank candidates with language model
        reranked, confidence = self.rerank_candidates(input_sentence, candidates)

        return self.postprocess_sentence(reranked), confidence


    def rerank_candidates(self, input_sentence, candidates):
        scores = []
        for candidate in candidates:
            candidate_sentence = self.postprocess_sentence(candidate)
            input_context = self.gpt2_tokenizer.encode(input_sentence + ' ' + candidate_sentence, return_tensors='pt')
            with torch.no_grad():  #save memory
                output = self.gpt2_model(input_context, labels=input_context)
                loss, logits = output[:2]
            scores.append(loss.item())
    
    #lower loss is better
        best_candidate_idx = np.argmin(scores)

        confidence = 1 / scores[best_candidate_idx]
        return candidates[best_candidate_idx], confidence  #return the best candidate sentence

    
    def modeltrain(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for (batch, (encoder_inputs, decoder_inputs)) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = self.model([encoder_inputs, decoder_inputs])
                    loss = self.compute_loss(decoder_inputs[:, 1:], logits[:, :-1, :])
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss
            print('Epoch {}, Loss {:.4f}'.format(epoch + 1, total_loss))
    
    def compute_loss(self, labels, logits):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        loss_value = loss(labels, logits, sample_weight=mask)
        return loss_value
    
    def save_model(self):
        self.model.save_weights(script_dir / 'model_weights.weights.h5')

        #if issue use the save weights line
        #self.model.save('/Users//Desktop/hello/model_weights', save_format="tf")
         
    def load_model(self):
        if os.path.exists(script_dir / 'model_weights.weights.h5'):
            #recreate the model with the known vocab size
            self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)

            #dummy call to initialize the variables
            dummy_input = [tf.zeros((1, 1)), tf.zeros((1, 1))]
            self.model(dummy_input)

            #Load the weights
            self.model.load_weights(script_dir / 'model_weights.weights.h5')
            
class intentrecognition:
    def __init__(self):
        self.dataset_path = script_dir / 'intent_dataset.csv'  # Path to store the dataset

        intentmodel_path = script_dir / 'intentmodels' / 'intent_model.pkl'
        intentvectorizer_path = script_dir / 'intentmodels' / 'tfidf_vectorizer.pkl'

        if os.path.exists(intentmodel_path) and os.path.exists(intentvectorizer_path):
            try:
                # Load model and vectorizer if they exist
                with open(intentmodel_path, 'rb') as intentmodel_file:
                    self.model = pickle.load(intentmodel_file)
                with open(intentvectorizer_path, 'rb') as intentvectorizer_file:
                    self.vectorizer = pickle.load(intentvectorizer_file)
                print("Model and vectorizer loaded successfully.")
            except (FileNotFoundError, pickle.UnpicklingError):
                print("Error loading model or vectorizer, training a new one...")
                self.train_model()
        else:
            print("Model or vectorizer not found, training a new one...")
            if not os.path.exists(self.dataset_path):
                print("Dataset not found, creating a new CSV...")
                self.create_csv()
            self.train_model()

    def create_csv(self):
        # Initialize a new DataFrame with columns 'user_input' and 'intent'
        data = {'user_input': [], 'intent': []}
        df = pd.DataFrame(data)
        
        # Save the empty DataFrame to a CSV file
        df.to_csv(self.dataset_path, index=False)
        print(f"Empty dataset created at {self.dataset_path}")
        
    def train_model(self):
        # Load the dataset from CSV
        try:
            data = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print("Dataset not found. Please ensure the dataset exists.")
            return

        if data.empty:
            print("No data available for training. Please add some intents first.")
            return

        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(data['user_input'], data['intent'], test_size=0.2, random_state=42)

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train Logistic Regression model
        self.model = LogisticRegression()
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate the model on the test set
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = (y_pred == y_test).mean()

        print(f"Model accuracy on the test set: {accuracy * 100:.2f}%")

        # Ensure the directory exists for saving
        save_path = script_dir / 'intentmodels'
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the model and vectorizer with absolute paths
        intentmodel_save_path = save_path / 'intent_model.pkl'
        intentvectorizer_save_path = save_path / 'tfidf_vectorizer.pkl'

        with open(intentmodel_save_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        with open(intentvectorizer_save_path, 'wb') as vectorizer_file:
            pickle.dump(self.vectorizer, vectorizer_file)

        print(f"Model saved at {intentmodel_save_path}")
        print(f"Vectorizer saved at {intentvectorizer_save_path}")


    def extract_entities(self, user_input):
        """Extract named entities from the user input."""
        doc = nlp(user_input)
        entities = {ent.text: ent.label_ for ent in doc.ents}
        return entities

    def intentunderstand(self, user_input):
        user_input_tfidf = self.vectorizer.transform([user_input])
        
        # Predict the intent and calculate probabilities
        predicted_intent = self.model.predict(user_input_tfidf)[0]
        #intent_probabilities = self.model.predict_proba(user_input_tfidf)

        # Get the confidence of the predicted intent
        #confidence = max(intent_probabilities[0])

        # Define a confidence threshold
        # confidence_threshold = 0.6  # You can adjust this based on testing

        # # If confidence is below the threshold, flag as unknown
        # if confidence < confidence_threshold:
        #     print(f"Bot: I'm not sure about that. Can you clarify?")
        #     speak("I'm not sure about that. Can you clarify your request?")
        #     clarified_intent = generalvoiceinput()  # Capture user's clarification

        # # If the user provides clarification, attempt to classify it again
        #     if clarified_intent:
        #         self.add_new_intent(user_input, clarified_intent)  # Add the clarified intent
        #         return clarified_intent  # Return the clarified intent
        #     else:
        #         return "unkown"

        return predicted_intent

    # def add_new_intent(self, user_input, clarified_intent):
    #     # Load the existing dataset (or create a new one)
    #     if not os.path.exists(self.dataset_path):
    #         self.create_csv()
        
    #     # Append the new input and its clarified intent to the dataset
    #     data = pd.read_csv(self.dataset_path)
    #     new_data = {'user_input': user_input, 'intent': clarified_intent}
    #     data = data.append(new_data, ignore_index=True)

    #     # Save the updated dataset back to the file
    #     data.to_csv(self.dataset_path, index=False)

    #     # Retrain the model with the new data
    #     self.train_model()

       
#actions that can be taken
def action_time():
    current_time = time.strftime("%H:%M")
    return f"Bot: The current time is {current_time}"

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
        "exclamation point": "!"
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


def generalvoiceinput(): 
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")  
        try: 
            voiceinputauido = recognizer.listen(source, timeout=4)
            inputbyvoice = recognizer.recognize_google(voiceinputauido, language='en-us')
            print("Me  --> ", inputbyvoice)
            return inputbyvoice
        except:
            print("Me  -->  ERROR")     
            return None     
        
def listen_for_wake_word():
    global is_thinking
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        #print("Listening for wake word and input...")
        while True:
            print("Listening for wake word and input...")
            audio = recognizer.listen(source)
            is_thinking = True
            try:
                maininput = recognizer.recognize_google(audio, language='en-us')
                is_thinking = False
                print("Me  --> ", maininput)
                if "argus" in maininput.lower():
                    #use regex to remove the wake word and any surrounding spaces
                    maincommand = re.sub(r'\b(argus)\b', '', maininput, flags=re.IGNORECASE).strip()
                    #remove any double spaces created by removing the wake word
                    maincommand = re.sub(r'\s+', ' ', maincommand)
                    #print("this the command:", command)
                    handle_wake_word_detected(maincommand)
            except sr.WaitTimeoutError:
                is_thinking = False
                print("No audio heard")
            except sr.UnknownValueError:
                is_thinking = False
            except sr.RequestError as e:
                is_thinking = False
                print(f"Error with the request; {e}")
    
        
def handle_wake_word_detected(maininputcommand):
    if maininputcommand:
        #print("Processing input") debugging
        process_user_input(maininputcommand)
    #process_user_input(maininputcommand)
    
    listen_for_wake_word()

    
def process_user_input(user_input):
    #print(f"Processing user input: {user_input}") debugging
    
    intent_recog = intentrecognition()
    
    #extract entities and recognize intent
    entities = intent_recog.extract_entities(user_input)
    intent = intent_recog.intentunderstand(user_input)
    
    if intent == "exit":
        chatbot.save_model()
        print("Vocabulary Size:", vocab_size)
        root.quit()
        exit()
        
        
    elif intent == "searchwiki" and 'PERSON' in entities.values():
        #print("person describer active")
        wiki = gatherinfofromknowledgebase(entities)
        if wiki:
            responsewhoiswhatis = f"Bot: {wiki}"
        elif "No results found." in wiki:
            responsewhoiswhatis = "Bot: I couldn't find any information regarding that."
        response = responsewhoiswhatis
        print(responsewhoiswhatis)
        speak(responsewhoiswhatis)
        
    #elif intent == "objrecog":
    #    objectrecognitionrun()
        
    elif intent == "connectionwithinternet":
        connectionstatus = identifynetworkconnect()
        #print(connectionstatus)
        if connectionstatus:
            speak("Internet is connected")
        else: 
            speak("Internet is not connected")
            
            
    elif intent == "stock_data":
        # Ask the user for the stock symbol
        print("Please tell me the stock symbol you're interested in.")
        speak("Please tell me the stock symbol you're interested in.")
        stock_symbol = generalvoiceinput()
        if stock_symbol:
            # Initialize the stock data stream
            stock_stream = data_analysis.StockDataStream(symbol=stock_symbol.upper(), api_key='MJX8BVSA9W1WOEH4')
            data = stock_stream.fetch_data()
            if not data.empty:
                stock_stream.analyze_data(data)
                # Convert analysis to speech
                latest_price = data['Close'].iloc[-1]
                speak(f"The latest price of {stock_symbol.upper()} is ${latest_price:.2f}")
                
                #visualize data among the stream data
                #stock_stream.visualize_data(data)
            else:
                speak(f"Sorry, I couldn't fetch data for {stock_symbol.upper()}.")
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
         
    elif intent == "searchwiki":
        kw_model = KeyBERT()

        keywords = kw_model.extract_keywords(user_input, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)

        #Print the most probable key term
        if keywords:
            most_probable_keyword = keywords[0][0]  #the first element is the most probable keyword
            print("Most Probable Key Term:", most_probable_keyword)
        else:
            print("No keywords found.")

        # Perform the Wikipedia search
        wiki = gatherinfofromknowledgebase(most_probable_keyword)
        if wiki:
            responsewhoiswhatis = f"Bot: {wiki}"
        elif "No results found." in wiki:
            responsewhoiswhatis = "Bot: I couldn't find any information regarding that."
        response = responsewhoiswhatis
        print(responsewhoiswhatis)
        speak(responsewhoiswhatis)          
                
                      
    elif intent == "websearch":    
        speak(f"What can I search for you, {MASTER}?")
        speak("Just state what you want to search.")
        searchstuff = generalvoiceinput()

        if searchstuff:
            webbrowser.open('https://www.google.com/search?q=' + searchstuff)
            speak("I searched what you asked me to")
        else:
            speak("I didn't catch that. Please try again.")   
            
            
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
            epochs3 = 10
            train_with_feedback(chatbot, sampled_dataset, feedback_data, epochs3)
            
            chatbot.save_model() #save model after training
            
        elif manualinput.lower() == 'train':
            epochtotrain = 10
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
            "- Searching on Google\n"
            "- Opening apps\n"
            "- Telling you the date and time\n"
            "- I can have conversations\n"
            "- I can run custom tools\n"
            "- Writing Notes\n"
            "- Running 'hide me'\n"
            "- Identifying people\n"
            "- Searching for information\n"
            "- Providing news updates\n"
            "- Telling jokes"
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
        response, confidence = chatbot.generate_seq2seqresponse(user_input, temperature=0.6)# temp was 0.5
        #print("This is the confidence value:", confidence)
        reward = reward_system.evaluate_response(user_input, response)
        #print("this is new reward eval resp:", reward)
        response_printed = False
        confidence_threshold = 0.165
        corrected_response = None
        responseisgoodorbad = False
        
        
        if reward >= 5:   #adjust as needed
            responseisgoodorbad = True  
            print("True")  
        elif reward <= 4:  #adjust as needed
            responseisgoodorbad = False
            print("False")
            
        #print(confidence)    
        #print(responseisgoodorbad)
        
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
        print("Confidence:", confidence)
        print("Response time:", response_time) #amount of time it takes for bot to respond
        
        
        conversation_history.append((user_input, response))
        
        conversation_data = {
            'user_input': user_input,
            'bot_response': response,
            'reward': reward,  #reward mechanism
            'flagged_for_retraining': flagged_for_retraining  #flag for retraining based on checking system
        }
        data_store.save_data(conversation_data)  
        
                        
#if the paths exist then its not redownloaded
if os.path.exists(script_dir / "data/inputtexts.txt") and os.path.exists(script_dir / "data/outputtexts.txt"):
    print("Files already exist.")
else:
    #print("Missing data files.")
    print("Downloading dataset...")
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    response = requests.get(url)
    zipfile.ZipFile(io.BytesIO(response.content)).extractall()

    #download and extract the dataset
    dataset_dir = 'cornell movie-dialogs corpus'
    dialogues_file = os.path.join(dataset_dir, 'movie_lines.txt')
    conversations_file = os.path.join(dataset_dir, 'movie_conversations.txt')

#load the dialogues into a dictionary
    dialogues = {}
    with open(dialogues_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            dialogues[parts[0]] = parts[-1].strip()

#extract the conversations and save to files
    with open(conversations_file, 'r', encoding='iso-8859-1') as f, \
         open(script_dir / 'data/inputtexts.txt', 'w', encoding='utf-8') as conv_file, \
         open(script_dir / 'data/outputtexts.txt', 'w', encoding='utf-8') as ans_file:
        for line in f:
            parts = line.split(' +++$+++ ')
            conversation = eval(parts[-1].strip())
            for i in range(len(conversation) - 1):
                input_dialogue = dialogues[conversation[i]]
                target_dialogue = dialogues[conversation[i + 1]]
                conv_file.write(input_dialogue + '\n')
                ans_file.write(target_dialogue + '\n')

with open(script_dir / 'data/inputtexts.txt', 'r', encoding='utf-8') as conv_file:
    input_texts = conv_file.readlines()
    

with open(script_dir / 'data/outputtexts.txt', 'r', encoding='utf-8') as ans_file:
    target_texts = ans_file.readlines()
    

dataset = list(zip(input_texts, target_texts))

#sample a subset of the dataset
#sample_size = 200
sampled_dataset = (dataset)
#load tokenizer and get vocab size

if os.path.exists(script_dir / 'tokenizer.pickle'):
    with open(script_dir / 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        vocab_size = tokenizer.vocab_size
else:
    input_texts, target_texts = zip(*sampled_dataset)  #use sampled dataset here
    all_texts = input_texts + target_texts
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (text for text in all_texts), target_vocab_size=2**13)
    vocab_size = tokenizer.vocab_size
    with open(script_dir / 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    reward_system = DynamicRewardSystem()
    #define your dataset and other parameters
    input_texts, target_texts = zip(*sampled_dataset)  #use sampled dataset here
    all_texts = input_texts + target_texts

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
(text for text in all_texts), target_vocab_size=2**13)
    

    start_token = '<start>'
    end_token = '<end>'
    vocab_size = tokenizer.vocab_size
    embedding_dim = 200 #256 is default
    hidden_units = 256 #default 512
    max_length = 25 #default 20 new 30 #25 is average sentence length of data
    epochs = 20 #default 10
    epochsstart = 20 #old value was 35
    #start and train the chatbot
    chatbot = Chatbot(vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length)

    initial_learning_rate = 0.004 #orginal value of initial learning rate was 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=634, decay_rate=0.998, staircase=True) #orginal decay steps 100000
    chatbot.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    input_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_length, padding='post')
    #uncomment out below line with suflle method if porblem arrise that makes training more random with sequences
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)
    #suffle method not included in below line reduces randomness of training of sequences
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32)
    
    #old line is above new line below delte if issue all it does is change batch size from 32 to 8
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(8)



    #os.system('clear') #clear anything before program starting up
    if os.path.exists(script_dir / 'model_weights.weights.h5'):
        chatbot.load_model()
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
    else:
        chatbot.modeltrain(train_dataset, epochsstart)
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
        chatbot.save_model()

    

    #conversation_history = []
    data_store = DataStore(script_dir / 'conversation_history.json')
    
    root = tk.Tk()
    master = root
    master.title("ARGUS")
    master.geometry("600x600")
    master.configure(bg='black')

    canvas = tk.Canvas(master, width=600, height=600, bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    create_circle()
    audio_setup()
    update_circle()

    wake_word_thread = threading.Thread(target=listen_for_wake_word)
    wake_word_thread.start()    


    thinking_animation()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

                
            

        
        
        
