import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random
#import zipfile
#import io
import json
import string
from transformers import AutoModelForSequenceClassification, AutoTokenizer #PT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
#import torch
import pickle
import logging
from datetime import datetime
import time
import sounddevice as sd
#import tkinter as tk  # no longer used for GUI
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
#import yfinance as yf      #not sure if this import is being used anymore
import csv
from pytickersymbols import PyTickerSymbols
import pandas as pd
import sys
import math

from PyQt5.QtCore import QTimer, Qt, QRectF, QMetaObject, Q_ARG
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QTextEdit, QPushButton, QSplitter


#file tool imports
from utilsfunctions import calculate, gatherinfofromknowledgebase, identifynetworkconnect, start_timer, coin_flip, cocktail
#from objectrecognitionARGUSnewfacerecogmesh import objectrecognitionrun 
#above line was orginally supposed to allow argus to have eyes but ran into a issue with the current threading system and opencv this resulted in crashing of the system
import data_analysis 
from arguscode_model import argus_code_generation


#nlp = spacy.load('en_core_web_sm')
#above line uses the old en_core_web_sm but that doesnt have word vectors and the new line below en_core_web_md does so that is now used globally
#if there is a issue use en_core_web_sm instead and work on fixing issues that arrise 
#also en_core_web_md is a larger model than that of en_core_web_sm which can be extremely benefical 
nlp = spacy.load("en_core_web_md")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#new line below
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  #prevents performance issues on apple silicon
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#MY_TOKEN = "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"
#os.environ["HF_AUTH_TOKEN"] = MY_TOKEN

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


#forget if used or not
#sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


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


#Output Redirection

class OutputRedirector:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        
        # QMetaObject.invokeMethod(
        #     self.text_edit,
        #     "append",
        #     Qt.QueuedConnection,
        #     Q_ARG(str, text)
        # )
        QMetaObject.invokeMethod(
            self.text_edit,
            "insertPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, text)
        )
        
    def flush(self):
        pass

circle_base_radius = 100
circle_center = (300, 300)  #(x, y) not used directly here but you can adjust drawing logic
circles = []  #Will be generated in the custom widget
CHUNK = 1024
RATE = 44100
stream = None
is_thinking = False
is_listening = False
thinking_animation_step = 0

def audio_setup():
    global stream, CHUNK, RATE
    CHUNK = 1024  
    RATE = 44100  
    stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=CHUNK, dtype='int16')
    stream.start()

def update_audio_volume():
    global stream
    data, _ = stream.read(CHUNK)
    data = np.frombuffer(data, dtype=np.int16)
    volume = np.linalg.norm(data) / 10
    return volume

#GUI Components

class AnimatedCanvas(QWidget):
    def __init__(self, parent=None):
        super(AnimatedCanvas, self).__init__(parent)
        self.animation_step = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(20)  #update every 20ms

        #timer for audio driven update
        self.audio_timer = QTimer(self)
        self.audio_timer.timeout.connect(self.update)
        self.audio_timer.start(20)

    def update_animation(self):
        global is_thinking, thinking_animation_step
        if is_thinking:
            thinking_animation_step += 1
        else:
            thinking_animation_step = 0
        self.update()  #trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        #fill background with black
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        #get widget center
        center_x = self.width() // 2
        center_y = self.height() // 2

        #determine a radius based on audio volume
        volume = update_audio_volume() if stream is not None else 0
        #map volume to a radius value between 50 and 200
        radius = min(max(volume / 3000, 50), 200)

        #draw 5 circles with offsets and different colors
        for i in range(5):
            radius_offset = i * 10
            #if thinking apply a scale factor
            scale_factor = 1.05 + 0.05 * math.sin(thinking_animation_step / 10.0) if is_thinking else 1.0
            r = (circle_base_radius * scale_factor + radius_offset) + radius - 50

            #change outline color based on listening state
            if is_listening or volume > 300:
                r_color = max(0, 255 - i * 20)
                g_color = max(0, 255 - i * 30)
            else:
                r_color = max(0, 255 - i * 40)
                g_color = max(0, 255 - i * 50)
            color = QColor(r_color, g_color, 255)
            pen_width = max(1, 5 - i)
            pen = QPen(color, pen_width)
            painter.setPen(pen)

            #draw the ellipse centered in the widget
            rect = QRectF(center_x - r, center_y - r, 2 * r, 2 * r)
            painter.drawEllipse(rect)
            
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ARGUS")
        self.resize(600, 750) #orginal values 600 and 600
        
        #Use a splitter to separate the canvas and output area
        splitter = QSplitter(Qt.Vertical)
        
        #your animated canvas (ARGUS visualization)
        self.canvas = AnimatedCanvas(self)
        splitter.addWidget(self.canvas)
        
        #create a text area for terminal output
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #ffffff;
                font-family: Menlo, Monaco, Courier, monospace;
            }
        """)
        splitter.addWidget(self.output_area)
        
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #222; }")
        
        splitter.setSizes([600, 200]) # orginal values 500 and 100
        
        #add a clear button
        clear_btn = QPushButton("Clear Output")
        clear_btn.clicked.connect(self.output_area.clear)
        
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(clear_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        #redirect stdout and stderr to the text area
        sys.stdout = OutputRedirector(self.output_area)
        sys.stderr = OutputRedirector(self.output_area)
        
    def closeEvent(self, event):
        if stream is not None:
            stream.stop()
            stream.close()
        event.accept()


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
    #combined_dataset = feedback_data #just feedback data
    print(combined_dataset)
    #preprocess the combined dataset as you did with the original dataset
    input_texts, target_texts = zip(*combined_dataset)
    input_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=chatbot.max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=chatbot.max_length, padding='post')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)  #shuffle so the model generalizes better
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32)  #this is the old line without shuffle

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
        self.nlp = nlp
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        #self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #maybe remove these
        #self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2') #maybe remove these
        self.intent_classifier = IntentClassifier()  
        self.reward_score = 0

    def evaluate_response(self, user_input, bot_response):
        
        self.reward_score = 0  # Reset reward score for each evaluation this is a massive test to ensure new normalziation between -30 and 30 (0 and 1) works
        
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
        
        #maybe multihead attention here instead of BahdanauAttention might be extremely beneficial from a attention standpoint
        #import torch.nn as nn
        #self.attention = nn.MultiheadAttention(hidden_units)
        
        
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
        #self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #maybe remove these
        #self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2') #maybe remove these
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.optimizer = tf.keras.optimizers.Adam()
        
        #new inputs below
        self.nlp = nlp
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')        
        self.sentiment_analyzer = DynamicRewardSystem()
        self.intent_classifier = IntentClassifier()  
        
    def preprocess_sentence(self, sentence):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  #remove punctuation
        encoded_sentence = [self.tokenizer.encode(self.start_token)[0]] + self.tokenizer.encode(sentence) + [self.tokenizer.encode(self.end_token)[0]]
        #ensure the sentence does not exceed max_length
        encoded_sentence = encoded_sentence[:self.max_length]
        #pad the sentence to max_length
        encoded_sentence = encoded_sentence + [0] * (self.max_length - len(encoded_sentence))
        return encoded_sentence

    #deletes start and end tokens and does post processing new mothod delte if not working
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
        
        #remove whitespace new line below
        decoded_sentence = re.sub(r'\s+', ' ', decoded_sentence).strip()

        #Fix splitup words 
        decoded_sentence = re.sub(r'\b(\w)\s(?=\w\b)', r'\1', decoded_sentence)
        # Specific fixes for common contractions
        decoded_sentence = re.sub(r"\b([Ii])\s+m\b", r"\1'm", decoded_sentence)          # i m -> I'm
        decoded_sentence = re.sub(r"\b([Yy]ou)\s+re\b", r"\1're", decoded_sentence)       # you re -> you're
        decoded_sentence = re.sub(r"\b([Ww]e)\s+re\b", r"\1're", decoded_sentence)         # we re -> we're
        decoded_sentence = re.sub(r"\b([Tt]hey)\s+re\b", r"\1're", decoded_sentence)       # they re -> they're
        decoded_sentence = re.sub(r"\b([Dd]on)\s+t\b", r"\1't", decoded_sentence)          # don t -> don't
        decoded_sentence = re.sub(r"\b([Cc]an)\s+t\b", r"\1't", decoded_sentence)          # can t -> can't
        decoded_sentence = re.sub(r"\b([Ww]on)\s+t\b", r"\1't", decoded_sentence)          # won t -> won't
        decoded_sentence = re.sub(r"\b([Dd]idn)\s+t\b", r"\1't", decoded_sentence)         # didn t -> didn't
        decoded_sentence = re.sub(r"\b([Dd]oesn)\s+t\b", r"\1't", decoded_sentence)         # doesn t -> doesn't
        decoded_sentence = re.sub(r"\b([Ss]houldn)\s+t\b", r"\1't", decoded_sentence)       # shouldn t -> shouldn't
        decoded_sentence = re.sub(r"\b([Cc]ouldn)\s+t\b", r"\1't", decoded_sentence)        # couldn t -> couldn't
        decoded_sentence = re.sub(r"\b([Ww]ouldn)\s+t\b", r"\1't", decoded_sentence)        # wouldn t -> wouldn't
        decoded_sentence = re.sub(r"\b([Ii])\s+ve\b", r"\1've", decoded_sentence)           # i ve -> i've
        decoded_sentence = re.sub(r"\b([Yy]ou)\s+ve\b", r"\1've", decoded_sentence)         # you ve -> you've
        decoded_sentence = re.sub(r"\b([Ww]e)\s+ve\b", r"\1've", decoded_sentence)           # we ve -> we've
        decoded_sentence = re.sub(r"\b([Tt]hey)\s+ve\b", r"\1've", decoded_sentence)         # they ve -> they've

        # Then, add a generic fix for cases like "what s" -> "what's"
        decoded_sentence = re.sub(r"\b(\w+)\s+s\b", r"\1's", decoded_sentence)
        # Optional: further fix split-up words if needed
        #decoded_sentence = re.sub(r'\b(\w)\s(?=\w\b)', r'\1', decoded_sentence)
        
        return decoded_sentence
    
    def calculate_confidencevalue_response(self, user_input, bot_response, fitness_score, candidates):
        """
        Calculate confidence based on fitness score, semantic similarity, intent match, and diversity.
        """
        #Normalize Fitness Score
        #max_fitness = 20  # Define a reasonable max fitness score based on observations
        #F = max(0, min(fitness_score / max_fitness, 1))  #Normalize between 0 and 1
        
        max_fitness = 20  #initial assumption

        F = fitness_score / max_fitness  # Normalize
        F = min(F, 1)  #cap at 1
        
        #Semantic Similarity (S)
        user_embedding = self.semantic_model.encode(user_input, convert_to_tensor=True)
        bot_embedding = self.semantic_model.encode(bot_response, convert_to_tensor=True)
        S = util.pytorch_cos_sim(user_embedding, bot_embedding).item()

        #Response Diversity Factor (D)
        candidate_embeddings = [self.semantic_model.encode(c, convert_to_tensor=True) for c in candidates]
        avg_similarity = np.mean([util.pytorch_cos_sim(bot_embedding, c).item() for c in candidate_embeddings])
        D = 1 - avg_similarity  #Lower similarity = higher diversity

        #Compute final confidence score with weights
        #confidence = (0.3 * F) + (0.4 * S) + (0.2 * D)  # Reduce fitness dominance
        #confidence = (0.3 * F) + (0.3 * S) + (0.4 * D) #more diversty in responses      this was old line 
        #confidence = max(0, min(confidence, 1))  #Ensure it's between 0-1
        #confidence = np.clip(confidence, 0, 1)      this was old line that was used optimized line below

        confidence = np.clip(0.3 * F + 0.3 * S + 0.4 * D, 0, 1)
        
        return confidence
    
    def generate_seq2seqresponse(self, input_sentence, num_candidates=5, temperature=0.6, top_k=30):
        #preprocess the input sentence and convert to a tensor
        input_sequence = self.preprocess_sentence(input_sentence)
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [input_sequence], maxlen=self.max_length, padding='post'
        )
        input_tensor = tf.convert_to_tensor(input_sequence)

        #prepare start/end tokens
        start_token_id = self.tokenizer.encode(self.start_token)[0]
        end_token_id = self.tokenizer.encode(self.end_token)[0]
        

        #collect multiple candidate responses from generation
        unique_candidates = set()
        all_candidates = []
        while len(all_candidates) < num_candidates:
            decoder_input = tf.expand_dims([start_token_id], 0)
            response_ids = []
            for _ in range(self.max_length):
                #pass through the model
                predictions = self.model([input_tensor, decoder_input])
                
                #apply temperature scaling
                predictions = predictions / temperature
                
                #convert to probabilities
                predicted_probabilities = tf.nn.softmax(predictions[:, -1, :], axis=-1).numpy()[0]
                
                #select top-k tokens
                top_k_indices = np.argsort(predicted_probabilities)[-top_k:]
                top_k_probs = predicted_probabilities[top_k_indices]
                
                #normalize the top-k probabilities
                top_k_probs /= np.sum(top_k_probs)
                
                #sample from the top-k tokens
                predicted_id = np.random.choice(top_k_indices, p=top_k_probs)
                
                #stop if we hit the end token
                if predicted_id == end_token_id:
                    break

                #append to response
                response_ids.append(predicted_id)
                
                #update decoder input
                decoder_input = tf.concat(
                    [decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1
                )

            #postprocess token ids into a string
            candidate_text = self.postprocess_sentence(response_ids)
            if candidate_text not in unique_candidates:
                unique_candidates.add(candidate_text)
                all_candidates.append(candidate_text)
                print("New candidate:", candidate_text)
            else:
                print("Duplicate candidate detected. Regenerating...")
            
        #return the list of candidate strings
        return all_candidates


    def crossover_text_advanced(self, parent_a: str, parent_b: str) -> str:
        """
        Perform parse-tree–based crossover by extracting top-level clauses
        from each parent, then splicing them.

        1) Parse each parent's text with spaCy.
        2) Extract clauses with 'extract_clauses'.
        3) Randomly choose half from each and combine.
        4) Optionally reorder them or do grammar check at the end.
        """
        
        def subtree_text(root_token):
            """
            Return the text of an entire subtree from 'root_token',
            including punctuation tokens that fall under it in the parse tree.
            """
            #collect all tokens in the subtree
            tokens = list(root_token.subtree)
            #sort them by their position in doc
            tokens = sorted(tokens, key=lambda t: t.i)
            #join their text
            return " ".join(t.text for t in tokens)

        def extract_clauses(doc):
            """
            Extract top-level clauses (root subtrees) from a spaCy doc.
            For each sentence, we grab the subtree of the main root.
            We also handle certain conj or coordinating roots for multiple clauses.
            """
            clauses = []
            for sent in doc.sents:
                # main root
                root = sent.root
                if root is not None:
                    clause_text = subtree_text(root)
                    if clause_text.strip():
                        clauses.append(clause_text)

                #optionally look for coordinated roots or conj
                #eg "He ran and he jumped"
                #we can gather other 'conj' heads that match the root or sentence boundary
                for token in sent:
                    if token.dep_ == "conj" and token.head == root:
                        conj_text = subtree_text(token)
                        if conj_text.strip():
                            clauses.append(conj_text)
            return clauses
        
        doc_a = nlp(parent_a)
        doc_b = nlp(parent_b)

        clauses_a = extract_clauses(doc_a)  # list of strings
        clauses_b = extract_clauses(doc_b)

        if not clauses_a and not clauses_b:
            return parent_a  #fallback if both empty
        if not clauses_a:
            return parent_b
        if not clauses_b:
            return parent_a

        #pick half from A and half from B
        half_a = random.sample(clauses_a, k=max(1, len(clauses_a)//2))
        half_b = random.sample(clauses_b, k=max(1, len(clauses_b)//2))

        child_clauses = half_a + half_b
        random.shuffle(child_clauses)  #optional shuffle

        #Duplicate Removal Only When Needed
        if len(set(child_clauses)) < len(child_clauses):  #check if duplicates exist
            child_clauses = list(dict.fromkeys(child_clauses))  #remove duplicates efficiently

        child_text = " ".join(child_clauses)
        print("Child text:", child_text)

        #optionally run a grammar correction pass would have to make a new method for this 
        #child_text = self.grammar_correct(child_text)

        return child_text


    #----------------------------------------------------------------------------------------



    def mutate_text_all_in_one(self, text: str) -> str:
        """
        A single function that performs advanced mutation:
        1) Parse text, extract clauses
        2) Randomly do one of:
            - remove a clause
            - reorder clauses
            - synonym-replace up to 2 content words in one random clause

        This function inlines:
        - Clause extraction
        - get similar word lookup
        - Clause-level mutation
        """

        # -------------- Inline Helpers ---------------

        def subtree_text(root_token):
            """
            Return the text of an entire subtree from 'root_token',
            including punctuation tokens in that subtree.
            """
            tokens = list(root_token.subtree)
            tokens = sorted(tokens, key=lambda t: t.i)
            return " ".join(t.text for t in tokens)

        def extract_clauses(doc):
            """
            Extract top-level clauses (root subtrees) from a spacy doc
            For each sentence we grab the subtree of the main root
            We also handle 'conj' heads that match the root for additional clauses
            """
            clauses = []
            for sent in doc.sents:
                root = sent.root
                if root is not None:
                    clause_text = subtree_text(root)
                    if clause_text.strip():
                        clauses.append(clause_text)

                # Look for conj tokens
                for token in sent:
                    if token.dep_ == "conj" and token.head == root:
                        conj_text = subtree_text(token)
                        if conj_text.strip():
                            clauses.append(conj_text)
            return clauses
        
        def get_similar_word(word, threshold=0.7, top_n=10):
            """
            Return a semantically similar word (if available) using word vectors.
            If no candidate meets the similarity threshold, return the original word.
            """
            #get the token from the word (using the first token of the doc)
            token = nlp(word)[0]
            if not token.has_vector:
                return word  #if no vector is available for this word, return it unchanged.
            
            candidates = []
            
            # Iterate over a filtered subset of the vocabulary.
            # This filters out words that are not lower-case not alphabetic or are very rare.
            for lex in nlp.vocab:
                if (lex.has_vector and lex.is_lower and lex.is_alpha and 
                    lex.prob > -15 and lex.text != word):
                    sim = token.similarity(lex)
                    if sim >= threshold:
                        candidates.append((lex.text, sim))
            
            #sort candidates by similarity (highest first) and take the top_n
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
            
            if candidates:
                #randomly choose one of the top candidates
                return random.choice([w for w, sim in candidates])
            
            return word

        # -------------- Start of Actual Mutation Logic ---------------

        doc = nlp(text)
        clauses = extract_clauses(doc)
        if not clauses:
            return text  # nothing to mutate

        # We'll pick one of three mutation types
        mutation_type = random.choice(["remove_clause", "reorder", "synonym_replace"])
        
        mutated_clauses = clauses[:]  # Initialize with the original clauses

        if mutation_type == "remove_clause" and len(clauses) > 1:
            remove_idx = random.randint(0, len(clauses) - 1)
            mutated_clauses.pop(remove_idx)  
            print("mutated clauses before mutation type remove clause", mutated_clauses)
            mutated_text = " ".join(mutated_clauses)

        # 2) reorder
        elif mutation_type == "reorder" and len(clauses) > 1:
            #old line for below new line random.shuffle(clauses)
            random.shuffle(mutated_clauses)
            print("mutated clauses before mutation type remove clause", mutated_clauses)
            mutated_text = " ".join(mutated_clauses)

        # 3) synonym_replace
        else:
            # pick one random clause to synonym-replace up to 2 content words
            clause_idx = random.randint(0, len(clauses) - 1)
            chosen_clause = clauses[clause_idx]

            # parse the chosen clause
            clause_doc = nlp(chosen_clause)
            tokens = [t for t in clause_doc]

            # find content tokens
            content_indices = [i for i, t in enumerate(tokens)
                            if t.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
            if not content_indices:
                # if no content words, we can't do synonyms; fallback to original
                #old line for below new line mutated_clauses = clauses
                mutated_clauses = clauses[:]
            else:
                # pick how many replacements we do (1 or 2)
                replace_count = random.randint(1, min(2, len(content_indices)))
                indices_to_replace = random.sample(content_indices, replace_count)

                mutated_tokens = tokens[:]
                for idx_replace in indices_to_replace:
                    old_token = tokens[idx_replace]
                    new_word = get_similar_word(old_token.text, threshold=0.7, top_n=10)
                    if new_word != old_token.text:
                        print(f"Replacing '{old_token.text}' with '{new_word}'")
                        mutated_tokens[idx_replace] = new_word
                        #         mutated_tokens[idx_replace] = nlp(new_word)[0].text  # Ensures correct formatting

                        
                # reconstruct mutated clause
                mutated_clause_text = " ".join(t if isinstance(t, str) else t.text
                                            for t in mutated_tokens)
                mutated_clauses = clauses[:clause_idx] + [mutated_clause_text] + clauses[clause_idx+1:]

            

            if len(set(mutated_clauses)) < len(mutated_clauses):
                print("Duplicates detected and removed:", set(mutated_clauses))
                mutated_clauses = list(dict.fromkeys(mutated_clauses))
            
            print("Mutated clauses:", mutated_clauses)
            
            mutated_text = " ".join(mutated_clauses)
            print("Mutated text:", mutated_text)
        

        return mutated_text

    def get_fitness_score(self, user_input, bot_response):
        # This version does not rely on a running self.reward_score.
        reward = 0

        # (1) Do the same checks you do in evaluate_response
        user_doc = self.nlp(user_input)
        bot_doc = self.nlp(bot_response)
        user_intent = self.intent_classifier.predict_intent(user_input)
        bot_intent = self.intent_classifier.predict_intent(bot_response)

        relevance, similarity = self.sentiment_analyzer.check_relevance(user_doc, bot_doc, user_input, bot_response)
        sentiment_score = self.sentiment_analyzer.analyze_sentiment(user_input, bot_response)
        intent_match = (user_intent == bot_intent)

        # (2) Adjust 'reward' but not self.reward_score
        if relevance:
            reward += 10
        if similarity > 0.5:
            reward += 5
        if sentiment_score < 0.1:
            reward += 5
        if intent_match:
            reward += 10

        if not relevance:
            reward -= 10
        if similarity < 0.3:
            reward -= 5
        if sentiment_score > 0.5:
            reward -= 5
        if not intent_match:
            reward -= 10

        return reward
    
    def ga_rerank_candidates(self, user_input, candidates,
                            pop_size=10, generations=3,
                            crossover_rate=0.5, mutation_rate=0.3): #mutation rate was 0.1 orginally generations was 3
        """
        1) Start with 'candidates' as the initial population.
        2) Evaluate them with 'reward_system' as the fitness.
        3) Evolve for 'generations' times.
        4) Return the best final string.
        """
        # ensure population has at least pop_size
        candidates = list(set(candidates))  # Remove duplicates
        population = candidates[:pop_size]
        while len(population) < pop_size:
            population.append(random.choice(candidates))

        def fitness_func(candidate_text):
            #used to be reward_system.get_fitness_score in the below line
            return self.get_fitness_score(user_input, candidate_text)

        # Evaluate initial population
        #scores = [fitness_func(c) for c in population] # old line new below
        scores = np.array([fitness_func(c) for c in population]) #new optimized line with nummpy less memory useage delete line and use one above if issue

        for gen in range(generations):
            # Selection: pick top half as parents
            ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            parents = [p for p, s in ranked[:pop_size // 2]]

            # Make new population
            new_population = []
            while len(new_population) < pop_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                child = parent1
                if random.random() < crossover_rate:
                    child = self.crossover_text_advanced(parent1, parent2)

                if random.random() < mutation_rate:
                    child = self.mutate_text_all_in_one(child)

                new_population.append(child)

            population = new_population
            #scores = [fitness_func(c) for c in population] # old line new below
            scores = np.array([fitness_func(c) for c in population]) #new optimized line with nummpy less memory useage delete line and use one above if issue

        # Finally, pick the best
        best_idx = max(range(len(population)), key=lambda i: scores[i])
        best_candidate = population[best_idx] 
        print("best canidate GA method return:", best_candidate)
        best_score = scores[best_idx]
        
        #trying to do postprocessing below diffrent order new line below

        return best_candidate, best_score
    
    
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
    
    def calculate_confidence(self, user_input, intent_keywords):
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
            "codemodel": arguscodemodel
        }
        
        #user_input_lower = user_input.lower()
        best_intent = "unknown"
        highest_confidence = 0.0
        
        for intent, keywords in intent_map.items():
            confidence = self.calculate_confidence(user_input, keywords)
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

       
#actions that can be taken
def action_time():
    current_time = time.strftime("%I:%M %p")
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
    intent = intent_recog.interactiveintentrecognition(user_input) #used to be intent_recog.intentunderstand(user_input) orginally
    
    if intent == "exit":
        chatbot.save_model()
        print("Vocabulary Size:", vocab_size)
        QApplication.quit()
        
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
            "- Searching on Google and Wikipedia\n"
            "- Opening and closing apps\n"
            "- Telling you the date and time\n"
            "- I can have conversations\n"
            "- I can run custom tools\n"
            "- Writing Notes\n"
            "- Running 'hide me'\n"
            "- Identifying people\n"
            "- Searching for information\n"
            "- Providing news updates\n"
            "- Telling jokes\n"
            "- Write code based on a prompt\n"
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
        confidence_threshold = 0.6
        
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

        
#if the paths exist then its not redownloaded
csv_file_path = script_dir / 'data/ARGUS_data_training.csv'
conversation_df = pd.read_csv(csv_file_path)

#check the first few rows to verify the column names
print(conversation_df.head())

#CSV has columns named "input" and "response"
input_texts = conversation_df['input'].tolist()
target_texts = conversation_df['response'].tolist()

#create a list of (input, response) pairs
dataset = list(zip(input_texts, target_texts))
    
sampled_dataset = (dataset)
     
          

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
print("Built new tokenizer with vocab size:", vocab_size)
print("Dataset loaded with", len(dataset), "pairs.")



if __name__ == '__main__':
    audio_setup()
    
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
    epochsstart = 25 #old value was 20
    #start and train the chatbot
    chatbot = Chatbot(vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length)

    initial_learning_rate = 0.002 #orginal value of initial learning rate was 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1783, decay_rate=0.90, staircase=True) #orginal decay steps 100000 the decay steps value after using the equation is 634 with current amount of data
    chatbot.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    input_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_length, padding='post')
    #uncomment out below line with suflle method if porblem arrise that makes training more random with sequences
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)
    #suffle method not included in below line reduces randomness of training of sequences
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32) # old line that was being used
    
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
    
    
    
#GUI INITATION

    #start wake word listener in a background thread
    wake_word_thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    wake_word_thread.start()
    
    #create and display the PyQt application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
    
        
        
