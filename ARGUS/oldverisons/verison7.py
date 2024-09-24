import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random
import requests
import zipfile
import io
import wikipedia
import json
import string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import pipeline
import pickle
import logging
from datetime import datetime
import openai
import time
import random
import pyaudio
import tkinter as tk
import numpy as np
import threading
import subprocess
import webbrowser
import os
import speech_recognition as sr
import requests
from gtts import gTTS
from playsound import playsound
from bs4 import BeautifulSoup
import wikipedia
import pyjokes
from pathlib import Path
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

script_dir = Path(__file__).parent
config_path = script_dir / 'config.json'

# Load configuration
with open(config_path) as config_file:
    config = json.load(config_file)

base_dir = Path.home()
wake_word_sound_path = base_dir / config['wake_word_sound']
hide_me_script_path = base_dir / config['hide_me_script']
password_checker_path = base_dir / config['password_checker']
spider_crawler_path = base_dir / config['spider_crawler']



MASTER = "Blake"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.system('clear')
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
os.system('clear')

openai.api_key = 'sk-proj-uab2gfjbJX2kaAp5EwA4T3BlbkFJEuSrWTw98Y8xjMn4YlWA'

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
    global p, stream, CHUNK, RATE
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

def update_circle():
    global stream, CHUNK, canvas, circle_center, circles, is_listening
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
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
    global stream, p, master
    stream.stop_stream()
    stream.close()
    p.terminate()
    master.destroy()

def wishme():
    hour = int(datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning " + MASTER)
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon " + MASTER)
    else:
        speak("Good Evening " + MASTER)

print("----------------------------")
print("----- Starting up Josh -----")
print("----------------------------")
#wishme()

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
        
data_store = DataStore('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/conversation_history.json')
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
logging.basicConfig(level=logging.INFO, filename='chatbot_metrics.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

def log_metrics(user_input, bot_response, response_time, reward):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp}, User Input: {user_input}, Bot Response: {bot_response}, Response Time: {response_time}, Reward: {reward}")

def collect_human_feedback(conversation_history):
    feedback_data = []
    for conversation in conversation_history:
        if isinstance(conversation, dict):
            user_input = conversation.get('user_input', '')
            bot_response = conversation.get('bot_response', '')
            print("User:", user_input)
            print("Bot:", bot_response)
            corrected_response = input("Correct the bot's response if needed or press Enter to keep it: ")
            feedback_data.append((user_input, corrected_response if corrected_response else bot_response))
        elif isinstance(conversation, tuple):
            user_input, bot_response = conversation
            print("User:", user_input)
            print("Bot:", bot_response)
            corrected_response = input("Correct the bot's response if needed or press Enter to keep it: ")
            feedback_data.append((user_input, corrected_response if corrected_response else bot_response))
        else:
            print("Skipping invalid conversation:", conversation)
    return feedback_data


def train_with_feedback(chatbot, original_dataset, feedback_data, conversation_history, epochs):
    #combine the original dataset feedback data and conversation history
    combined_dataset = original_dataset + feedback_data + conversation_history

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
    
class DynamicRewardSystem:
    def __init__(self):
        self.reward_score = 0
        
    def check_response_relevance(self, user_input, bot_response):
        messages = [
            {"role": "system", "content": "You are a response classifying assistant."}, #"You are a helpful assistant."
            {"role": "user", "content": f"""
            Consider a simple conversational interaction between a user and a chatbot. The user says: "{user_input}". The chatbot responds: "{bot_response}".

            Based on this interaction, evaluate the chatbot's response according to the following criteria:
            1. Does it make sense within the context of a simple conversational interaction?
            2. Is it something a simple chatbot would likely say in response to the user input?
            3. Does the response directly address the user's input in a meaningful way?

            Please provide a simple "yes" if the response meets all three criteria, or "no" if it fails to meet any of the criteria. Additionally, if the response is somewhat relevant but lacks detail or clarity, please indicate this by saying "was not detailed enough" and briefly explain why.
            """}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  #using the chat model gpt
                messages=messages,
                max_tokens=50
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return exit()

        answer = response.choices[0].message['content'].strip()

        #the response for relevance or not
        if "yes" in answer.lower():
            print(answer)
            print('')
            return True
        #else:
            #one line below is new
        
        if "no" in answer.lower(): # or "was not detailed enough" in answer.lower():
            print(answer)
            print('')
            return False
        
        #three lines below are new, moved the else statment from above if issue uncoment else and indent the 4 lines below it also changed the if to elif in the above line where old else was
        else:
            print(f"An error occurred")
            return exit()
              
    def evaluate_response(self, user_input, bot_response):
        #get sentiment scores
        user_sentiment = sentiment_analyzer(user_input)[0]
        bot_sentiment = sentiment_analyzer(bot_response)[0]

        #this line is new
        is_response_relevant = self.check_response_relevance(user_input, bot_response)  #this function needs to be defined

        #determine if the bot's sentiment aligns with the user's sentiment
        if user_sentiment['label'] == 'POSITIVE' and bot_sentiment['label'] == 'POSITIVE':
            if is_response_relevant == True: #new line delte if problem
            #if is_response_relevant:
            #if user_sentiment['label'] == 'POSITIVE' and bot_sentiment['label'] == 'POSITIVE':
                self.reward_score += 1
                print('The statement is relevent postive')
                return 1
            #this needs to work doffrenlently it needs to make sure both relv and confi are correct and then reward or if is_response_relevant or confidence >= 0.185:

        #if
        elif user_sentiment['label'] == 'NEGATIVE' and bot_sentiment['label'] == 'NEGATIVE':
            if is_response_relevant == True: #new line delte if problem
            #if is_response_relevant:
            #if user_sentiment['label'] == 'NEGATIVE' and bot_sentiment['label'] == 'NEGATIVE':
                self.reward_score += 1
                print('The statement is relevent negative')
                return 1
        
        #this down below is new delte if issue 
        elif is_response_relevant == True:
                self.reward_score += 1
                print('The statement is relevent and makes sense')
                return 1    

        #else:
        elif is_response_relevant == False: 
            # Penalize the bot if the sentiments do not align
            #if is_response_relevant == False: 
            if self.reward_score >= 0 or self.reward_score < 0:
                #if self.reward_score >= 0:     
                self.reward_score -= 1 # Negative reward   
                print("The statement is not good return false")
                return -1
        #return 0
    
    

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
        self.model.save_weights('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/model_weights.h5')
        #if issue use the save weights line
        #self.model.save('/Users/blakeweiss/Desktop/hello/model_weights', save_format="tf")

         
    def load_model(self):
        if os.path.exists('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/model_weights.h5'):
            #recreate the model with the known vocab size
            self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)

            #dummy call to initialize the variables
            dummy_input = [tf.zeros((1, 1)), tf.zeros((1, 1))]
            self.model(dummy_input)

            #Load the weights
            self.model.load_weights('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/model_weights.h5')

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

def getPerson(user_input):

    wordList = user_input.lower().split()

    for i in range(0, len(wordList)):
        if i + 3 <= len(wordList) - 1 and wordList[i].lower() == 'who' and wordList[i+1].lower() == 'is':
            return wordList[i+2] + ' ' + wordList[i+3]    
        
def recognize_speech():
    global is_thinking
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        #print("Taking input")
        #audio = recognizer.listen(source)
        #is_thinking = True
        try:
            print("Taking input")
            audio = recognizer.listen(source)
            is_thinking = True
            recognized_text = recognizer.recognize_google(audio, language='en-us')
            is_thinking = False
            print("You said:", recognized_text)
            #process_user_input(recognized_text)
            return recognized_text
        except sr.UnknownValueError:
            is_thinking = False
            print("Could not understand the audio, please try again.")
            return None
        except sr.RequestError as e:
            is_thinking = False
            print(f"Error with the request; {e}")
            return None

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

   
def listen_for_wake_word():
    global is_thinking
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening for wake word...")
        while True:
            audio = recognizer.listen(source)
            is_thinking = True
            try:
                intro = recognizer.recognize_google(audio, language='en-us')
                is_thinking = False
                print("Me  --> ", intro)
                if "Josh" in intro:
                    handle_wake_word_detected()
            except sr.UnknownValueError:
                is_thinking = False
            except sr.RequestError as e:
                is_thinking = False
                print(f"Error with the request; {e}")
        
        
def handle_wake_word_detected():
    #threading.Thread(target=recognize_speech).start()
    user_input = recognize_speech()
    #print("Recognized text complete") debugging
    if user_input:
        #print("Processing input") debugging
        process_user_input(user_input)
    listen_for_wake_word()

def process_user_input(user_input):
    #print(f"Processing user input: {user_input}") debugging
    if user_input.lower() == 'exit':
        chatbot.save_model()
        print("Vocabulary Size:", vocab_size)
        root.quit()
        
    elif "are you there" in user_input.lower():
                speak(f"I'm here {MASTER}")
                    
    elif "what does Josh stand for" in user_input.lower():
        speak("Josh stands for Just Ordinary Selfless Helper")
        
    elif "hide me" in user_input.lower():
        os.system(f"bash {hide_me_script_path}")
        speak("Hiding your IP with Tor")
    
    elif user_input.lower() in ["i need to manually adjust some things with the model", "adjust the model", "manually mantain the model"]:  
        print('Please enter the command of whatver needs to be done in regards to the model')
        speak('Please enter the command of whatver needs to be done in regards to the model')
        print('1. feedback')
        print('2. train')
        print('3. save')
        manualinput = input("Just enter the word associated with what needs to be done: ")
    #usage of training with feedback and historical data
        if manualinput.lower() == 'feedback':
            feedback_data = collect_human_feedback(conversation_history)
            epochs3 = 10
            train_with_feedback(chatbot, sampled_dataset, feedback_data, conversation_history, epochs3)
#clear the history after using it for training
        elif manualinput.lower() == 'train':
            epochtotrain = input("Number of epochs to train by: ")
            epochtotrain = int(epochtotrain)
            #train the model
            chatbot.modeltrain(train_dataset, epochtotrain)   
        elif manualinput.lower() == 'save':
            chatbot.save_model()
            with open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/tokenizer.pickle', 'wb') as handle:
                pickle.dump(chatbot.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif "tell me a joke" in user_input.lower():
            speak(pyjokes.get_joke())     
                
    elif "movie" in user_input.lower():
            goodmovies = ["Star Wars", "Jurassic Park", "Clear and Present Danger", "War Dogs", "Wolf of Wall Street", "The Big Short", "Trading Places", "The Gentlemen", "Ferris Bueller's Day Off", "Goodfellas", "Lord of War", "Borat", "Marvel movies", "The Hurt Locker", "Hustle", "Forrest Gump", "Darkest Hour", "Coming to America", "Warren Miller movies", "The Dictator"]
            moviechoice = random.choice(goodmovies)
            speak(f"A good movie you could watch is {moviechoice}, {MASTER}")
            
    elif "what are your skills" in user_input.lower():
        skillsforuse = (
            "-Hi, I am Josh. I can perform various tasks, including:\n"
            "- Searching on Google\n"
            "- Opening apps\n"
            "- Telling you the date and time\n"
            "- Running chat conversations\n"
            "- Running 'hide me'\n"
            "- Identifying people\n"
            "- Searching for information\n"
            "- Providing news updates\n"
            "- Telling jokes"
        )
        speak(skillsforuse)
        
    elif "open" in user_input.lower():
        app_name = user_input.split("open ")[1]
        print(f"Opening app: {app_name}")  # Debug statement
        open_app(app_name)
        
    elif "close" in user_input.lower():
        app_name = user_input.split("close ")[1]
        print(f"Closing app: {app_name}")  # Debug statement
        close_application(app_name)        

    elif user_input.lower() in ["news", "new", "tell me today's news", "tell me the news", "whats the news", "whats the news today", "whats happening in the world"]:
        print("Bot: Here's what's happening in the news:")
        speak("Bot: Here's what's happening in the news:")
        news_today_recent = get_the_news()
        for category, headlines in news_today_recent.items():
            print(f"\n{category} News:")
            for i, headline in enumerate(headlines, 1):
                print(f"{i}. {headline}")
                
    elif "time" in user_input.lower():
        timeofday = action_time()    
        print(timeofday)
        speak(timeofday)
        
    
        
    elif "spider crawler" in user_input.lower():
        print("Starting up SpiderCrawler")
        speak("Starting up SpiderCrawler")
        os.system(f"python3.10 {spider_crawler_path}")
        
    elif user_input.lower() in ["run password checker", "password checker", "can you check if my password has been breached", "I think my password is comprimised"]:
        print("lets check if your password is compromised")
        speak("lets check if your password is compromised")
        os.system(f"python3.10 {password_checker_path}")
        
    elif "search" in user_input.lower():
        speak(f"What can I search for you, {MASTER}?")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            speak("Just state what you want to search, sir.")
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            searchaudio = recognizer.listen(source)
        try:
            searchstuff = recognizer.recognize_google(searchaudio, language='en-us')
            print("Me  --> ", searchstuff)
            webbrowser.open('https://www.google.com/search?q=' + searchstuff)
        except:
            print("Me  -->  ERROR")
                    
    elif "who is" in user_input.lower():
        person = getPerson(user_input)
        wiki = wikipedia.summary(person, sentences = 2)
        responsewhois = ("Bot: This is") + ' ' + wiki
        response = responsewhois
        print(responsewhois)    
        speak(responsewhois)    

    elif user_input.lower() == 'save json to text':
        #This gets the input and response from json file and puts into a txt so its possible to go into input txt and output txt
        conversation_jsontotxt = data_store.load_data()
        text_data = json_to_text(conversation_jsontotxt)
        file_pathforjsontotxt = '/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/conversation_datajsontotxt.txt'

        with open(file_pathforjsontotxt, 'w') as file:
            file.write(text_data)
    
    else:
        start_time = datetime.now()
        response, confidence = chatbot.generate_seq2seqresponse(user_input, temperature=0.6)# temp was 0.5
        
        is_response_relevant = reward_system.check_response_relevance(user_input, response)

        response_printed = False
        confidence_threshold = 0.165
        #check if the response's confidence is below the threshold
        if confidence < confidence_threshold or not is_response_relevant:
            print("")
            print("Bot (Uncertain):", response)
            
            #new line below testing see what it does
            speak(f"Bot: {response}")

            print("")                    
            speak("This response is flagged for human review correct the response.")
            print("This response is flagged for human review correct the response.")
            corrected_response = input("Please provide the correct response or press Enter to accept as is: ")
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
            
        reward = reward_system.evaluate_response(user_input, response)
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        #log information
        log_metrics(user_input, response, response_time, reward)
                    
        if not response_printed:
            print("Bot:", response.replace(start_token, '').replace(end_token, ''))
            speak("Bot:", response.replace(start_token, '').replace(end_token, ''))
        print('')
        print("Reward for this response:", reward)
        print("Total reward:", reward_system.get_total_reward())   
        print("Confidence:", confidence)
        print("Response time:", response_time) #amount of time it takes for bot to respond
        print('')
        
        
        conversation_history.append((user_input, response))
        
        conversation_data = {
            'user_input': user_input,
            'bot_response': response,
            'reward': reward,  #reward mechanism
            'flagged_for_retraining': flagged_for_retraining  #flag for retraining based on checking system
        }
        data_store.save_data(conversation_data)  
    #pass
        
                        
#if the paths exist then its not redownloaded
if os.path.exists("/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/inputtexts.txt") and os.path.exists("/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/outputtexts.txt"):
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
         open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/inputtexts.txt', 'w', encoding='utf-8') as conv_file, \
         open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/outputtexts.txt', 'w', encoding='utf-8') as ans_file:
        for line in f:
            parts = line.split(' +++$+++ ')
            conversation = eval(parts[-1].strip())
            for i in range(len(conversation) - 1):
                input_dialogue = dialogues[conversation[i]]
                target_dialogue = dialogues[conversation[i + 1]]
                conv_file.write(input_dialogue + '\n')
                ans_file.write(target_dialogue + '\n')

with open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/inputtexts.txt', 'r', encoding='utf-8') as conv_file:
    input_texts = conv_file.readlines()
    

with open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/data/outputtexts.txt', 'r', encoding='utf-8') as ans_file:
    target_texts = ans_file.readlines()
    

dataset = list(zip(input_texts, target_texts))

#sample a subset of the dataset
#sample_size = 200
sampled_dataset = (dataset)
#load tokenizer and get vocab size

if os.path.exists('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/tokenizer.pickle'):
    with open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        vocab_size = tokenizer.vocab_size
else:
    input_texts, target_texts = zip(*sampled_dataset)  #use sampled dataset here
    all_texts = input_texts + target_texts
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (text for text in all_texts), target_vocab_size=2**13)
    vocab_size = tokenizer.vocab_size
    with open('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

reward_system = DynamicRewardSystem()




if __name__ == '__main__':
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
    if os.path.exists('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/model_weights.h5'):
        chatbot.load_model()
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
    else:
        chatbot.modeltrain(train_dataset, epochsstart)
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
        chatbot.save_model()

    

    #conversation_history = []
    data_store = DataStore('/Users/blakeweiss/Desktop/NEWJOSHWITHAI/conversation_history.json')
    
    root = tk.Tk()
    master = root
    master.title("JOSH")
    master.geometry("600x600")
    master.configure(bg='black')

    canvas = tk.Canvas(master, width=600, height=600, bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    create_circle()
    audio_setup()
    update_circle()

    #wake_word_thread = threading.Thread(target=listen_for_wake_word)
    #wake_word_thread.daemon = True
    #wake_word_thread.start()    
    threading.Thread(target=listen_for_wake_word, daemon=True).start()

    thinking_animation()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

                
            

        
        
        
