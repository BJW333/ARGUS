from config import script_dir
from nlp.chatbot import Chatbot
from nlp.reward import DynamicRewardSystem
from datafunc.data_store import DataStore
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import pickle
import os
from gui.animated_canvas import audio_setup


def initialize_chatbot_components():
    #load conv data
    #conversation_history = []
    data_store = DataStore(script_dir / 'conversation_history.json')
    conversation_history = data_store.load_data()
    
    #if the paths exist then its not redownloaded
    csv_file_path = script_dir / 'trainingdata/ARGUS_data_training.csv'
    conversation_df = pd.read_csv(csv_file_path)

    #check the first few rows to verify the column names
    #print(conversation_df.head()) #debugging line makes display not pretty

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
    
    audio_setup() # not thousand percent sure if commenting this out is right
    
    
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
    #data_store = DataStore(script_dir / 'conversation_history.json')
    
    return chatbot, reward_system, data_store, conversation_history, sampled_dataset, train_dataset, vocab_size, start_token, end_token