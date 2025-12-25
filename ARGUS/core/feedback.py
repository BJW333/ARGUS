import tensorflow as tf
from datafunc.data_store import json_to_text

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


#the below function is no longer used due to using llama3 as the new model of choice 
#until I have enough money to train RoMaLM into a fully 7 billion + parameter model

# def train_with_feedback(chatbot, sampled_dataset, feedback_data, epochs): #could include sampled_dataset as a variable so that the main dataset is also in the feedback data
#     #combine the original dataset feedback data and conversation history
#     combined_dataset = sampled_dataset + feedback_data #could include sampled_dataset so that the main dataset is also in the feedback data you would replace orginal dataset with that variable
#     #combined_dataset = feedback_data #just feedback data
#     print(combined_dataset)
#     #preprocess the combined dataset as you did with the original dataset
#     input_texts, target_texts = zip(*combined_dataset)
#     input_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in input_texts]
#     target_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in target_texts]
#     input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=chatbot.max_length, padding='post')
#     target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=chatbot.max_length, padding='post')
    
#     train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)  #shuffle so the model generalizes better
#     #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32)  #this is the old line without shuffle

#     #train the model
#     chatbot.modeltrain(train_dataset, epochs)