from config import script_dir
from nlp.chatbot import Chatbot
from nlp.reward import DynamicRewardSystem
from datafunc.data_store import DataStore
from gui.animated_canvas import audio_setup


def initialize_chatbot_components():
    #load conv data
    #conversation_history = []
    data_store = DataStore(script_dir / 'conversation_history.json')
    conversation_history = data_store.load_data()
    
    audio_setup() # not thousand percent sure if commenting this out is right
    
    # Initialize the chatbot and reward system
    chatbot = Chatbot()  # Initialize the chatbot with default parameters
    reward_system = DynamicRewardSystem()

    return chatbot, reward_system, data_store, conversation_history