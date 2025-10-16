import os
import json
import spacy
import nltk
from pathlib import Path

#if nltk is missing this download it
nlp = spacy.load("en_core_web_md") #libarys for nlp/intent/reward

def ensure_vader_lexicon():
    """
    Ensure the VADER lexicon is downloaded
    
    VADER lexicon is used for nlp/intent/reward systems
    """
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        print("VADER lexicon already downloaded.")
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')
        print("VADER lexicon downloaded successfully.")

#Run the check
ensure_vader_lexicon()

#this is the main script directory path
script_dir = Path(__file__).parent

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#new line below
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  #prevents performance issues on apple silicon
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#load config
config_path = script_dir / 'config.json'
with open(config_path) as config_file:
    config = json.load(config_file)

base_dir = Path.home()
wake_word_sound_path = base_dir / config['wake_word_sound']
hide_me_script_path = base_dir / config['hide_me_script']
password_checker_path = base_dir / config['password_checker']
spider_crawler_path = base_dir / config['spider_crawler']

MASTER = "Blake"