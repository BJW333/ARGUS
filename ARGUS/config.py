import os
import json
from pathlib import Path
import spacy

nlp = spacy.load("en_core_web_md")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#new line below
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  #prevents performance issues on apple silicon
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#MY_TOKEN = "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"
#os.environ["HF_AUTH_TOKEN"] = MY_TOKEN

script_dir = Path(__file__).parent
config_path = script_dir / 'metrics/config.json'
#load config
with open(config_path) as config_file:
    config = json.load(config_file)

base_dir = Path.home()
wake_word_sound_path = base_dir / config['wake_word_sound']
hide_me_script_path = base_dir / config['hide_me_script']
password_checker_path = base_dir / config['password_checker']
spider_crawler_path = base_dir / config['spider_crawler']

MASTER = "Blake"