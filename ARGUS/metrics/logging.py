import logging
from datetime import datetime
from config import script_dir

log_file = script_dir / 'Metrics/chatbot_metrics.log'

logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    filemode='a',
    format='%(name)s - %(levelname)s - %(message)s'
)

def log_metrics(user_input, bot_response, response_time, reward):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(
        f"{timestamp}, User Input: {user_input}, Bot Response: {bot_response}, "
        f"Response Time: {response_time}, Reward: {reward}"
    )