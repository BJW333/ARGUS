import logging
from datetime import datetime
from config_metrics.main_config import script_dir

log_file = script_dir / 'Metrics/chatbot_metrics.log'

if not log_file.parent.exists():
    log_file.parent.mkdir(parents=True)
    
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("ARGUS")

def log_metrics(user_input, bot_response, response_time, reward):
    """Logs structured chatbot metrics."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(
        f"{timestamp}, User Input: {user_input}, Bot Response: {bot_response}, "
        f"Response Time: {response_time}, Reward: {reward}"
    )

def log_debug(*args):
    """Logs any general debug message with multiple args."""
    message = " ".join(str(arg) for arg in args)
    logger.debug(message) 
   