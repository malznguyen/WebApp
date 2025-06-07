import os
from dotenv import load_dotenv

load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
WINDOW_WIDTH = int(os.getenv("WINDOW_WIDTH", 1200)) 
WINDOW_HEIGHT = int(os.getenv("WINDOW_HEIGHT", 800))
MIN_WINDOW_WIDTH = int(os.getenv("MIN_WINDOW_WIDTH", 800))
MIN_WINDOW_HEIGHT = int(os.getenv("MIN_WINDOW_HEIGHT", 600))

VERTICAL_SPLITTER_RATIO = [600, 200]
HORIZONTAL_SPLITTER_RATIO = [300, 900]

MAX_LOG_LINES = int(os.getenv("MAX_LOG_LINES", 1000))
