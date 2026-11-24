from openai import OpenAI
from utils.config_tool import agent_conf
import os

base_client = OpenAI(api_key=os.getenv("API_KEY_qwen"), base_url=agent_conf["base_url"])

def get_base_client():
    return base_client