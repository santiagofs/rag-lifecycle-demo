# src/config.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

if PROVIDER == "ollama":
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
else:
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")