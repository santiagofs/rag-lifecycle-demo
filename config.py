# src/config.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

if PROVIDER == "ollama":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
else:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")

# Database Configuration
DB_PATH = os.getenv("DB_PATH", "./vectors.db")

# Embedding Model Configuration
EMBED_MODEL_DIGEST = os.getenv("EMBED_MODEL_DIGEST", "")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))
HYBRID = os.getenv("HYBRID", "false").lower() == "true"
