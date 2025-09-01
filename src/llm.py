import os
import requests

from config import BASE_URL, LLM_MODEL

def generate(prompt: str) -> str:
    try:
        r = requests.post(
            f"{BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"LLM request failed: {e}") from e
