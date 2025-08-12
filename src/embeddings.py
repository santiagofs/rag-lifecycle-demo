import requests
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def get_embedding(text: str, model: str = "nomic-embed-text"):
    """
    Generate a vector embedding for the given text using Ollama.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text}
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    data = response.json()
    return data["embedding"]

if __name__ == "__main__":
    sample = "Hello, this is a test for embeddings."
    vec = get_embedding(sample)
    print(f"Vector length: {len(vec)}")
    print(vec[:5], "...")  # preview first 5 numbers