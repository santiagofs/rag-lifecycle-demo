import requests
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL

def get_embedding(text: str, model: str = None):
    """
    Generate a vector embedding for the given text using Ollama.
    """
    if model is None:
        model = EMBEDDING_MODEL

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]
    except requests.RequestException as e:
        raise RuntimeError(f"Embeddings request failed: {e}") from e

if __name__ == "__main__":
    sample = "Hello, this is a test for embeddings."
    vec = get_embedding(sample)
    print(f"Vector length: {len(vec)}")
    print(vec[:5], "...")  # preview first 5 numbers
