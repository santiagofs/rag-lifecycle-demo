import numpy as np
from src.db import store

def add_doc(text: str, vector: list[float]) -> None:
    """Add document to SQLite store (backward compatibility)"""
    store.add_document(text, vector)

def search(vector: list[float], k: int = 3) -> list[str]:
    """Search documents using cosine similarity (backward compatibility)"""
    results = store.search_cosine(vector, k)
    return [r['text'] for r in results]