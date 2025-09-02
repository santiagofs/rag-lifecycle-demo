import numpy as np
from src.db import store
from config import TOP_K, HYBRID
from typing import List

def add_doc(text: str, vector: List[float]) -> None:
    """Add document to SQLite store (backward compatibility)"""
    store.add_document(text, vector)

def search(vector: List[float], k: int = None) -> List[str]:
    """Search documents using cosine similarity (backward compatibility)"""
    if k is None:
        k = TOP_K

    if HYBRID:
        # For hybrid search, we need the query text, but this function only has vector
        # So we fall back to cosine search for backward compatibility
        results = store.search_cosine(vector, k)
    else:
        results = store.search_cosine(vector, k)

    return [r['text'] for r in results]