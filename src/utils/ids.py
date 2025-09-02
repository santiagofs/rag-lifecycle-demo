"""
Deterministic ID utilities for document and embedding identification.
"""

import hashlib
from typing import List


def compute_doc_id(text: str) -> str:
    """Compute deterministic document ID from text content"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def compute_emb_id(doc_id: str, model: str) -> str:
    """Compute deterministic embedding ID from document ID and model"""
    return hashlib.sha256(f"{doc_id}:{model}".encode('utf-8')).hexdigest()[:16]
