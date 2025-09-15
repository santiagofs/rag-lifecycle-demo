from src.embeddings import get_embedding
from src.store import add_doc
from typing import List

def ingest_docs(docs: List[str]) -> None:
    for d in docs:
        v = get_embedding(d)
        add_doc(d, v)