from src.embeddings import get_embedding
from src.store import add_doc

def ingest_docs(docs: list[str]) -> None:
    for d in docs:
        v = get_embedding(d)
        add_doc(d, v)