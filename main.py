from src.ingest import ingest_docs
from src.embeddings import get_embedding
from src.store import search
from src.llm import generate
from config import TOP_K

docs = [
    "LangChain is a framework to build LLM applications.",
    "Vector databases store embeddings for semantic search.",
    "RAG augments prompts with retrieved context to improve answers.",
    "PostgreSQL with pgvector can store and search vectors efficiently."
]

if __name__ == "__main__":
    ingest_docs(docs)
    q = "How do apps use documents to answer questions?"
    qv = get_embedding(q)
    ctx = search(qv, k=TOP_K)
    context = "\n---\n".join(ctx)
    prompt = f"Use the context to answer.\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"
    answer = generate(prompt)
    print("Answer:\n", answer)