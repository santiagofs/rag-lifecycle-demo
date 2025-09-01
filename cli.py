#!/usr/bin/env python3
import argparse
import json
from src.embeddings import get_embedding
from src.db import store

def ingest_documents(docs, metadata=None):
    """Ingest a list of documents into the vector store"""
    print(f"Ingesting {len(docs)} documents...")

    for i, doc in enumerate(docs):
        try:
            embedding = get_embedding(doc)
            doc_metadata = metadata[i] if metadata and i < len(metadata) else None
            doc_id = store.add_document(doc, embedding, doc_metadata)
            print(f"✓ Document {i+1} ingested with ID {doc_id}")
        except Exception as e:
            print(f"✗ Failed to ingest document {i+1}: {e}")

    stats = store.get_stats()
    print(f"\nTotal documents in store: {stats['documents']}")
    print(f"Total vectors: {stats['vectors']}")

def main():
    parser = argparse.ArgumentParser(description="CLI for document ingestion")
    parser.add_argument("--docs", nargs="+", help="Documents to ingest")
    parser.add_argument("--file", help="JSON file containing documents")
    parser.add_argument("--metadata", help="JSON file containing metadata")

    args = parser.parse_args()

    if args.docs:
        ingest_documents(args.docs)
    elif args.file:
        with open(args.file, 'r') as f:
            data = json.load(f)
            docs = data.get('documents', [])
            metadata = data.get('metadata', None)
            ingest_documents(docs, metadata)
    else:
        # Default sample documents
        sample_docs = [
            "LangChain is a framework to build LLM applications.",
            "Vector databases store embeddings for semantic search.",
            "RAG augments prompts with retrieved context to improve answers.",
            "PostgreSQL with pgvector can store and search vectors efficiently.",
            "SQLite with FTS5 provides full-text search capabilities.",
            "Cosine similarity measures the angle between two vectors.",
            "Embeddings represent text as high-dimensional vectors.",
            "Retrieval-augmented generation combines search with language models."
        ]
        print("No documents provided, using sample documents...")
        ingest_documents(sample_docs)

if __name__ == "__main__":
    main()
