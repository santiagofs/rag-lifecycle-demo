#!/usr/bin/env python3
"""
Enhanced Ingestion Pipeline CLI
Chunk→embed→store in SQLite with deterministic IDs and duplicate detection.
Supports .txt, .md, .html, .pdf files.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings import get_embedding
from src.db import store
from src.loaders import get_loader_for_file, DocumentBlock
from src.utils.ids import compute_doc_id, compute_emb_id
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME, EMBED_MODEL_DIGEST


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """Split text into overlapping chunks"""
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at word boundaries
        if end < len(text):
            # Look for last space in chunk
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.5:  # Only break if we find a space in the latter half
                chunk = chunk[:last_space]
                end = start + last_space + 1

        chunks.append(chunk.strip())
        start = end - chunk_overlap

        if start >= len(text):
            break

    return [chunk for chunk in chunks if chunk.strip()]


def read_text_file(file_path: str) -> str:
    """Read text file with proper encoding detection"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def load_document_blocks(file_path: str) -> List[DocumentBlock]:
    """Load document blocks using appropriate loader"""
    try:
        loader = get_loader_for_file(file_path)
        return loader(file_path)
    except Exception as e:
        print(f"   ⚠️  Failed to load with specialized loader: {e}")
        # Fallback to text loader
        return [DocumentBlock(
            text=read_text_file(file_path),
            metadata={'source': file_path, 'section': 'fallback_text'}
        )]


def ingest_file(file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> Dict:
    """Ingest a single file with chunking and duplicate detection"""
    print(f"📄 Processing: {file_path}")

    # Load document blocks
    try:
        blocks = load_document_blocks(file_path)
        print(f"   📖 Loaded {len(blocks)} blocks")
    except Exception as e:
        print(f"   ❌ Failed to load file: {e}")
        return {'error': str(e)}

    # Process blocks and chunks
    stats = {
        'file': file_path,
        'total_blocks': len(blocks),
        'total_chunks': 0,
        'new_documents': 0,
        'new_embeddings': 0,
        'skipped_documents': 0,
        'skipped_embeddings': 0,
        'errors': 0
    }

    # Use single transaction for all chunks from this file
    with store.get_connection() as conn:
        for block_idx, block in enumerate(blocks):
            # Chunk the block text
            chunks = chunk_text(block.text, chunk_size, chunk_overlap)
            stats['total_chunks'] += len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Compute deterministic IDs
                    doc_id = compute_doc_id(chunk)
                    emb_id = compute_emb_id(doc_id, EMBED_MODEL_NAME)

                    # Merge block metadata with chunk metadata
                    chunk_metadata = {
                        **block.metadata,
                        'chunk_index': chunk_idx,
                        'block_index': block_idx
                    }

                    # Check if document already exists
                    cursor = conn.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
                    doc_exists = cursor.fetchone() is not None

                    if not doc_exists:
                        # Insert document
                        conn.execute(
                            "INSERT OR IGNORE INTO documents (id, text, metadata) VALUES (?, ?, ?)",
                            (doc_id, chunk, json.dumps(chunk_metadata))
                        )
                        stats['new_documents'] += 1
                        print(f"   ✅ Block {block_idx+1}, Chunk {chunk_idx+1}: New document {doc_id[:8]}...")
                    else:
                        stats['skipped_documents'] += 1
                        print(f"   ⏭️  Block {block_idx+1}, Chunk {chunk_idx+1}: Skipped existing document {doc_id[:8]}...")

                    # Check if embedding already exists
                    cursor = conn.execute("SELECT id FROM vectors WHERE id = ?", (emb_id,))
                    emb_exists = cursor.fetchone() is not None

                    if not emb_exists:
                        # Get embedding
                        embedding = get_embedding(chunk)

                        # Compute norm
                        vec_array = np.array(embedding, dtype=np.float32)
                        norm = float(np.linalg.norm(vec_array))

                        # Insert embedding
                        embedding_blob = vec_array.tobytes()
                        conn.execute("""
                            INSERT OR IGNORE INTO vectors (id, doc_id, vec, norm, model, digest, dim)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (emb_id, doc_id, embedding_blob, norm, EMBED_MODEL_NAME, EMBED_MODEL_DIGEST, len(embedding)))

                        stats['new_embeddings'] += 1
                        print(f"   🧠 Block {block_idx+1}, Chunk {chunk_idx+1}: New embedding {emb_id[:8]}...")
                    else:
                        stats['skipped_embeddings'] += 1
                        print(f"   ⏭️  Block {block_idx+1}, Chunk {chunk_idx+1}: Skipped existing embedding {emb_id[:8]}...")

                except Exception as e:
                    stats['errors'] += 1
                    print(f"   ❌ Block {block_idx+1}, Chunk {chunk_idx+1}: Error - {e}")

        conn.commit()

    return stats


def ingest_directory(dir_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Dict]:
    """Ingest all supported files in a directory"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"❌ Not a directory: {dir_path}")
        return []

    # Find all supported files
    supported_extensions = {'.txt', '.md', '.html', '.pdf'}
    files = []
    for ext in supported_extensions:
        files.extend(dir_path.glob(f"*{ext}"))

    if not files:
        print(f"❌ No supported files found in {dir_path}")
        print(f"   Supported extensions: {', '.join(supported_extensions)}")
        return []

    print(f"📁 Found {len(files)} supported files in {dir_path}")

    results = []
    for file_path in files:
        result = ingest_file(str(file_path), chunk_size, chunk_overlap)
        results.append(result)
        print()  # Empty line between files

    return results


def main():
    """CLI for enhanced document ingestion"""
    parser = argparse.ArgumentParser(description="Enhanced document ingestion with chunking and duplicate detection")
    parser.add_argument("--path", required=True, help="File path or directory to ingest")
    parser.add_argument("--chunk-size", type=int, help="Chunk size (default: from config)")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap (default: from config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        sys.exit(1)

    print(f"🚀 Starting ingestion pipeline")
    print(f"   Model: {EMBED_MODEL_NAME}")
    print(f"   Chunk size: {args.chunk_size or CHUNK_SIZE}")
    print(f"   Chunk overlap: {args.chunk_overlap or CHUNK_OVERLAP}")
    print()

    if path.is_file():
        if path.suffix.lower() not in {'.txt', '.md', '.html', '.pdf'}:
            print(f"❌ Unsupported file type: {path}")
            print(f"   Supported extensions: .txt, .md, .html, .pdf")
            sys.exit(1)

        result = ingest_file(str(path), args.chunk_size, args.chunk_overlap)
        results = [result]
    else:
        results = ingest_directory(str(path), args.chunk_size, args.chunk_overlap)

    # Summary
    print("📊 Ingestion Summary:")
    total_blocks = sum(r.get('total_blocks', 0) for r in results)
    total_chunks = sum(r.get('total_chunks', 0) for r in results)
    total_new_docs = sum(r.get('new_documents', 0) for r in results)
    total_new_embs = sum(r.get('new_embeddings', 0) for r in results)
    total_skipped_docs = sum(r.get('skipped_documents', 0) for r in results)
    total_skipped_embs = sum(r.get('skipped_embeddings', 0) for r in results)
    total_errors = sum(r.get('errors', 0) for r in results)

    print(f"   📄 Files processed: {len(results)}")
    print(f"   📦 Blocks processed: {total_blocks}")
    print(f"   ✂️  Chunks created: {total_chunks}")
    print(f"   ✅ New documents: {total_new_docs}")
    print(f"   🧠 New embeddings: {total_new_embs}")
    print(f"   ⏭️  Skipped documents: {total_skipped_docs}")
    print(f"   ⏭️  Skipped embeddings: {total_skipped_embs}")
    print(f"   ❌ Errors: {total_errors}")

    # Show database stats
    db_stats = store.get_stats()
    print(f"\n💾 Database Stats:")
    print(f"   Documents: {db_stats['documents']}")
    print(f"   Vectors: {db_stats['vectors']}")
    print(f"   Database: {db_stats['db_path']}")


if __name__ == "__main__":
    main()
