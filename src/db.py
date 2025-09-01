import sqlite3
import numpy as np
from typing import List, Dict, Optional
import json
from config import DB_PATH

class VectorStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.init_db()

    def init_db(self):
        """Initialize SQLite database with FTS5 and vector tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable FTS5 extension
            conn.execute("PRAGMA foreign_keys = ON")

            # Create documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create vectors table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents (id)
                )
            """)

            # Create FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                    text,
                    content='documents',
                    content_rowid='id'
                )
            """)

            # Create trigger to sync FTS5 with documents table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                    INSERT INTO docs_fts(rowid, text) VALUES (new.id, new.text);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, text) VALUES('delete', old.id, old.text);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON documents BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, text) VALUES('delete', old.id, old.text);
                    INSERT INTO docs_fts(rowid, text) VALUES (new.id, new.text);
                END
            """)

    def add_document(self, text: str, embedding: List[float], metadata: Optional[Dict] = None) -> int:
        """Add a document with its embedding to the store"""
        with sqlite3.connect(self.db_path) as conn:
            # Insert document
            cursor = conn.execute(
                "INSERT INTO documents (text, metadata) VALUES (?, ?)",
                (text, json.dumps(metadata) if metadata else None)
            )
            doc_id = cursor.lastrowid

            # Insert embedding
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO vectors (doc_id, embedding) VALUES (?, ?)",
                (doc_id, embedding_blob)
            )

            return doc_id

    def search_cosine(self, query_embedding: List[float], k: int = 3) -> List[Dict]:
        """Search documents using cosine similarity"""
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec) + 1e-9

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT d.id, d.text, d.metadata, v.embedding
                FROM documents d
                JOIN vectors v ON d.id = v.doc_id
            """)

            results = []
            for row in cursor.fetchall():
                doc_id, text, metadata, embedding_blob = row
                doc_vec = np.frombuffer(embedding_blob, dtype=np.float32)
                doc_norm = np.linalg.norm(doc_vec) + 1e-9

                similarity = (query_vec @ doc_vec) / (query_norm * doc_norm)
                results.append({
                    'id': doc_id,
                    'text': text,
                    'metadata': json.loads(metadata) if metadata else None,
                    'similarity': float(similarity)
                })

            # Sort by similarity and return top k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]

    def search_hybrid(self, query_embedding: List[float], query_text: str, k: int = 3,
                     cosine_weight: float = 0.7, fts_weight: float = 0.3) -> List[Dict]:
        """Hybrid search combining cosine similarity and FTS5"""
        # Get cosine results
        cosine_results = self.search_cosine(query_embedding, k * 2)  # Get more for reranking

        # Get FTS5 results
        fts_results = self.search_fts(query_text, k * 2)

        # Combine and rerank
        combined = {}

        # Add cosine results
        for i, result in enumerate(cosine_results):
            doc_id = result['id']
            combined[doc_id] = {
                'id': doc_id,
                'text': result['text'],
                'metadata': result['metadata'],
                'cosine_score': result['similarity'],
                'fts_score': 0.0,
                'combined_score': result['similarity'] * cosine_weight
            }

        # Add FTS results
        for i, result in enumerate(fts_results):
            doc_id = result['id']
            fts_score = 1.0 / (i + 1)  # Simple rank-based score

            if doc_id in combined:
                combined[doc_id]['fts_score'] = fts_score
                combined[doc_id]['combined_score'] += fts_score * fts_weight
            else:
                combined[doc_id] = {
                    'id': doc_id,
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'cosine_score': 0.0,
                    'fts_score': fts_score,
                    'combined_score': fts_score * fts_weight
                }

        # Sort by combined score and return top k
        results = list(combined.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]

    def search_fts(self, query_text: str, k: int = 3) -> List[Dict]:
        """Search documents using FTS5 full-text search"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT d.id, d.text, d.metadata, rank
                FROM documents d
                JOIN docs_fts fts ON d.id = fts.rowid
                WHERE docs_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query_text, k))

            results = []
            for row in cursor.fetchall():
                doc_id, text, metadata, rank = row
                results.append({
                    'id': doc_id,
                    'text': text,
                    'metadata': json.loads(metadata) if metadata else None,
                    'rank': rank
                })

            return results

    def get_document_count(self) -> int:
        """Get total number of documents"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

# Global instance
store = VectorStore()
