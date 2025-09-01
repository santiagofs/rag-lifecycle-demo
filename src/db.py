import sqlite3
import numpy as np
import argparse
import sys
import os
import hashlib
from typing import List, Dict, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, EMBED_MODEL_NAME, EMBED_MODEL_DIGEST

class VectorStore:
    def __init__(self, db_path: str = None, auto_init: bool = True):
        self.db_path = db_path or DB_PATH
        if auto_init:
            self.init_db()

    def get_connection(self):
        """Get SQLite connection with optimized PRAGMAs"""
        conn = sqlite3.connect(self.db_path)

        # Optimize for performance and reliability
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous = NORMAL")  # Good balance of speed/safety
        conn.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
        conn.execute("PRAGMA foreign_keys = ON")  # Enforce foreign key constraints
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory mapping

        return conn

    def init_db(self):
        """Initialize SQLite database with FTS5 and vector tables"""
        with self.get_connection() as conn:
            # Create documents table with deterministic IDs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,  -- Deterministic hash-based ID
                    text TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create vectors table with model tracking and pre-computed norms
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,  -- Deterministic hash-based ID
                    doc_id TEXT NOT NULL,
                    model TEXT NOT NULL,  -- Embedding model name
                    digest TEXT,  -- Model digest/version
                    dim INTEGER NOT NULL,  -- Vector dimensions
                    vec BLOB NOT NULL,  -- Vector data as float32 blob
                    norm REAL,  -- Pre-computed L2 norm for cosine similarity
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # Create FTS5 virtual table for full-text search (using internal rowid)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                    text,
                    content='documents',
                    content_rowid='rowid'
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_doc_id ON vectors(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)")

            # Create trigger to sync FTS5 with documents table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                    INSERT INTO docs_fts(rowid, text) VALUES (new.rowid, new.text);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON documents BEGIN
                    INSERT INTO docs_fts(docs_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                    INSERT INTO docs_fts(rowid, text) VALUES (new.rowid, new.text);
                END
            """)

    def add_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
        model: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> str:
        """Add a document with its embedding to the store (deterministic IDs).

        - Documents use a hash-based TEXT primary key.
        - Vectors store model/digest/dim and a precomputed norm.
        Returns the document id (string).
        """

        # Defaults from config
        model = model or EMBED_MODEL_NAME
        digest = digest or EMBED_MODEL_DIGEST

        # Deterministic IDs
        doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        vec_array = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec_array))
        dim = int(vec_array.size)
        embedding_blob = vec_array.tobytes()

        emb_id = hashlib.sha256(f"{doc_id}:{model}".encode("utf-8")).hexdigest()[:16]

        with self.get_connection() as conn:
            # Insert or ignore document
            conn.execute(
                "INSERT OR IGNORE INTO documents (id, text, metadata) VALUES (?, ?, ?)",
                (doc_id, text, json.dumps(metadata) if metadata else None),
            )

            # Insert or ignore embedding row aligned with new schema
            conn.execute(
                """
                INSERT OR IGNORE INTO vectors (id, doc_id, vec, norm, model, digest, dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (emb_id, doc_id, embedding_blob, norm, model, digest, dim),
            )

        return doc_id

    def search_cosine(self, query_embedding: List[float], k: int = 3, model: str = None) -> List[Dict]:
        """Search documents using cosine similarity with optional model filtering"""
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec) + 1e-9

        with self.get_connection() as conn:
            # Build query with optional model filter
            if model:
                sql = """
                    SELECT d.id, d.text, d.metadata, v.vec, v.norm
                    FROM documents d
                    JOIN vectors v ON d.id = v.doc_id
                    WHERE v.model = ?
                """
                cursor = conn.execute(sql, (model,))
            else:
                sql = """
                    SELECT d.id, d.text, d.metadata, v.vec, v.norm
                    FROM documents d
                    JOIN vectors v ON d.id = v.doc_id
                """
                cursor = conn.execute(sql)

            results = []
            for row in cursor.fetchall():
                doc_id, text, metadata, embedding_blob, doc_norm = row
                doc_vec = np.frombuffer(embedding_blob, dtype=np.float32)

                # Use pre-computed norm if available, otherwise compute
                if doc_norm is None:
                    doc_norm = np.linalg.norm(doc_vec) + 1e-9
                else:
                    doc_norm = float(doc_norm) + 1e-9

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
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT d.id, d.text, d.metadata, rank
                FROM documents d
                JOIN docs_fts fts ON d.rowid = fts.rowid
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
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            vec_count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

            return {
                'documents': doc_count,
                'vectors': vec_count,
                'db_path': self.db_path
            }

    def vacuum(self) -> None:
        """Optimize database by removing unused space"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            print("✅ Database optimized")


def main():
    """CLI for database operations"""
    parser = argparse.ArgumentParser(description="Database management CLI")
    parser.add_argument("--init", action="store_true", help="Initialize database schema")
    parser.add_argument("--force", action="store_true", help="Force reinitialize (drop existing tables)")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--vacuum", action="store_true", help="Optimize database")
    parser.add_argument("--db-path", help="Database file path")

    args = parser.parse_args()

    if not any([args.init, args.stats, args.vacuum]):
        parser.print_help()
        sys.exit(1)

    # Initialize database manager
    db_manager = VectorStore(args.db_path)

    if args.init:
        if args.force:
            # Drop existing tables for clean reinitialize
            with db_manager.get_connection() as conn:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.execute("DROP TABLE IF EXISTS vectors")
                conn.execute("DROP TABLE IF EXISTS docs_fts")
                conn.execute("DROP TABLE IF EXISTS documents")
                conn.execute("DROP TRIGGER IF EXISTS docs_ai")
                conn.execute("DROP TRIGGER IF EXISTS docs_ad")
                conn.execute("DROP TRIGGER IF EXISTS docs_au")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.commit()
            print("🗑️  Dropped existing tables")

        db_manager.init_db()
        print(f"✅ Database schema initialized at: {db_manager.db_path}")

    if args.stats:
        stats = db_manager.get_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Documents: {stats['documents']}")
        print(f"   Vectors: {stats['vectors']}")
        print(f"   Database: {stats['db_path']}")

    if args.vacuum:
        db_manager.vacuum()


# Global instance - created after main() to avoid schema conflicts
def get_store():
    """Get the global store instance, initializing if needed"""
    if not hasattr(get_store, '_instance'):
        get_store._instance = VectorStore(auto_init=False)
        get_store._instance.init_db()
    return get_store._instance

store = get_store()


if __name__ == "__main__":
    main()
