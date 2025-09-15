import os
import json
import sqlite3
from typing import List, Dict, Optional, Tuple

import numpy as np

# FAISS import (CPU)
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required. Install via `pip install faiss-cpu`." ) from e

# Project imports
from config import DB_PATH, EMBED_MODEL_NAME, INDEX_DIR
from src.embeddings import get_embedding


DEFAULT_INDEX_DIR = INDEX_DIR


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row‑wise L2 normalize a float32 numpy array."""
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _index_files(out_dir: str) -> Tuple[str, str, str]:
    idx_path = os.path.join(out_dir, "index.faiss")
    ids_path = os.path.join(out_dir, "ids.json")
    meta_path = os.path.join(out_dir, "meta.json")
    return idx_path, ids_path, meta_path


def build_from_db(
    db_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict:
    """Build a FAISS IP index from SQLite vectors and persist artifacts.

    - Uses cosine similarity by indexing L2-normalized vectors in IndexFlatIP.
    - Persists: index.faiss, ids.json (doc_id order), meta.json.
    """

    db_path = db_path or DB_PATH
    out_dir = out_dir or DEFAULT_INDEX_DIR
    _ensure_dir(out_dir)

    with sqlite3.connect(db_path) as conn:
        if model:
            cur = conn.execute(
                "SELECT doc_id, vec, dim, model FROM vectors WHERE model = ? ORDER BY rowid",
                (model,),
            )
        else:
            cur = conn.execute(
                "SELECT doc_id, vec, dim, model FROM vectors ORDER BY rowid"
            )

        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No vectors found in database for index build.")

    # Determine dimension from first row
    first_dim = int(rows[0][2])

    ids: List[str] = []
    vecs: List[np.ndarray] = []
    models: List[str] = []

    for doc_id, vec_blob, dim, vec_model in rows:
        dim = int(dim)
        if dim != first_dim:
            # Skip inconsistent dims to keep index coherent
            continue
        vec = np.frombuffer(vec_blob, dtype=np.float32)
        if vec.size != dim:
            continue
        ids.append(str(doc_id))
        vecs.append(vec)
        models.append(vec_model)

    if not vecs:
        raise RuntimeError("No valid vectors with consistent dimensions found.")

    mat = np.vstack(vecs).astype(np.float32)
    mat = _l2_normalize(mat)

    index = faiss.IndexFlatIP(first_dim)
    index.add(mat)

    idx_path, ids_path, meta_path = _index_files(out_dir)
    faiss.write_index(index, idx_path)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    # Choose a single model label for metadata
    model_label = model or (models[0] if len(set(models)) == 1 else "mixed")
    meta = {
        "db_path": db_path,
        "dim": first_dim,
        "count": len(ids),
        "model": model_label,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return meta


class _IndexHandle:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        idx_path, ids_path, meta_path = _index_files(out_dir)
        if not (os.path.exists(idx_path) and os.path.exists(ids_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"Index artifacts not found in {out_dir}. Build with build_from_db() first."
            )
        self.index = faiss.read_index(idx_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.ids: List[str] = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict = json.load(f)

        # Guardrails: basic consistency checks and helpful warnings
        try:
            # Check index size vs meta
            if hasattr(self.index, 'ntotal') and isinstance(self.meta.get("count"), int):
                if int(self.index.ntotal) != int(self.meta["count"]):
                    print(
                        f"⚠️  FAISS index size ({self.index.ntotal}) differs from meta.count ({self.meta['count']}). Consider rebuilding."
                    )

            # Check DB vector count vs meta.count
            db_path = self.meta.get("db_path") or DB_PATH
            if db_path and os.path.exists(db_path):
                import sqlite3 as _sqlite3
                with _sqlite3.connect(db_path) as _conn:
                    row = _conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
                    vec_count = int(row[0]) if row else 0
                    meta_count = int(self.meta.get("count", -1))
                    if meta_count >= 0 and vec_count != meta_count:
                        print(
                            f"⚠️  DB vectors ({vec_count}) differ from index meta.count ({meta_count}). Rebuild index for fresh results."
                        )
        except Exception:
            # Non-fatal; do not block loading if checks fail
            pass


_CACHED: Optional[_IndexHandle] = None


def load(out_dir: Optional[str] = None) -> _IndexHandle:
    """Load and cache FAISS index + mapping files."""
    global _CACHED
    out_dir = out_dir or DEFAULT_INDEX_DIR
    if _CACHED and _CACHED.out_dir == out_dir:
        return _CACHED
    _CACHED = _IndexHandle(out_dir)
    return _CACHED


def _fetch_docs(db_path: str, doc_ids: List[str]) -> Dict[str, Dict]:
    """Fetch texts and metadata for a list of doc_ids from SQLite."""
    if not doc_ids:
        return {}
    placeholders = ",".join(["?"] * len(doc_ids))
    sql = f"SELECT id, text, metadata FROM documents WHERE id IN ({placeholders})"
    out: Dict[str, Dict] = {}
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(sql, tuple(doc_ids)):
            did, text, metadata = row
            out[str(did)] = {
                "id": str(did),
                "text": text,
                "metadata": json.loads(metadata) if metadata else None,
            }
    return out


def search(
    query_text: str,
    k: int = 5,
    out_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict]:
    """Search the FAISS index using cosine similarity over normalized vectors.

    - Embeds `query_text` with the existing embedding pipeline.
    - Returns top-k with id, text, metadata, and score (cosine).
    """
    handle = load(out_dir)

    # Optional model sanity check: if requested, ensure index model matches
    if model and handle.meta.get("model") not in (model, "mixed"):
        raise ValueError(
            f"Requested model={model} but index built for model={handle.meta.get('model')}"
        )

    # Embed and normalize
    vec = np.array(get_embedding(query_text), dtype=np.float32)
    vec = _l2_normalize(vec)

    # FAISS expects shape (n, d)
    D, I = handle.index.search(vec.reshape(1, -1), k)
    scores = D[0]
    idxs = I[0]

    # Map indices to doc IDs
    doc_ids: List[str] = []
    for ix in idxs:
        if ix < 0 or ix >= len(handle.ids):
            continue
        doc_ids.append(handle.ids[ix])

    # Fetch documents
    db_path = db_path or DB_PATH
    docs = _fetch_docs(db_path, doc_ids)

    # Compose results maintaining order
    results: List[Dict] = []
    for rank, (ix, score) in enumerate(zip(idxs, scores)):
        if ix < 0 or ix >= len(handle.ids):
            continue
        did = handle.ids[ix]
        d = docs.get(did, {"id": did, "text": None, "metadata": None})
        results.append(
            {"id": did, "text": d.get("text"), "metadata": d.get("metadata"), "score": float(score)}
        )

    return results


if __name__ == "__main__":  # Simple CLI for convenience
    import argparse

    parser = argparse.ArgumentParser(description="FAISS index utilities")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="Build FAISS index from SQLite vectors")
    b.add_argument("--db", dest="db", default=DB_PATH)
    b.add_argument("--out", dest="out", default=DEFAULT_INDEX_DIR)
    b.add_argument("--model", dest="model", default=None)

    s = sub.add_parser("search", help="Query the FAISS index")
    s.add_argument("query")
    s.add_argument("--k", dest="k", type=int, default=5)
    s.add_argument("--out", dest="out", default=DEFAULT_INDEX_DIR)
    s.add_argument("--db", dest="db", default=DB_PATH)

    args = parser.parse_args()

    if args.cmd == "build":
        meta = build_from_db(db_path=args.db, out_dir=args.out, model=args.model)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "search":
        hits = search(args.query, k=args.k, out_dir=args.out, db_path=args.db)
        for h in hits:
            preview = (h["text"] or "")[:100].replace("\n", " ")
            print(f"{h['score']:.4f}\t{h['id']}\t{preview}")
    else:
        parser.print_help()
