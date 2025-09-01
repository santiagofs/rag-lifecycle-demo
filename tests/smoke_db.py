#!/usr/bin/env python3
"""
Offline smoke test for the vector store and retrieval logic.
It avoids network calls by supplying synthetic embeddings.
"""
import os
import sys
import json
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import VectorStore


def main():
    tmp_db = ROOT / "tests" / "tmp_vectors.db"
    # Clean up any previous run
    try:
        if tmp_db.exists():
            tmp_db.unlink()
    except Exception:
        pass

    store = VectorStore(db_path=str(tmp_db))

    # Create simple 3D embeddings for deterministic cosine behavior
    # Doc A: apple banana
    a_id = store.add_document("apple banana", [1.0, 1.0, 0.0])
    # Doc B: car truck
    b_id = store.add_document("car truck", [0.0, 0.0, 1.0])
    # Doc C: apple pie
    c_id = store.add_document("apple pie", [1.0, 0.5, 0.0])

    # Sanity: count
    cnt = store.get_document_count()
    assert cnt == 3, f"expected 3 docs, got {cnt}"

    # Cosine search: query close to A then C
    q_vec = [1.0, 0.8, 0.0]
    cos = store.search_cosine(q_vec, k=3)
    cos_ids = [r["id"] for r in cos]
    assert cos_ids[0] == a_id, f"cosine top-1 should be A ({a_id}), got {cos_ids[0]}"
    assert c_id in cos_ids[:2], f"C ({c_id}) should be in top-2, got {cos_ids}"

    # FTS search: query for 'car' should return B
    fts = store.search_fts("car", k=2)
    fts_ids = [r["id"] for r in fts]
    assert b_id in fts_ids, f"FTS should return B ({b_id}) for 'car', got {fts_ids}"

    # Hybrid: mix cosine (towards apple) and FTS (query 'apple')
    hyb = store.search_hybrid(q_vec, "apple", k=3, cosine_weight=0.7, fts_weight=0.3)
    hyb_ids = [r["id"] for r in hyb]
    assert a_id in hyb_ids[:2], f"Hybrid should rank A in top-2, got {hyb_ids}"

    print(json.dumps(
        {
            "status": "ok",
            "counts": {"documents": cnt},
            "cosine_ids": cos_ids,
            "fts_ids": fts_ids,
            "hybrid_ids": hyb_ids,
            "db_path": str(tmp_db),
        },
        indent=2,
    ))

    # Cleanup DB to avoid leaving artifacts
    try:
        if tmp_db.exists():
            tmp_db.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()

