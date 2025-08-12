import numpy as np

_docs: list[dict] = []

def add_doc(text: str, vector: list[float]) -> None:
    _docs.append({"text": text, "vec": np.array(vector, dtype=np.float32)})

def search(vector: list[float], k: int = 3) -> list[str]:
    q = np.array(vector, dtype=np.float32)
    sims = []
    qn = np.linalg.norm(q) + 1e-9
    for d in _docs:
        dn = np.linalg.norm(d["vec"]) + 1e-9
        sims.append(((q @ d["vec"]) / (qn * dn), d["text"]))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in sims[:k]]