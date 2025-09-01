# rag-lifecycle-demo

## Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline, enabling a Large Language Model (LLM) to answer user questions using relevant context from a custom document set.

### High-level flow

1. **Document Embedding & Storage**

   - Source documents are split into chunks.
   - Each chunk is converted into a **semantic vector embedding** using an embedding model.
   - Embeddings are stored in a **vector database** for similarity search.

2. **Query Processing**

   - The user query is embedded using the _same_ embedding model, ensuring both documents and queries live in the same semantic space.
   - The vector database retrieves the top-_k_ most relevant document chunks based on vector similarity.

3. **Prompt Assembly**

   - Retrieved context is combined with the original query to form the **Final Prompt**.
   - The final prompt is sent to the LLM.

4. **Response Generation**
   - The LLM generates a grounded response using both the query and the retrieved context.

---

### Architecture Diagram

![RAG pipeline diagram](docs/diagram.png)

## Quick Start

### Requirements

- Python 3.10+
- Ollama running locally (`ollama serve`)
- Models:
  - `nomic-embed-text:latest` (embeddings)
  - `qwen3:8b` (LLM) — or any pulled chat model

### Setup MacOS/Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text:latest
ollama pull qwen3:8b
```

### Setup Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text:latest
ollama pull qwen3:8b
```

### Run the app

Option A — Python directly:

```bash
python main.py
```

Option B — via npm scripts (added in this repo):

```bash
npm run dev
```

### Run the tests (eval harness)

The evaluation harness calls Ollama's generate endpoint using unified env vars.

```bash
# optional, defaults shown
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b

npm run eval
# or
node eval/run.mjs eval/tests/golden.json
```

### Compare two specific archived test runs

node eval/compare.mjs eval/runs/<old>.json eval/runs/<new>.json --golden eval/tests/golden.json

### Compare newest vs previous (no args!)

node eval/compare.mjs

### Compare newest vs 3rd previous

node eval/compare.mjs --prev 3

### Golden auto-used if present, or pass explicitly

node eval/compare.mjs --prev 2 --golden eval/tests/golden.json

### Environment variables

- `OLLAMA_BASE_URL`: Base URL for Ollama (used by both Python and Node evals). Defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: Model name/tag to use for evals (default `llama3.1:8b`).
- `LLM_MODEL`: Model used by the Python demo app (read by `config.py`).
