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

**Note**: The requirements now include additional dependencies for Week 3 document processing:

- `beautifulsoup4`: HTML parsing and text extraction
- `markdown`: Markdown processing
- `PyPDF2`: PDF text extraction

### Setup Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text:latest
ollama pull qwen3:8b
```

**Note**: The requirements now include additional dependencies for Week 3 document processing:

- `beautifulsoup4`: HTML parsing and text extraction
- `markdown`: Markdown processing
- `PyPDF2`: PDF text extraction

### Run the app

Option A — Python directly:

```bash
python main.py
```

Option B — via npm scripts (added in this repo):

```bash
npm run dev
```

## Week 3 — Document Ingestion

The enhanced ingestion pipeline supports multiple document formats with structured processing and deterministic IDs.

### Supported Formats

- **Text files** (`.txt`): Basic text processing
- **Markdown files** (`.md`): Heading-based block extraction with title preservation
- **HTML files** (`.html`): Text extraction with title and paragraph separation
- **PDF files** (`.pdf`): Page-based processing with paragraph extraction

### Basic Usage

```bash
# Ingest a single file
npm run ingest -- --path sample/sample.md

# Ingest all supported files in a directory
npm run ingest -- --path sample/

# Custom chunking parameters
npm run ingest -- --path sample/ --chunk-size 300 --chunk-overlap 25
```

### Features

- **Block-based processing**: Documents are first split into logical blocks (sections, paragraphs, pages)
- **Deterministic IDs**: Content-based hashing ensures identical content gets the same ID
- **Duplicate detection**: Automatic skipping of existing documents and embeddings
- **Metadata preservation**: Source, title, page numbers, and section information
- **Idempotent operation**: Safe to run multiple times without creating duplicates

### Database Management

```bash
# Initialize database schema
npm run db:init

# Force reinitialize (drops existing data)
npm run db:reinit

# Check database statistics
npm run db:stats

# Optimize database
npm run db:vacuum

# Create WAL checkpoint
npm run db:checkpoint
```

### Environment Variables

- `CHUNK_SIZE`: Maximum characters per chunk (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K`: Number of results to retrieve (default: 5)
- `HYBRID`: Enable hybrid search (default: false)

### Troubleshooting

**Missing dependencies**: If you encounter `ModuleNotFoundError` for `bs4`, `markdown`, or `PyPDF2`, run:

```bash
pip install -r requirements.txt
```

**PDF processing**: PDF support requires `PyPDF2`. If you don't need PDF support, you can skip this dependency.

**Database schema changes**: If you encounter database errors after schema updates, run:

```bash
npm run db:reinit
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
