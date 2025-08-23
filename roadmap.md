# Microplan — 12 Weeks (RAG Repo, Python core, Node eval)

- Week 1 – Ingestion Basics

  - Goal: Establish the simplest ingestion path to see the text→retrieval→answer loop
  - Steps:
    - Start from inline text phrases, hardcoded in Python lists
    - Keep everything in-memory, no database yet
    - Initial experiments with splitting into chunks by line
    - Basic manual retrieval of lines (simulating similarity search)
    - Set foundation for later structured ingestion

- Week 2 – Harness & Golden Dataset

  - Goal: Create a reproducible way to evaluate answers against expectations
  - Steps:
    - Build a Node/TypeScript evaluation harness
    - Define golden.csv with representative Q&A pairs
    - Run first manual checks comparing model outputs against expected answers
    - Standardize result format (pass/fail, latency, error)
    - Explore integration path between Node harness and Python pipeline
    - Begin structuring regression testing process

- Week 3 – Document Ingestion

  - Goal: Move beyond inline text to ingest structured documents
  - Steps:
    - Add ingestion for PDFs, HTML, Markdown
    - Normalize docs into a common block schema
    - Introduce metadata fields (title, source, page)
    - Implement chunking strategy (recursive split by headings, paragraphs)
    - Prepare sample corpora for later retrieval tests

- Week 4 – Vector Storage

  - Goal: Enable semantic similarity search with embeddings
  - Steps:
    - Set up FAISS index in Python
    - Embed document chunks using sentence-transformers or OpenAI embeddings
    - Store embeddings + metadata in FAISS, text in SQLite
    - Provide retrieval by top-k similarity
    - Validate by retrieving passages for golden Q&A pairs

- Week 5 – Retrieval Layer

  - Goal: Build a robust retrieval API on top of the vector store
  - Steps:
    - Implement FastAPI endpoints for retrieval
    - Add metadata filtering (by doc type, source, etc.)
    - Handle context assembly (multiple chunks per query)
    - Log queries + results for later analysis
    - First tests of Python retrieval against harness questions

- Week 6 – Generation Layer

  - Goal: Produce LLM answers using retrieved context
  - Steps:
    - Connect OpenAI or Ollama model to retrieved text
    - Define prompt template with placeholders for context + question
    - Stream responses back via FastAPI
    - Ensure outputs are captured for harness evaluation
    - First full pipeline: question → retrieve → generate → eval

- Week 7 – Evaluation Framework

  - Goal: Automate measurement of system quality
  - Steps:
    - Expand harness to run end-to-end pipeline automatically
    - Score outputs (exact match + semantic similarity)
    - Store metrics in SQLite for history
    - Generate accuracy/latency reports
    - Establish baseline for comparison over future weeks

- Week 8 – Augmentation

  - Goal: Extend RAG with tool use for harder questions
  - Steps:
    - Add optional tool layer (web search, calculator)
    - Implement tool interface classes in Python
    - Chain tool outputs into prompt context
    - Run harness against tool-enabled pipeline
    - Measure benefits vs overhead in evaluation

- Week 9 – Harness Expansion

  - Goal: Support multiple datasets and regression across versions
  - Steps:
    - Organize golden sets by domain/topic
    - Allow harness to run across multiple configs/models
    - Persist results per run with timestamp + version tag
    - Enable comparison of current vs past performance
    - Generate markdown or CSV regression reports

- Week 10 – Optimization

  - Goal: Improve retrieval and answer quality while lowering cost
  - Steps:
    - Experiment with different chunking sizes + overlaps
    - Test reranking models (BM25, cross-encoder)
    - Introduce caching layer for frequent queries
    - Compare latency and accuracy before/after optimizations
    - Document best practices from experiments

- Week 11 – Scaling

  - Goal: Handle larger corpora and higher query loads
  - Steps:
    - Build batch ingestion pipeline for thousands of docs
    - Add concurrency (async tasks, job queues)
    - Implement streaming retrieval and generation
    - Perform load testing with synthetic traffic
    - Identify bottlenecks and apply fixes

- Week 12 – Productization
  - Goal: Package the prototype into a usable app
  - Steps:
    - Wrap pipeline in stable FastAPI endpoints
    - Build minimal UI (Vue or simple HTML) for querying
    - Containerize with Docker for reproducible deploy
    - Define environment configuration (dotenv)
    - Deploy to a cloud instance for live demo
