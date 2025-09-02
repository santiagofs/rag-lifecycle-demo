# Project Goals (North Star)

- Build a **reliable RAG pipeline** that ingests sources → chunks → embeds → indexes → retrieves → answers with citations.
- Maintain **engineering hygiene** (tests, lint, typing, CI) so progress is measurable and reviews are objective.
- Produce **repeatable evaluations** (golden set + harness) to track quality over time.

# Success Metrics

- ≥90% pipeline tests passing; ≥80% code coverage on core modules.
- Retrieval quality: ≥X% answer correctness on golden set; stable latency budget (p95 ≤ Y ms).
- Reproducible runs: same commit + config ⇒ same results.

# Constraints & Standards

- Languages/stack: **Python** for pipeline; **Node** only for eval/harness if needed.
- Local models via **Ollama**; vectors via **SQLite/FTS5** (or drop-in vector DB when noted).
- Style: black/ruff (py), eslint/prettier (ts). Type hints mandatory. No `any`.
- Configuration via `.env` + typed config module. Deterministic seeds.

# Major Milestones (with reviewable artifacts)

## 1. Repo Foundation & Governance

- Repo structure, Makefile/justfile, pyproject, lint/test setup, pre-commit, CI
- Deliverables: CI badge, CONTRIBUTING.md, CODEOWNERS
- Review: deterministic installs, clear run/test targets

## 2. Ingestion Layer

- Loaders for plain text; source registry; content hashing & dedupe
- Deliverables: `ingest/` module + CLI; raw docs table
- Review: idempotency, metadata completeness, failure logs

## 3. Chunking & Normalization

- Rule-based chunker; metadata propagation; unit tests
- Deliverables: `chunk/` module; chunk schema; configs
- Review: stable chunk sizes, no metadata loss

## 4. Embeddings & Indexing

- Ollama embeddings wrapper; batcher; vector schema (SQLite/FTS5)
- Deliverables: `embed/` module; `index build` CLI; index stats
- Review: vector dim checks, insert throughput, integrity

## 5. Retrieval API

- KNN + keyword filter; hybrid scoring; re-rank hook
- Deliverables: `retrieve/` module; query CLI with JSON
- Review: scoring transparency, deterministic results

## 6. Answering (RAG Compose)

- Context window builder; prompt templates; citation stitching
- Deliverables: `rag/answer` CLI; streaming; YAML prompts
- Review: prompt versioning, token guards, citation alignment

## 7. Golden Set & Evaluation Harness

- `golden.csv` (Q, expected, refs); harness for scoring
- Deliverables: `/eval` with runners, metrics, trend report
- Review: stable metrics, easy comparisons

## 8. Observability & Logging

- Structured logs, timing metrics, counters, error taxonomy
- Deliverables: run summary, HTML/Markdown report
- Review: regression diagnosis possible from logs

## 9. Configuration & Profiles

- Profiles: local-cpu, quality, fast; env+YAML merging
- Deliverables: `/configs` with profiles; README matrix
- Review: minimal toggles, documented defaults

## 10. Packaging & Interfaces

- Python package; CLI; thin HTTP endpoint
- Deliverables: `pip install -e .`; `/api` minimal server
- Review: boundaries clear, concurrency basics

## 11. Docs & Examples

- Quickstart, architecture diagram, examples notebook
- Deliverables: `/docs` with 10-min run guide; ADRs
- Review: new dev can onboard easily

## 12. Quality Gates & Release Process

- CI gates (lint, type, test, eval threshold); versioning; changelog
- Deliverables: GitHub Actions with fail on regressions; RELEASE.md
- Review: prevents regressions; reproducible releases

# Peer-Review Rubric

- **Correctness**: Does it do what it claims? Tested?
- **Determinism**: Same inputs ⇒ same outputs?
- **Observability**: Logs/metrics/errors sufficient?
- **Interfaces**: Clear types, docs, boundaries?
- **Performance**: Obvious hot paths resolved?
- **Security/Safety**: Validations, prompt guards, secrets?
- **Regression Risk**: Eval metrics stable/improved?

# Progress Signals to Track

- CI status + coverage trend
- Eval dashboard deltas (accuracy, citation hit-rate, latency)
- Data volume (docs, chunks, index size)
- Release tags with passing gates; ADR count
