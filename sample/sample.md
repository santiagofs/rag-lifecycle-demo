# Sample Markdown Document

## Introduction

This is a sample markdown document for testing the enhanced ingestion pipeline. It contains multiple sections with different types of content.

## Features

The enhanced ingestion pipeline supports:

- **Markdown files** (.md) with heading-based chunking
- **HTML files** (.html) with text extraction
- **PDF files** (.pdf) with page-based processing
- **Text files** (.txt) with basic chunking

## Technical Details

### Block Processing

Each document is processed into blocks:

- Markdown: Sections based on headings
- HTML: Paragraphs with title extraction
- PDF: Pages with paragraph separation
- Text: Single block with full content

### Chunking Strategy

Blocks are further chunked using:

- Configurable chunk size (default: 500 characters)
- Configurable overlap (default: 50 characters)
- Word boundary preservation
- Deterministic ID generation

### Metadata Preservation

Each chunk preserves metadata:

- Source file path
- Block index and chunk index
- Title (for HTML/Markdown)
- Page number (for PDF)
- Section information
