"""
Document loaders for various file formats.
Each loader returns a list of blocks with text and metadata.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup


class DocumentBlock:
    """Represents a block of text with metadata"""
    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata


def load_markdown(file_path: str) -> List[DocumentBlock]:
    """Load Markdown file and extract blocks with metadata"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    blocks = []
    lines = content.split('\n')
    current_block = []
    current_title = None

    for line in lines:
        # Check for headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            # Save previous block if it exists
            if current_block:
                block_text = '\n'.join(current_block).strip()
                if block_text:
                    blocks.append(DocumentBlock(
                        text=block_text,
                        metadata={
                            'source': file_path,
                            'title': current_title,
                            'section': current_title
                        }
                    ))

            # Start new block
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_title = title if level == 1 else current_title
            current_block = [line]
        else:
            current_block.append(line)

    # Add final block
    if current_block:
        block_text = '\n'.join(current_block).strip()
        if block_text:
            blocks.append(DocumentBlock(
                text=block_text,
                metadata={
                    'source': file_path,
                    'title': current_title,
                    'section': current_title
                }
            ))

    return blocks


def load_html(file_path: str) -> List[DocumentBlock]:
    """Load HTML file and extract text blocks with metadata"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Extract title
    title_tag = soup.find('title')
    title = title_tag.get_text().strip() if title_tag else None

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Extract text from body
    body = soup.find('body')
    if not body:
        body = soup

    # Get text and split into paragraphs
    text = body.get_text()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    blocks = []
    for i, para in enumerate(paragraphs, 1):
        if para:
            blocks.append(DocumentBlock(
                text=para,
                metadata={
                    'source': file_path,
                    'title': title,
                    'section': f'paragraph_{i}'
                }
            ))

    return blocks


def load_pdf(file_path: str) -> List[DocumentBlock]:
    """Load PDF file and extract text blocks with page metadata"""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

    blocks = []

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

                    for para_num, para in enumerate(paragraphs, 1):
                        if para:
                            blocks.append(DocumentBlock(
                                text=para,
                                metadata={
                                    'source': file_path,
                                    'page': page_num,
                                    'section': f'page_{page_num}_para_{para_num}'
                                }
                            ))
    except Exception as e:
        # Fallback: treat as text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        blocks.append(DocumentBlock(
            text=content,
            metadata={
                'source': file_path,
                'section': 'fallback_text'
            }
        ))

    return blocks


def load_text(file_path: str) -> List[DocumentBlock]:
    """Load plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    return [DocumentBlock(
        text=content,
        metadata={
            'source': file_path,
            'section': 'text_content'
        }
    )]


def get_loader_for_file(file_path: str):
    """Get appropriate loader based on file extension"""
    ext = Path(file_path).suffix.lower()

    if ext == '.md':
        return load_markdown
    elif ext == '.html':
        return load_html
    elif ext == '.pdf':
        return load_pdf
    elif ext == '.txt':
        return load_text
    else:
        raise ValueError(f"Unsupported file type: {ext}")
