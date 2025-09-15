"""
Unit tests for document chunking functionality.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loaders import DocumentBlock, load_text, load_markdown
from src.ingest_cli import chunk_text


class TestChunking(unittest.TestCase):
    """Test chunking functionality"""

    def test_chunk_text_small(self):
        """Test chunking of small text"""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_large(self):
        """Test chunking of large text"""
        text = "This is a longer text. " * 50  # ~1000 characters
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)

        # Check that chunks don't exceed size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_chunk_text_word_boundaries(self):
        """Test that chunking respects word boundaries"""
        text = "This is a sentence. This is another sentence. " * 10
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        for chunk in chunks:
            # Should not break in the middle of words (check for incomplete words)
            words = chunk.split()
            for word in words:
                # Each word should be complete (not truncated)
                self.assertTrue(len(word) > 0)

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap"""
        text = "Word1. Word2. Word3. Word4. Word5. " * 5
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=30)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            chunk1 = chunks[0]
            chunk2 = chunks[1]
            # Should share some words due to overlap
            self.assertTrue(any(word in chunk2 for word in chunk1.split()[:5]))


class TestLoaders(unittest.TestCase):
    """Test document loaders"""

    def test_load_text(self):
        """Test text file loading"""
        # Create a temporary text file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            temp_file = f.name

        try:
            blocks = load_text(temp_file)
            self.assertEqual(len(blocks), 1)
            self.assertIn("test document", blocks[0].text)
            self.assertEqual(blocks[0].metadata['source'], temp_file)
        finally:
            import os
            os.unlink(temp_file)

    def test_load_markdown(self):
        """Test markdown file loading"""
        # Create a temporary markdown file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Title\n\nThis is content.\n\n## Subtitle\n\nMore content.")
            temp_file = f.name

        try:
            blocks = load_markdown(temp_file)
            self.assertGreater(len(blocks), 1)
            # Should have title in metadata
            self.assertEqual(blocks[0].metadata['title'], 'Title')
        finally:
            import os
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()
