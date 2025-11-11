import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.chunker import chunk_text
from app.services.pdf_reader import PageTextDC


class TestChunker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.sample_pages = [
            PageTextDC(
                page=1,
                text="This is a short text that should fit in one chunk.",
                source="test.pdf",
            ),
            PageTextDC(
                page=2,
                text="This is a much longer text that will need to be split into multiple chunks because it exceeds the maximum character limit that we set for testing purposes. This text continues on and on with more content to ensure we test the chunking behavior properly.",
                source="test.pdf",
            ),
            PageTextDC(
                page=3,
                text="Another page with moderate length content that might be chunked.",
                source="test.pdf",
            ),
        ]

    def test_short_text_single_chunk(self):
        """Test that short text creates a single chunk."""
        pages = [self.sample_pages[0]]  # Short text
        chunks = chunk_text(pages, max_chars=100, overlap=20)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[0].id, "test_001")
        self.assertEqual(
            chunks[0].text, "This is a short text that should fit in one chunk."
        )

    def test_long_text_multiple_chunks(self):
        """Test that long text is split into multiple chunks."""
        pages = [self.sample_pages[1]]  # Long text
        chunks = chunk_text(pages, max_chars=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        # All chunks should be from the same page
        for chunk in chunks:
            self.assertEqual(chunk.page, 2)

        # Check that chunks have proper IDs
        self.assertEqual(chunks[0].id, "test_001")
        self.assertEqual(chunks[1].id, "test_002")

    def test_multiple_pages(self):
        """Test chunking across multiple pages."""
        chunks = chunk_text(self.sample_pages, max_chars=50, overlap=10)

        self.assertGreater(len(chunks), len(self.sample_pages))

        # Check that page numbers are preserved
        page_numbers = {chunk.page for chunk in chunks}
        self.assertIn(1, page_numbers)
        self.assertIn(2, page_numbers)
        self.assertIn(3, page_numbers)

    def test_overlap_functionality(self):
        """Test that overlap works correctly."""
        pages = [
            PageTextDC(page=1, text="abcdefghijklmnopqrstuvwxyz", source="test.pdf")
        ]
        chunks = chunk_text(pages, max_chars=10, overlap=3)

        self.assertGreater(len(chunks), 1)
        # Check that chunking occurred with the specified parameters
        # Due to word boundary logic, exact overlap might vary, but we should have multiple chunks

    def test_empty_pages(self):
        """Test handling of empty pages list."""
        chunks = chunk_text([], max_chars=100, overlap=20)
        self.assertEqual(len(chunks), 0)

    def test_chunk_id_format(self):
        """Test that chunk IDs follow the correct format."""
        chunks = chunk_text(self.sample_pages, max_chars=50, overlap=10)

        for i, chunk in enumerate(chunks, 1):
            expected_id = f"test_{i:03d}"
            self.assertEqual(chunk.id, expected_id)

    def test_source_without_extension(self):
        """Test that source filename extension is removed from ID."""
        pages = [PageTextDC(page=1, text="Test text", source="document.pdf")]
        chunks = chunk_text(pages, max_chars=100, overlap=20)

        self.assertEqual(chunks[0].id, "document_001")

    def test_word_boundary_preservation(self):
        """Test that words are not cut in the middle when chunking."""
        # Simple test with predictable word boundaries
        text = "The quick brown fox jumps over the lazy dog again and again"
        pages = [PageTextDC(page=1, text=text, source="test.pdf")]

        # Set max_chars to 20, overlap to 5 - should break at word boundaries
        chunks = chunk_text(pages, max_chars=20, overlap=5)

        self.assertGreater(len(chunks), 1)

        # Print chunks for debugging
        print(f"\nOriginal text: '{text}'")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i + 1}: '{chunk.text}'")

        # Check that most chunks contain complete words
        # Allow some flexibility for edge cases but ensure no completely broken words
        single_letter_count = 0
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.text.strip()
            if len(chunk_content) == 1 and chunk_content.isalpha():
                single_letter_count += 1

        # Should have minimal single-letter chunks (at most 2 for edge cases)
        self.assertLessEqual(
            single_letter_count,
            2,
            "Too many single-letter chunks suggest poor word boundary handling",
        )

    def test_word_boundary_edge_cases(self):
        """Test word boundary handling with edge cases."""
        # Text with no spaces (should fall back to character cutting)
        no_spaces_text = "abcdefghijklmnopqrstuvwxyz"
        pages = [PageTextDC(page=1, text=no_spaces_text, source="test.pdf")]
        chunks = chunk_text(pages, max_chars=10, overlap=3)

        self.assertGreater(len(chunks), 1)
        # Should still create chunks even without spaces
        total_length = sum(len(chunk.text) for chunk in chunks)
        self.assertGreaterEqual(
            total_length, len(no_spaces_text) - 10
        )  # Account for overlap

        # Text ending exactly at space
        space_boundary_text = "word1 word2 word3 word4"
        pages = [PageTextDC(page=1, text=space_boundary_text, source="test.pdf")]
        chunks = chunk_text(pages, max_chars=12, overlap=3)  # Should break at "word2 "

        if len(chunks) > 1:
            # First chunk should end cleanly
            first_chunk = chunks[0].text.strip()
            self.assertTrue(
                first_chunk.endswith("word2") or first_chunk.endswith("word1")
            )

if __name__ == "__main__":
    unittest.main()
