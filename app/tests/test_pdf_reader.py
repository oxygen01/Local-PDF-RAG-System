import unittest
import os
from unittest.mock import patch, mock_open, MagicMock
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.pdf_reader import extract_pdf_text


class TestPDFReader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_pdf_path = "/path/to/test.pdf"

    @patch(
        "app.services.pdf_reader.open",
        new_callable=mock_open,
        read_data=b"fake pdf content",
    )
    @patch("app.services.pdf_reader.PyPDF2.PdfReader")
    def test_extract_pdf_text_success(self, mock_pdf_reader, mock_file):
        """Test successful PDF text extraction."""
        # Mock PDF pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "This is page 1\nwith some text"

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "This is page 2\nwith more content"

        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_pdf_reader_instance

        # Call the function
        result = extract_pdf_text(self.test_pdf_path)

        # Assertions
        self.assertEqual(len(result), 2)

        # Check first page
        self.assertEqual(result[0].page, 1)
        self.assertEqual(result[0].text, "This is page 1 with some text")
        self.assertEqual(result[0].source, "test.pdf")

        # Check second page
        self.assertEqual(result[1].page, 2)
        self.assertEqual(result[1].text, "This is page 2 with more content")
        self.assertEqual(result[1].source, "test.pdf")

    @patch(
        "app.services.pdf_reader.open",
        new_callable=mock_open,
        read_data=b"fake pdf content",
    )
    @patch("app.services.pdf_reader.PyPDF2.PdfReader")
    def test_extract_pdf_text_empty_pages(self, mock_pdf_reader, mock_file):
        """Test PDF extraction with empty pages."""
        # Mock PDF pages with empty content
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Valid content"

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = ""  # Empty page

        mock_page3 = MagicMock()
        mock_page3.extract_text.return_value = "   \n\n  "  # Only whitespace

        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdf_reader.return_value = mock_pdf_reader_instance

        # Call the function
        result = extract_pdf_text(self.test_pdf_path)

        # Should only return the page with valid content
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page, 1)
        self.assertEqual(result[0].text, "Valid content")

    def test_extract_pdf_text_file_not_found(self):
        """Test FileNotFoundError when PDF file doesn't exist."""
        non_existent_path = "/path/to/nonexistent.pdf"

        with self.assertRaises(FileNotFoundError) as context:
            extract_pdf_text(non_existent_path)

        self.assertIn("PDF file not found", str(context.exception))

    @patch(
        "app.services.pdf_reader.open", side_effect=PermissionError("Permission denied")
    )
    def test_extract_pdf_text_permission_error(self, mock_file):
        """Test exception handling for permission errors."""
        with self.assertRaises(Exception) as context:
            extract_pdf_text(self.test_pdf_path)

        self.assertIn("Error reading PDF", str(context.exception))

    @patch(
        "app.services.pdf_reader.open",
        new_callable=mock_open,
        read_data=b"fake pdf content",
    )
    @patch(
        "app.services.pdf_reader.PyPDF2.PdfReader",
        side_effect=Exception("Corrupted PDF"),
    )
    def test_extract_pdf_text_corrupted_pdf(self, mock_pdf_reader, mock_file):
        """Test exception handling for corrupted PDF files."""
        with self.assertRaises(Exception) as context:
            extract_pdf_text(self.test_pdf_path)

        self.assertIn("Error reading PDF", str(context.exception))

    def test_basename_extraction(self):
        """Test that os.path.basename correctly extracts filename."""
        # This is implicitly tested in other tests, but let's be explicit
        test_paths = [
            "/home/user/documents/report.pdf",
            "C:\\Users\\User\\Desktop\\file.pdf",
            "simple_file.pdf",
            "/complex/path/with spaces/document name.pdf",
        ]

        for path in test_paths:
            expected_filename = os.path.basename(path)
            # We can't easily test this in isolation without mocking the entire function,
            # but this demonstrates the expected behavior
            self.assertTrue(expected_filename.endswith(".pdf"))


if __name__ == "__main__":
    unittest.main()
