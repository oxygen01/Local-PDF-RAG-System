import os
import PyPDF2
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass(slots=True)
class PageTextDC:
    page: int
    text: str
    source: str

def extract_pdf_text(pdf_path: str) -> List[PageTextDC]:
    """
    Extract text from a PDF file and return structured data.

    Args:
        pdf_path (str): Path to the local PDF file

    Returns:
        List[Dict]: List of dictionaries with page number, text, and source filename

    Raises:
        FileNotFoundError: If the PDF file is not found
        Exception: If there's an error reading the PDF
    """
    results = []
    filename = os.path.basename(pdf_path)

    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                # Clean line breaks and extra whitespace
                cleaned_text = text.replace("\n", " ").strip()

                if cleaned_text:  # Only add non-empty pages
                    results.append(
                        PageTextDC(page=page_num, text=cleaned_text, source=filename)
                    )

    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

    return results
