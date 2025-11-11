from typing import List
from dataclasses import dataclass
from app.services.pdf_reader import PageTextDC


@dataclass(slots=True)
class TextChunk:
    id: str
    page: int
    text: str


def chunk_text(
    pages: List[PageTextDC], max_chars: int = 1000, overlap: int = 100
) -> List[TextChunk]:
    """
    Split text from pages into overlapping chunks.

    Args:
        pages: List of PageTextDC objects from extract_pdf_text
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List[TextChunk]: List of text chunks with id, page, and text
    """
    chunks = []

    for page_data in pages:
        text = page_data.text
        page_num = page_data.page
        source = page_data.source.rsplit(".", 1)[0]  # Remove file extension

        # If text is shorter than max_chars, create single chunk
        if len(text) <= max_chars:
            chunk_id = f"{source}_{len(chunks) + 1:03d}"
            chunks.append(TextChunk(id=chunk_id, page=page_num, text=text))
        else:
            # Split text into overlapping chunks
            i = 0
            while i < len(text):
                # Find the best end position to avoid cutting words
                end = text.rfind(" ", i, i + max_chars)
                if end == -1:
                    end = i + max_chars

                # Extract chunk text
                chunk_text = text[i:end].strip()

                # Create chunk
                chunk_id = f"{source}_{len(chunks) + 1:03d}"
                chunks.append(TextChunk(id=chunk_id, page=page_num, text=chunk_text))

                # Move start position for next chunk (with overlap)
                if end >= len(text):
                    break

                # Calculate next start position with proper overlap
                next_start = end - overlap
                if next_start <= i:
                    # If overlap is too large, just move forward by 1
                    next_start = i + 1

                # Try to start at word boundary for overlap
                if next_start > 0 and next_start < len(text):
                    # Look for the start of a word from the calculated position
                    word_start = text.find(" ", next_start)
                    if word_start != -1 and word_start - next_start < overlap // 2:
                        next_start = word_start + 1  # Start after the space

                i = next_start

    return chunks
