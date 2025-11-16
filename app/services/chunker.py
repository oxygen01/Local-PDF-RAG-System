import nltk
from typing import List
from dataclasses import dataclass
from app.services.pdf_reader import PageTextDC

@dataclass(slots=True)
class TextChunk:
    id: str
    page: int
    text: str

def chunk_text(pages: List[PageTextDC], max_tokens: int = 300, overlap: int = 50) -> List[TextChunk]:
    """
    Split text into semantically meaningful chunks based on sentences.
    Ensures chunks do not break semantic boundaries and keeps fixed token limits.
    """
    chunks = []
    chunk_id_counter = 1

    for page_data in pages:
        sentences = nltk.sent_tokenize(page_data.text)
        current_chunk = []
        token_count = 0

        for sentence in sentences:
            sentence_tokens = sentence.split()
            if token_count + len(sentence_tokens) > max_tokens:
                # Save current chunk
                text = " ".join(current_chunk)
                chunks.append(
                    TextChunk(
                        id=f"{page_data.source}_{chunk_id_counter:04d}",
                        page=page_data.page,
                        text=text,
                    )
                )
                chunk_id_counter += 1

                # Overlap
                overlap_tokens = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_tokens.copy()
                token_count = len(current_chunk)

            # Add sentence
            current_chunk.extend(sentence_tokens)
            token_count += len(sentence_tokens)

        # Add the last chunk of the page
        if current_chunk:
            text = " ".join(current_chunk)
            chunks.append(
                TextChunk(
                    id=f"{page_data.source}_{chunk_id_counter:04d}",
                    page=page_data.page,
                    text=text,
                )
            )
            chunk_id_counter += 1

    return chunks