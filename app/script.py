from app.services.pdf_reader import extract_pdf_text
from app.services.chunker import chunk_text
from app.services.embedder import Embedder
from app.services.chroma_store import ChromaStore
from rich import print  # optional, for colored output
from rich.console import Console
import numpy as np

def process_pdf(pdf_path: str = "app/files/attention.pdf"):
    # Extract text from PDF
    pages = extract_pdf_text(pdf_path)

    # Chunk the extracted text
    chunks = chunk_text(pages)

    # Prepare texts for embedding
    texts = [chunk.text for chunk in chunks]
    ids = [chunk.id for chunk in chunks]
    metas = [{"page": chunk.page, "source": "transformer_paper"} for chunk in chunks]

    embedder = Embedder()
    # 1. embed texts
    vectors = embedder.embed(texts)
    # 2. normalize vectors (critical for cosine search!)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    store = ChromaStore()
    store.add(ids=ids, texts=texts, metadatas=metas, embeddings=vectors)
    print("✅ PDF processed and stored successfully.")


def collection_count():
    store = ChromaStore()
    count = store.collection.count()
    print("Total documents in collection:", count)


def query_pdf(question: str = "What is attention mechanism?", top_k: int = 5):
    from app.services.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    answer = rag.query(question=question, k=top_k)
    print("\n💬 RAG Answer:\n", answer)
 
