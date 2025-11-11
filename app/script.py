from app.services.pdf_reader import extract_pdf_text
from app.services.chunker import chunk_text
from app.services.embedder import Embedder
from app.services.chroma_store import ChromaStore
from rich import print  # optional, for colored output
from rich.console import Console


def process_pdf(pdf_path: str = "app/files/attention.pdf"):
    # Extract text from PDF
    pages = extract_pdf_text(pdf_path)

    # Chunk the extracted text
    chunks = chunk_text(pages)

    # Prepare texts for embedding
    texts = [chunk.text for chunk in chunks]
    ids = [chunk.id for chunk in chunks]
    metas = [{"page": chunk.page, "source": "nist_ai_framework"} for chunk in chunks]

    embedder = Embedder()

    vectors = embedder.embed(texts)

    store = ChromaStore()
    store.add(ids=ids, texts=texts, metadatas=metas, embeddings=vectors)


def collection_count():
    store = ChromaStore()
    count = store.collection.count()
    print("Total documents in collection:", count)


def pretty_print_results(results, top_k: int = 5):
    """Nicely format and print Chroma retrieval results."""
    if not results["documents"]:
        print("[red]No results found.[/red]")
        return

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    # Sort by distance (lowest = best)
    ranked = sorted(zip(docs, dists, metas), key=lambda x: x[1])

    console = Console()
    console.print("[bold cyan]Top Retrieval Results[/bold cyan]")
    console.print("-" * 60)

    for i, (doc, dist, meta) in enumerate(ranked[:top_k], 1):
        page = meta.get("page", "?")
        source = meta.get("source", "?")
        console.print(f"[bold]#{i}[/bold] (page {page}, source: {source})")
        console.print(f"[green]Distance:[/green] {dist:.3f}")
        console.print(f"{doc[:300]}{'...' if len(doc) > 300 else ''}")
        console.print("-" * 60)


def query_pdf(question: str = "What is attention mechanism?", top_k: int = 5):
    from app.services.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    answer = rag.query(question=question, k=top_k)
    print("\n💬 RAG Answer:\n", answer)
 
