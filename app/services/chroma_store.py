import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

class ChromaStore:
    def __init__(self, persist_directory: str = "data/chroma"):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.persist_directory = persist_directory
        self.collection = self.client.get_or_create_collection(name="pdf_chunks", 
                                                               # ensures cosine distance metric
                                                               metadata={"hnsw:space": "cosine"})

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ):
        """Add embeddings and metadata to Chroma collection."""
        self.collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def query(self, embedding: list[float], k: int = 5):
        """Retrieve top-k similar chunks."""
        results = self.collection.query(query_embeddings=[embedding], n_results=k)
        return results
