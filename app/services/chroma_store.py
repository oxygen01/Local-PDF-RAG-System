import chromadb

class ChromaStore:
    def __init__(self):
        self.client = chromadb.HttpClient(host="chroma", port=8000)
        self.collection = self.client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )

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
