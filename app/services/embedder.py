from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors."""
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)