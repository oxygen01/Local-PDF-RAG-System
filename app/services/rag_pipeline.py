from app.services.embedder import Embedder
from app.services.chroma_store import ChromaStore
from app.services.llm_client import LLMClient
import numpy as np


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder("multi-qa-MiniLM-L6-cos-v1")
        self.store = ChromaStore()
        self.llm = LLMClient()
        self.last_contexts: list[str] = []
        self.last_metadatas: list[dict] = []

    def query(self, question: str, k: int = 5) -> str:
        # 1. Embed the question
        q_vec = self.embedder.embed([question])[0]

        # 2. Retrieve top-k relevant chunks
        q_vec = q_vec / np.linalg.norm(q_vec)
        results = self.store.query(embedding=q_vec, k=k)
        contexts = results["documents"][0]
        metadatas = results["metadatas"][0]
        self.last_contexts = contexts
        self.last_metadatas = metadatas

        # 3. Build a prompt
        context_block = "\n\n".join(contexts)

        prompt = f"""You are an expert assistant.
				You MUST answer ONLY using the provided context.
        If the answer is NOT contained in the context, reply with exactly:
        "I don't know based on the provided context."
				Context:
				{context_block}

				Question: {question}

				Answer:
				"""

        # 4. Generate answer
        answer = self.llm.generate(prompt)
        return answer
