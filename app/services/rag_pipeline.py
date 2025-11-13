from app.services.embedder import Embedder
from app.services.chroma_store import ChromaStore
from app.services.llm_client import LLMClient


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder("multi-qa-MiniLM-L6-cos-v1")
        self.store = ChromaStore("data/chroma")
        self.llm = LLMClient()

    def query(self, question: str, k: int = 5) -> str:
        # 1. Embed the question
        q_vec = self.embedder.embed([question])[0]

        # 2. Retrieve top-k relevant chunks
        results = self.store.query(embedding=q_vec, k=k)
        contexts = results["documents"][0]

        # 3. Build a prompt
        context_block = "\n\n".join(contexts)
        print("--> context_block", context_block)
        prompt = f"""You are an expert assistant.
				Use only the context below to answer the user's question.

				Context:
				{context_block}

				Question: {question}

				Answer clearly and factually:
				"""

        # 4. Generate answer
        answer = self.llm.generate(prompt)
        return answer
