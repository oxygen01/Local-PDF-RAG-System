from app.services.embedder import Embedder
from app.services.chroma_store import ChromaStore
from sentence_transformers import util
import numpy as np
import json


def semantic_recall(chunks, expected_text, embedder, threshold=0.45):
    """
    Semantic Recall@k:
    - chunks: List[str] retrieved chunk texts
    - expected_text: str ground-truth short answer
    - embedder: Embedder instance
    - threshold: cosine similarity threshold

    Returns:
        1 if any chunk semantically matches expected_text, otherwise 0.
    """

    if not chunks:
        return 0

    chunk_embs = embedder.embed(chunks)
    chunk_embs = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)

    expected_emb = embedder.embed([expected_text])[0]
    expected_emb = expected_emb / np.linalg.norm(expected_emb)

    sims = util.cos_sim(expected_emb, chunk_embs)[0]

    return int(float(max(sims)) >= threshold)


def run_retrieval_eval():
    embedder = Embedder("multi-qa-MiniLM-L6-cos-v1")
    store = ChromaStore()

    with open("app/evaluation/eval_questions.json") as f:
        eval_data = json.load(f)

    results = []

    for item in eval_data:
        q = item["question"]
        expected = item["expected"]

        # Embed & normalize query
        q_vec = embedder.embed([q])[0]
        q_vec = q_vec / np.linalg.norm(q_vec)

        # Retrieve
        retrieved = store.query(embedding=q_vec, k=5)
        chunks = retrieved["documents"][0]

        score = semantic_recall(chunks, expected, embedder)

        results.append((q, score))

    return results
