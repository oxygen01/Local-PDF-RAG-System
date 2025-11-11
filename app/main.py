from fastapi import FastAPI

app = FastAPI(
    title="Local PDF RAG API",
    description="A local Retrieval-Augmented Generation pipeline with Chroma and LLM",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "🚀 Local PDF RAG API is running!"}