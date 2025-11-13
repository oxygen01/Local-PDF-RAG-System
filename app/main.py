from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Local PDF RAG API",
    description="A local Retrieval-Augmented Generation pipeline with Chroma and LLM",
    version="0.1.0"
)

rag_pipeline = RAGPipeline()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def root():
    return {"message": "🚀 Local PDF RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        answer = rag_pipeline.query(request.question)
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

