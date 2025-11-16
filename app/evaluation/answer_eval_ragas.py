"""
RAGAS-based Answer Quality Evaluation Module

This module evaluates the end-to-end RAG pipeline using RAGAS (Retrieval-Augmented
Generation Assessment) framework, which provides comprehensive metrics for both
retrieval and generation quality.

RAGAS evaluates four key aspects:
1. Context Precision: How relevant are the retrieved contexts
2. Context Recall: How well retrieved contexts cover the ground truth
3. Faithfulness: How grounded the generated answer is in the retrieved contexts
4. Answer Relevancy: How relevant the generated answer is to the question

Note: There's a typo in line 10 - should be "eval_questions.json" not "questinos.json"
"""

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

from app.services.rag_pipeline import RAGPipeline
import json



def run_ragas():
    """
    Run comprehensive RAGAS evaluation on the RAG pipeline.

    This function performs end-to-end evaluation by:
    1. Loading evaluation questions from JSON file
    2. Running each question through the complete RAG pipeline
    3. Collecting questions, generated answers, contexts, and ground truths
    4. Computing multiple RAGAS metrics to assess quality

    Returns:
        ragas.EvaluationResult: RAGAS evaluation results containing:
            - context_precision: Precision of retrieved contexts (0-1)
            - context_recall: Recall of retrieved contexts (0-1)
            - faithfulness: How grounded the answer is in context (0-1)
            - answer_relevancy: Relevance of answer to question (0-1)

    Raises:
        RuntimeError: If RAGAS dependencies are not available

    Example Usage:
        results = run_ragas()
        print(f"Context Precision: {results['context_precision']}")
        print(f"Answer Relevancy: {results['answer_relevancy']}")
    """
 
    # Initialize the complete RAG pipeline
    rag = RAGPipeline()

    # Load evaluation questions (Note: filename has typo - should be eval_questions.json)
    with open("app/evaluation/eval_questions.json") as f:
        data = json.load(f)

    # Initialize lists to collect evaluation data
    questions, answers, ground_truths, contexts = [], [], [], []

    # Process each evaluation question through the RAG pipeline
    for item in data:
        q = item["question"]  # The question to ask
        ground = item["expected"]  # Expected/ground truth answer

        # Generate answer using the complete RAG pipeline
        ans = rag.query(q)

        # Get the contexts that were retrieved and used for generation
        ctx = rag.last_contexts

        # Collect all components needed for RAGAS evaluation
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(ground)

    # Create HuggingFace dataset format required by RAGAS
    dataset = Dataset.from_dict({
        "question": questions,        # Input questions
        "answer": answers,           # Generated answers by RAG
        "contexts": contexts,        # Retrieved contexts used
        "ground_truth": ground_truths  # Expected correct answers
    })

    # Run RAGAS evaluation with comprehensive metrics
    results = evaluate(
        dataset,
        metrics=[
            context_precision,   # How precise are retrieved contexts
            context_recall,      # How complete are retrieved contexts
            faithfulness,        # How faithful is answer to contexts
            answer_relevancy,    # How relevant is answer to question
        ],
    )

    return results