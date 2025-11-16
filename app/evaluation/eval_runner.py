"""
Evaluation Runner - Orchestrates All RAG System Evaluations

This module serves as the main entry point for running comprehensive evaluations
of the RAG pipeline. It coordinates both retrieval-specific and end-to-end
evaluations, then generates markdown reports for analysis.

The runner performs two types of evaluation:
1. Retrieval Evaluation: Measures how well the vector search finds relevant chunks
2. RAGAS Evaluation: Comprehensive assessment of the complete RAG pipeline

Reports are saved in markdown format for easy viewing and sharing.
"""

from app.evaluation.retrieval_eval import run_retrieval_eval
from pathlib import Path

from app.evaluation.answer_eval_ragas import run_ragas


def run_all_evaluations():
    """
    Execute all available evaluations and generate comprehensive reports.

    This function orchestrates the complete evaluation workflow:
    1. Runs retrieval evaluation to assess vector search quality
    2. Runs RAGAS evaluation to assess end-to-end pipeline quality
    3. Creates reports directory if it doesn't exist
    4. Generates markdown reports for both evaluations
    5. Saves reports to app/evaluation/reports/ directory

    The function creates two report files:
    - retrieval_report.md: Contains Recall@5 scores for each test question
    - ragas_report.md: Contains comprehensive RAGAS metrics in table format

    Example Usage:
        from app.evaluation.eval_runner import run_all_evaluations
        run_all_evaluations()
        # Check app/evaluation/reports/ for generated reports
    """
    # Phase 1: Evaluate retrieval quality
    print("Running Retrieval Evaluation...")
    retrieval_results = run_retrieval_eval()

    # Ensure reports directory exists
    Path("app/evaluation/reports").mkdir(exist_ok=True)

    # Generate retrieval evaluation report in markdown format
    with open("app/evaluation/reports/retrieval_report.md", "w") as f:
        f.write("# Retrieval Evaluation Report\n\n")
        f.write("This report shows Recall@5 scores for each evaluation question.\n")
        f.write(
            "Score of 1 means the expected answer was found in top 5 retrieved chunks.\n\n"
        )

        for q, score in retrieval_results:
            f.write(f"### Q: {q}\nRecall@5: {score}\n\n")

    # Phase 2: Evaluate end-to-end pipeline quality using RAGAS (if available)
    print("Running RAGAS Evaluation...")
    try:
        ragas_scores = run_ragas()

        # Generate RAGAS evaluation report in markdown table format
        with open("app/evaluation/reports/ragas_report.md", "w") as f:
            f.write("# RAGAS Evaluation Report\n\n")
            f.write(
                "This report contains comprehensive metrics for the RAG pipeline:\n"
            )
            f.write(
                "- **Context Precision**: How relevant are retrieved contexts (higher is better)\n"
            )
            f.write(
                "- **Context Recall**: How well contexts cover ground truth (higher is better)\n"
            )
            f.write(
                "- **Faithfulness**: How grounded answers are in contexts (higher is better)\n"
            )
            f.write(
                "- **Answer Relevancy**: How relevant answers are to questions (higher is better)\n\n"
            )

            # Convert RAGAS results to pandas DataFrame and then to markdown table
            f.write(ragas_scores.to_pandas().to_markdown())

        print("Done! Reports saved in app/evaluation/reports")
        print("- retrieval_report.md: Retrieval quality metrics")
        print("- ragas_report.md: End-to-end pipeline quality metrics")
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        print("Only retrieval evaluation completed successfully")

