# Evaluation Module

This directory contains the evaluation framework for the RAG (Retrieval-Augmented Generation) pipeline. The evaluation system assesses both retrieval quality and end-to-end answer generation quality.

## Files Overview

### `eval_questions.json`

Contains evaluation questions and expected answers for testing the RAG system. Each entry has:

- `question`: The test question to ask the RAG system
- `expected`: The expected answer or key phrase that should be found

### `retrieval_eval.py`

Evaluates the retrieval component using Recall@K metrics. This measures how often the expected answer appears in the top K retrieved document chunks.

**Key Function**: `evaluate_retrieval()`

- Tests vector search quality
- Returns binary recall scores (1 = found, 0 = not found)

### `answer_eval_ragas.py`

Performs comprehensive evaluation using the RAGAS framework, which assesses four key metrics:

- **Context Precision**: Relevance of retrieved contexts
- **Context Recall**: Coverage of ground truth in contexts
- **Faithfulness**: How grounded answers are in retrieved contexts
- **Answer Relevancy**: Relevance of generated answers to questions

**Key Function**: `run_ragas()`

- Evaluates end-to-end pipeline quality
- Returns detailed metrics for each aspect

### `eval_runner.py`

Orchestrates all evaluations and generates reports.

**Key Function**: `run_all_evaluations()`

- Runs both retrieval and RAGAS evaluations
- Generates markdown reports in `app/evaluation/reports/`

## Usage

Run complete evaluation:

```python
from app.evaluation.eval_runner import run_all_evaluations
run_all_evaluations()
```

Run individual evaluations:

```python
# Retrieval only
from app.evaluation.retrieval_eval import evaluate_retrieval
results = evaluate_retrieval()

# RAGAS only
from app.evaluation.answer_eval_ragas import run_ragas
scores = run_ragas()
```

## Output

Evaluation reports are saved to `app/evaluation/reports/`:

- `retrieval_report.md`: Recall@5 scores per question
- `ragas_report.md`: Comprehensive RAGAS metrics table

## Requirements

### Core Requirements (Always Available)
- All dependencies in `requirements.txt`
- Retrieval evaluation works out of the box

### Optional Requirements (For RAGAS Evaluation)
The RAGAS evaluation requires additional dependencies:
- `ragas`: RAGAS evaluation framework
- `datasets`: HuggingFace datasets for RAGAS
- `git`: Git executable (required by RAGAS)

**Installation:**
```bash
pip install ragas datasets
```

**Note:** If you encounter git-related errors, ensure git is properly installed and accessible in your PATH.

### Graceful Degradation
The evaluation system is designed to work gracefully:
- If RAGAS dependencies are missing, only retrieval evaluation runs
- Clear error messages guide users on how to enable full evaluation
- The system never crashes due to missing optional dependencies

## Common Issues

### Git Executable Error
If you see `Bad git executable` error when using RAGAS:
1. Install git: `apt-get install git` (Ubuntu) or `brew install git` (macOS)
2. Or set the git path: `export GIT_PYTHON_GIT_EXECUTABLE=/path/to/git`

### Missing Dependencies
The system automatically detects missing dependencies and provides helpful installation instructions.
