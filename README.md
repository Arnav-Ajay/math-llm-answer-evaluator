# Mathematical Reasoning Evaluator

A lightweight evaluator for checking LLM-generated math answers with SymPy.

The project has:
- A Streamlit app for interactive runs
- A sample runner script for reproducible CLI demos
- A core engine that generates answers via OpenAI and scores correctness

## Features

- LLM-generated `ai_response` (OpenAI provider)
- Symbolic equivalence checking with SymPy
- Partial-credit scoring via randomized substitution
- Tabular results with:
  - `problem_id`
  - `problem`
  - `ai_response`
  - `correct_answer`
  - `is_correct`
  - `score`
- CSV export in Streamlit

## Project Structure

```text
app/
  core/
    config.py
    schemas.py
    helpers.py
    evaluator.py
    engine.py
  providers/
    base.py
    openai_provider.py
  stream_app.py
examples/
  sample_run.py
```

## Requirements

- Python 3.10+
- OpenAI API key

## Setup

Create and activate a virtual environment

  ```bash
  python -m venv .venv
  ```

Windows PowerShell:

  ```powershell
  .venv\Scripts\Activate.ps1
  ```


Configure environment

  ```bash
  cp .env.example .env
  ```

Set:

  ```env
  OPENAI_API_KEY=your_key_here
  ```

Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Run Streamlit App

```bash
streamlit run app/stream_app.py
```

Input format in the app:
- One row per line
- `problem || correct_answer`

Example:

```text
(x-3)**2 || x**2 - 6*x + 9
(x+4)*(x+2) || 20*x**4 - 2*x
```

## Run Sample Script

```bash
python examples/sample_run.py
```

or

```bash
python -m examples.sample_run
```

## Programmatic Usage

```python
import pandas as pd
from app.core.evaluator import evaluate_dataframe

df = pd.DataFrame(
    [
        {
            "problem_id": 1,
            "problem": "(x+1)**2",
            "ai_response": "x**2 + 2*x + 1",
            "correct_answer": "x**2 + 2*x + 1",
        },
        {
            "problem_id": 2,
            "problem": "diff(3*x**3, x)",
            "ai_response": "9*x**2",
            "correct_answer": "9*x**2",
        },
    ]
)

scored = evaluate_dataframe(df, ai_col="ai_response", correct_col="correct_answer")
print(scored[["problem_id", "problem", "ai_response", "correct_answer", "is_correct", "score"]])
```

## Scoring Notes

- Exact symbolic equality is attempted first.
- If exact match fails, expressions/equations are numerically sampled.
- `is_correct` is `OK` when `score >= 0.9999`, else `X`.