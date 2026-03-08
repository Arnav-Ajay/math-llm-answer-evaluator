# examples/sample_run.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.engine import OpenAIMathEngine, dataframe_to_evaluation_results
from app.core.schemas import Action


MODEL = "gpt-4o-mini"

FEW_SHOT_EXAMPLES = [
    {"problem": "(x+2)**2", "answer": "x**2 + 4*x + 4"},
    {"problem": "(x-1)**2", "answer": "x**2 - 2*x + 1"},
    {"problem": "(2*x+5)**2", "answer": "4*x**2 + 20*x + 25"},
    {"problem": "(x-5)*(x+5)", "answer": "x**2 - 25"},
    {"problem": "diff(5*x**4, x)", "answer": "20*x**3"},
    {"problem": "diff(7*x**2 + 3*x + 1, x)", "answer": "14*x + 3"},
    {"problem": "diff(x**3 + x**2, x)", "answer": "3*x**2 + 2*x"},
    {"problem": "x + 5 = 9", "answer": "x=4"},
    {"problem": "2*x - 6 = 0", "answer": "x=3"},
    {"problem": "4*x + 8 = 0", "answer": "x=-2"},

    ############# Commented to test on unseen problems ##############

    # {"problem": "expand((x+3)*(x+4))", "answer": "x**2 + 7*x + 12"},
    # {"problem": "expand((x+7)*(x+1))", "answer": "x**2 + 8*x + 7"},
    # {"problem": "integrate(2*x, x)", "answer": "x**2"},
    # {"problem": "integrate(6*x**5, x)", "answer": "x**6"}
]

def _build_data() -> List[Dict[str, str | None]]:
    return [
        {
            "problem_id": 1,
            "problem": "(x-3)**2",
            "ai_response": None,
            "correct_answer": "x**2 - 6*x + 9",
        },
        {
            "problem_id": 2,
            "problem": "(x+4)*(x+2)",
            "ai_response": None,
            "correct_answer": "20*x**4 - 2*x",
        },
        {
            "problem_id": 3,
            "problem": "diff(4*x**5 - x**2, x)",
            "ai_response": None,
            "correct_answer": "20*x**4 - 2*x",
        },
        {
            "problem_id": 4,
            "problem": "diff(x**2, x)",
            "ai_response": None,
            "correct_answer": "2*x",
        },
        {
            "problem_id": 5,
            "problem": "3*x + 9 = 0",
            "ai_response": None,
            "correct_answer": "x=-3",
        },
        {
            "problem_id": 6,
            "problem": "5*x - 15 = 0",
            "ai_response": None,
            "correct_answer": "x=3",
        },
        {
            "problem_id": 7,
            "problem": "expand((x-2)*(x+6))",
            "ai_response": None,
            "correct_answer": "x**2 + 4*x - 12",
        },
        {
            "problem_id": 8,
            "problem": "expand((2*x+1)*(x+3))",
            "ai_response": None,
            "correct_answer": "2*x**2 + 7*x + 3",
        },
        {
            "problem_id": 9,
            "problem": "integrate(3*x**2, x)",
            "ai_response": None,
            "correct_answer": "x**3",
        },
        {
            "problem_id": 10,
            "problem": "integrate(4*x**3, x)",
            "ai_response": None,
            "correct_answer": "x**4",
        }
    ]


def _to_actions(data: List[Dict[str, str | None]]) -> List[Action]:
    return [
        Action(
            problem_id=int(row["problem_id"]),
            problem=str(row["problem"]),
            correct_answer=str(row["correct_answer"]),
        )
        for row in data
    ]


def _to_example_actions() -> List[Action]:
    return [
        Action(problem_id=i + 1, problem=ex["problem"], correct_answer=ex["answer"])
        for i, ex in enumerate(FEW_SHOT_EXAMPLES)
    ]


def main() -> None:
    load_dotenv(override=True)
    print(f"Generating responses with OpenAI model: {MODEL}")

    engine = OpenAIMathEngine(model=MODEL)
    if not engine.available():
        raise RuntimeError("OpenAI provider unavailable. Set a valid OPENAI_API_KEY in .env.")

    actions = _to_actions(_build_data())
    example_actions = _to_example_actions()
    display, generation_results = engine.run(actions, examples=example_actions)

    errors = [r for r in generation_results if r.error]
    if errors:
        msg = "; ".join(f"problem_id={r.action.problem_id}: {r.error}" for r in errors)
        raise RuntimeError(f"OpenAI generation failed: {msg}")

    _ = dataframe_to_evaluation_results(display)
    print(display[["problem_id", "problem", "ai_response", "correct_answer", "is_correct", "score"]])


if __name__ == "__main__":
    main()
