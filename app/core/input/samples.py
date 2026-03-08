# app/core/input/samples.py
from __future__ import annotations

SAMPLE_PROBLEM_ROWS = [
    ("integrate(4*x**3, x)", "x**4"),
    ("expand((x+1)**2)", "x**2 + 2*x + 1"),
    ("diff(x**3, x)", "3*x**2"),
]

def sample_rows_text() -> str:
    return "\n".join([f"{problem} || {correct}" for problem, correct in SAMPLE_PROBLEM_ROWS])
