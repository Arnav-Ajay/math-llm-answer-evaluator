# app/core/config.py
from __future__ import annotations

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_MODELS = [
    "gpt-4o-mini",
    "o1-mini",
]

DEFAULT_EVAL_SAMPLES = 8
DEFAULT_EVAL_TOL = 1e-6
CORRECT_THRESHOLD = 0.9999

OPENAI_SYSTEM_PROMPT = (
    "You are a precise math solver. "
    "Return only the final SymPy-compatible expression or equation. No prose."
)
