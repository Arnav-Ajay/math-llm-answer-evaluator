# app/core/input/parsing.py
from __future__ import annotations
import re
from typing import Iterable

import pandas as pd
from sympy import sympify

from app.core.schemas import Action

REQUIRED_COLUMNS = ("problem", "correct_answer")
_MATH_KEYWORDS = (
    "diff",
    "integrate",
    "expand",
    "factor",
    "simplify",
    "solve",
    "sin",
    "cos",
    "tan",
    "log",
    "exp",
    "sqrt",
    "limit",
)
_ALLOWED_BARE_IDENTIFIERS = {"x", "y", "z", "t", "n", "pi", "e"}


def _looks_like_math_problem(text: str) -> bool:
    s = text.strip()
    lower = s.lower()
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return True
    if any(ch in "+-*/^=()[]{}" for ch in s):
        return True
    if any(k in lower for k in _MATH_KEYWORDS):
        return True
    if re.search(r"\b[xyztn]\b", lower):
        return True
    return False


def _is_valid_math_answer(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    normalized = s.replace("^", "**")

    try:
        if "=" in normalized:
            left, right = [part.strip() for part in normalized.split("=", 1)]
            if not left or not right:
                return False
            sympify(left)
            sympify(right)
            return True

        expr = sympify(normalized)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
            return normalized.lower() in _ALLOWED_BARE_IDENTIFIERS
        _ = expr
        return True
    except Exception:
        return False


def parse_actions_text(raw: str) -> tuple[list[Action], list[str]]:
    actions: list[Action] = []
    issues: list[str] = []
    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]

    for i, line in enumerate(lines, start=1):
        if "||" not in line:
            issues.append(f"Line {i} missing separator '||'.")
            continue
        problem, correct = [part.strip() for part in line.split("||", 1)]
        if not problem or not correct:
            issues.append(f"Line {i} has empty problem or correct answer.")
            continue
        if not _looks_like_math_problem(problem):
            issues.append(f"Line {i} problem does not look like a math prompt.")
            continue
        if not _is_valid_math_answer(correct):
            issues.append(f"Line {i} correct_answer is not a valid math expression/equation.")
            continue
        actions.append(Action(problem_id=len(actions) + 1, problem=problem, correct_answer=correct))

    if not actions and not issues:
        issues.append("Provide at least one problem row.")

    return actions, issues


def dataframe_to_actions(df: pd.DataFrame) -> tuple[list[Action], list[str]]:
    issues: list[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return [], [f"Missing required column(s): {', '.join(missing)}."]

    actions: list[Action] = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        problem = str(row.get("problem", "")).strip()
        correct = str(row.get("correct_answer", "")).strip()
        if not problem or not correct:
            issues.append(f"Row {i} has empty problem or correct_answer.")
            continue
        if not _looks_like_math_problem(problem):
            issues.append(f"Row {i} problem does not look like a math prompt.")
            continue
        if not _is_valid_math_answer(correct):
            issues.append(f"Row {i} correct_answer is not a valid math expression/equation.")
            continue
        actions.append(Action(problem_id=len(actions) + 1, problem=problem, correct_answer=correct))

    if not actions and not issues:
        issues.append("Uploaded file has no usable rows.")
    return actions, issues


def actions_to_text(actions: Iterable[Action]) -> str:
    return "\n".join([f"{a.problem} || {a.correct_answer}" for a in actions])
