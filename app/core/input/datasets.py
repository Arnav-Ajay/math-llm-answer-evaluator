# app/core/input/datasets.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.input.parsing import dataframe_to_actions
from app.core.schemas import Action

_ROOT = Path(__file__).resolve().parents[3]
_DATASETS = {
    "GSM8K-lite": _ROOT / "data" / "benchmarks" / "gsm8k_lite.csv",
    "Symbolic Algebra": _ROOT / "data" / "benchmarks" / "symbolic_algebra.csv",
    "Calculus": _ROOT / "data" / "benchmarks" / "calculus.csv",
}


def available_benchmark_datasets() -> list[str]:
    return list(_DATASETS.keys())


def load_benchmark_dataset(name: str) -> tuple[list[Action], list[str]]:
    path = _DATASETS.get(name)
    if path is None:
        return [], [f"Unknown dataset: {name}"]
    if not path.exists():
        return [], [f"Dataset file not found: {path}"]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return [], [f"Failed to read dataset '{name}': {e}"]

    actions, issues = dataframe_to_actions(df)
    return actions, issues
