from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.core.config import DEFAULT_EVAL_SAMPLES, DEFAULT_EVAL_TOL
from app.core.engine import OpenAIMathEngine
from app.core.schemas import Action


def run_openai_model_benchmark(
    actions: Iterable[Action],
    models: list[str],
    *,
    samples: int = DEFAULT_EVAL_SAMPLES,
    tol: float = DEFAULT_EVAL_TOL,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    actions_list = list(actions)
    summary_rows: list[dict] = []
    detail_frames: list[pd.DataFrame] = []
    issues: list[str] = []

    for model in models:
        engine = OpenAIMathEngine(model=model)
        if not engine.available():
            issues.append(f"{model}: provider unavailable")
            continue

        display, generation_results = engine.run(actions_list, samples=samples, tol=tol)
        errors = [r for r in generation_results if r.error]
        if errors:
            issues.append(f"{model}: {len(errors)} generation error(s)")

        total = len(display)
        correct = int((display["is_correct"] == "OK").sum())
        accuracy = (correct / total) if total else 0.0
        avg_score = float(display["score"].mean()) if total else 0.0

        summary_rows.append(
            {
                "model": model,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "avg_score": avg_score,
                "errors": len(errors),
            }
        )

        detail = display.copy()
        detail["model"] = model
        detail_frames.append(detail)

    summary_df = pd.DataFrame(summary_rows)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_df, details_df, issues
