# app/core/engine.py
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from app.core.config import (
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_EVAL_TOL,
    DEFAULT_OPENAI_MODEL,
    OPENAI_SYSTEM_PROMPT,
)
from app.core.helpers import (
    build_generation_prompt,
    format_scored_dataframe,
    normalize_response,
)
from app.core.evaluator import evaluate_dataframe
from app.core.schemas import Action, EvaluationResult, GenerationResult
from app.providers import OpenAIProvider


class OpenAIMathEngine:
    def __init__(self, *, model: str = DEFAULT_OPENAI_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.provider = OpenAIProvider(api_key=api_key)

    def available(self) -> bool:
        return self.provider.available()

    def generate(
        self,
        action: Action,
        *,
        examples: Optional[Iterable[Action]] = None,
    ) -> GenerationResult:
        if not self.available():
            return GenerationResult(
                action=action,
                model=self.model,
                error="OpenAI provider unavailable. Set OPENAI_API_KEY.",
            )

        prompt = build_generation_prompt(action.problem, examples=examples)
        res = self.provider.generate(
            model=self.model,
            prompt=prompt,
            system_prompt=OPENAI_SYSTEM_PROMPT,
            temperature=0,
        )
        if res.error or not res.output:
            return GenerationResult(
                action=action,
                model=self.model,
                error=res.error or "Empty model output",
            )

        try:
            parsed = normalize_response(res.output)
            return GenerationResult(action=action, model=self.model, response=parsed)
        except Exception as e:
            return GenerationResult(action=action, model=self.model, error=str(e))

    def run(
        self,
        actions: Iterable[Action],
        *,
        examples: Optional[Iterable[Action]] = None,
        samples: int = DEFAULT_EVAL_SAMPLES,
        tol: float = DEFAULT_EVAL_TOL,
    ) -> tuple[pd.DataFrame, List[GenerationResult]]:
        generation_results: List[GenerationResult] = []
        rows = []

        for action in actions:
            gen = self.generate(action, examples=examples)
            generation_results.append(gen)
            rows.append(
                {
                    "problem_id": action.problem_id,
                    "problem": action.problem,
                    "ai_response": gen.response.ai_response if gen.response else "ERROR",
                    "correct_answer": action.correct_answer,
                    "model": gen.model,
                    "error": gen.error,
                    "meta_note": gen.response.meta_note if gen.response else None,
                }
            )

        df = pd.DataFrame(rows)
        scored = evaluate_dataframe(
            df,
            ai_col="ai_response",
            correct_col="correct_answer",
            samples=samples,
            tol=tol,
        )
        display = format_scored_dataframe(scored)
        return display, generation_results


def dataframe_to_evaluation_results(df: pd.DataFrame) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for row in df.to_dict("records"):
        results.append(
            EvaluationResult(
                problem_id=int(row["problem_id"]),
                problem=str(row["problem"]),
                ai_response=str(row["ai_response"]),
                correct_answer=str(row["correct_answer"]),
                is_correct=str(row["is_correct"]),
                score=float(row["score"]),
                model=str(row.get("model", "")),
                error=row.get("error"),
                meta_note=row.get("meta_note"),
            )
        )
    return results
