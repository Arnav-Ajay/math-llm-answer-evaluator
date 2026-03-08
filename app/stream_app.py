# app/stream_app.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DEFAULT_EVAL_SAMPLES, DEFAULT_EVAL_TOL, DEFAULT_OPENAI_MODEL
from app.core.engine import OpenAIMathEngine, dataframe_to_evaluation_results
from app.core.schemas import Action
from app.providers import OpenAIProvider

load_dotenv(override=True)

st.set_page_config(page_title="Math Reasoning Evaluator", layout="wide")

st.title("Mathematical Reasoning Evaluator")
st.caption("Enter problems and correct answers, then generate LLM answers and score them.")


@st.cache_data
def _available_models() -> List[str]:
    return OpenAIProvider.default_models()


def _parse_actions(raw: str) -> List[Action]:
    actions: List[Action] = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for i, line in enumerate(lines, start=1):
        if "||" not in line:
            raise ValueError(f"Line {i} must use 'problem || correct_answer' format.")
        problem, correct = [part.strip() for part in line.split("||", 1)]
        if not problem or not correct:
            raise ValueError(f"Line {i} has empty problem or correct answer.")
        actions.append(Action(problem_id=i, problem=problem, correct_answer=correct))
    if not actions:
        raise ValueError("Provide at least one problem row.")
    return actions


with st.form("run_form"):
    model = st.selectbox("Model", options=_available_models(), index=0)
    samples = st.slider("Partial scoring samples", min_value=1, max_value=20, value=DEFAULT_EVAL_SAMPLES)
    tol = st.number_input("Tolerance", min_value=1e-9, max_value=1e-3, value=DEFAULT_EVAL_TOL, format="%.1e")
    raw_rows = st.text_area(
        "Rows (one per line): problem || correct_answer",
        value="",
        placeholder="(x-3)**2 || x**2 - 6*x + 9\n(x+4)*(x+2) || 20*x**4 - 2*x",
        height=220,
    )
    submitted = st.form_submit_button("Generate and Evaluate", type="primary")

if submitted:
    try:
        actions = _parse_actions(raw_rows)
    except ValueError as e:
        st.error(str(e))
    else:
        engine = OpenAIMathEngine(model=model or DEFAULT_OPENAI_MODEL)
        if not engine.available():
            st.error("OpenAI provider unavailable. Set a valid OPENAI_API_KEY in .env.")
        else:
            with st.spinner("Generating LLM responses and evaluating..."):
                display, generation_results = engine.run(
                    actions,
                    samples=int(samples),
                    tol=float(tol),
                )
            errors = [r for r in generation_results if r.error]
            if errors:
                for err in errors:
                    st.error(f"problem_id={err.action.problem_id}: {err.error}")

            result_view = display[
                ["problem_id", "problem", "ai_response", "correct_answer", "is_correct", "score"]
            ].copy()
            result_view["score"] = result_view["score"].map(lambda x: f"{x:.3f}")
            st.dataframe(result_view, use_container_width=True)

            _ = dataframe_to_evaluation_results(display)
            csv = result_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
            )
