from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.benchmark import run_openai_model_benchmark
from app.core.config import DEFAULT_EVAL_SAMPLES, DEFAULT_EVAL_TOL
from app.core.input import (
    actions_to_text,
    available_benchmark_datasets,
    dataframe_to_actions,
    load_benchmark_dataset,
    parse_actions_text,
    sample_rows_text,
)
from app.providers import OpenAIProvider

load_dotenv(override=True)

st.set_page_config(page_title="Math Reasoning Evaluator", layout="wide")

st.title("Mathematical Reasoning Evaluator")
st.caption("Symbolically verifies LLM math answers using SymPy.")


@st.cache_data
def _available_models() -> List[str]:
    return OpenAIProvider.default_models()


def _load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Use CSV or Excel.")


def _render_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    if len(summary_df) == 1:
        row = summary_df.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{float(row['accuracy']):.1%}")
        m2.metric("Correct", f"{int(row['correct'])} / {int(row['total'])}")
        m3.metric("Avg Score", f"{float(row['avg_score']):.3f}")
        m4.metric("Model", str(row["model"]))
    else:
        view = summary_df.copy()
        view["accuracy"] = view["accuracy"].map(lambda x: f"{x:.1%}")
        view["avg_score"] = view["avg_score"].map(lambda x: f"{x:.3f}")
        st.subheader("Model Accuracy")
        st.dataframe(view[["model", "accuracy", "correct", "total", "avg_score", "errors"]], width="stretch")


def _render_details(details_df: pd.DataFrame, *, csv_name: str) -> None:
    if details_df.empty:
        st.warning("No results to display.")
        return

    view_cols = ["model", "problem_id", "problem", "ai_response", "correct_answer", "is_correct", "score"]
    result_view = details_df[view_cols].copy()
    result_view["score"] = result_view["score"].map(lambda x: f"{x:.3f}")
    styled = result_view.style.map(
        lambda v: "color: #2e7d32; font-weight: 600" if v == "OK" else "color: #c62828; font-weight: 600",
        subset=["is_correct"],
    )
    st.dataframe(styled, width="stretch")
    st.download_button(
        "Download CSV",
        data=result_view.to_csv(index=False).encode("utf-8"),
        file_name=csv_name,
        mime="text/csv",
    )


if "raw_rows" not in st.session_state:
    st.session_state.raw_rows = ""
if "last_upload_token" not in st.session_state:
    st.session_state.last_upload_token = ""

tabs = st.tabs(["Custom Input", "Dataset Runner"])

with tabs[0]:
    col_load, _ = st.columns([1, 4])
    with col_load:
        if st.button("Load Sample Problems"):
            st.session_state.raw_rows = sample_rows_text()

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel (columns: problem, correct_answer)",
        type=["csv", "xlsx", "xls"],
        key="custom_upload",
    )
    if uploaded_file is not None:
        upload_token = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.last_upload_token != upload_token:
            try:
                uploaded_df = _load_uploaded_dataframe(uploaded_file)
                uploaded_actions, uploaded_issues = dataframe_to_actions(uploaded_df)
                for issue in uploaded_issues:
                    st.warning(issue)
                if uploaded_actions:
                    st.session_state.raw_rows = actions_to_text(uploaded_actions)
                    st.success(f"Loaded {len(uploaded_actions)} problem(s) from {uploaded_file.name}.")
                st.session_state.last_upload_token = upload_token
            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}")

    with st.form("custom_form"):
        models = st.multiselect("Models", options=_available_models(), default=_available_models()[:1])
        samples = st.slider("Partial scoring samples", min_value=1, max_value=20, value=DEFAULT_EVAL_SAMPLES)
        tol = st.number_input("Tolerance", min_value=1e-9, max_value=1e-3, value=DEFAULT_EVAL_TOL, format="%.1e")
        raw_rows = st.text_area(
            "Rows (one per line): problem || correct_answer",
            key="raw_rows",
            placeholder="(x-3)**2 || x**2 - 6*x + 9\n(x+4)*(x+2) || 20*x**4 - 2*x",
            height=220,
        )
        submitted_custom = st.form_submit_button("Run Custom Evaluation", type="primary")

    preview_actions, preview_issues = parse_actions_text(st.session_state.raw_rows)
    st.write(f"Detected: {len(preview_actions)} problem(s)")
    for issue in preview_issues:
        st.warning(issue)

    if submitted_custom:
        actions, issues = parse_actions_text(raw_rows)
        if not models:
            st.error("Select at least one model.")
        elif issues:
            for issue in issues:
                st.error(issue)
        else:
            with st.spinner("Running benchmark across selected model(s)..."):
                summary_df, details_df, bench_issues = run_openai_model_benchmark(
                    actions,
                    models,
                    samples=int(samples),
                    tol=float(tol),
                )
            for issue in bench_issues:
                st.warning(issue)
            _render_summary(summary_df)
            _render_details(details_df, csv_name="custom_evaluation_results.csv")

with tabs[1]:
    st.subheader("Run Benchmark Dataset")
    dataset_name = st.selectbox("Benchmark Dataset", options=available_benchmark_datasets(), index=0)
    bench_models = st.multiselect(
        "Models",
        options=_available_models(),
        default=_available_models()[:1],
        key="bench_models",
    )
    bench_samples = st.slider(
        "Partial scoring samples",
        min_value=1,
        max_value=20,
        value=DEFAULT_EVAL_SAMPLES,
        key="bench_samples",
    )
    bench_tol = st.number_input(
        "Tolerance",
        min_value=1e-9,
        max_value=1e-3,
        value=DEFAULT_EVAL_TOL,
        format="%.1e",
        key="bench_tol",
    )

    if st.button("Run Dataset Benchmark", type="primary"):
        if not bench_models:
            st.error("Select at least one model.")
        else:
            actions, load_issues = load_benchmark_dataset(dataset_name)
            if load_issues:
                for issue in load_issues:
                    st.error(issue)
            else:
                st.write(f"Loaded {len(actions)} problem(s) from `{dataset_name}`.")
                with st.spinner("Running dataset benchmark..."):
                    summary_df, details_df, bench_issues = run_openai_model_benchmark(
                        actions,
                        bench_models,
                        samples=int(bench_samples),
                        tol=float(bench_tol),
                    )
                for issue in bench_issues:
                    st.warning(issue)
                _render_summary(summary_df)
                _render_details(details_df, csv_name=f"{dataset_name.lower().replace(' ', '_')}_benchmark.csv")
