"""Streamlit research interface for CiteCheck."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from citecheck.agent import run_agent
from citecheck.config import Settings
from citecheck.logging_utils import StepTrace
from citecheck.presentation import render_point_html
from citecheck.schemas import Answer, Source


def _inject_styles() -> None:
    """Apply the app's single light-theme stylesheet."""
    st.markdown(
        """
        <style>
        :root {
            --cc-ink: #172a2c;
            --cc-muted: #5c6e70;
            --cc-accent: #147d78;
            --cc-line: #dce5e4;
            --cc-surface: #ffffff;
            --cc-background: #f4f8f7;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--cc-background); color: var(--cc-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 900px; padding: 2.4rem 1.25rem 4rem; }
        h1, h2, h3, p, label { color: var(--cc-ink); }
        h1 { font-size: 2.25rem; font-weight: 720; margin-bottom: 0.15rem; }
        h2 { font-size: 1.05rem; font-weight: 700; margin: 1.6rem 0 0.6rem; }
        .cc-subtitle { color: var(--cc-muted); font-size: 1rem; margin-bottom: 1.7rem; }
        .cc-section-label {
            color: var(--cc-accent); font-size: 0.72rem; font-weight: 750;
            letter-spacing: 0.08em; text-transform: uppercase; margin: 1.25rem 0 0.35rem;
        }
        .cc-badge {
            display: inline-flex; align-items: center; border-radius: 999px;
            padding: 0.18rem 0.55rem; font-size: 0.72rem; line-height: 1.35;
            font-weight: 700; white-space: nowrap;
        }
        .cc-supported { color: #155d39; background: #e4f4e9; }
        .cc-partial { color: #80530c; background: #fff2d6; }
        .cc-unsupported { color: #8e2b2b; background: #fde9e8; }
        .cc-neutral { color: #40575a; background: #e8eeee; }
        .cc-point { margin: 0.65rem 0; line-height: 1.6; }
        .cc-point-status {
            display: inline-block; border-radius: 999px; padding: 0.12rem 0.48rem;
            margin-right: 0.35rem; font-size: 0.7rem; font-weight: 700;
        }
        .cc-verified { color: #155d39; background: #e4f4e9; }
        .cc-unverified { color: #8e2b2b; background: #fde9e8; }
        a.cc-citation {
            color: var(--cc-accent); font-size: 0.76rem; font-weight: 750;
            text-decoration: none; margin-left: 0.1rem;
        }
        a.cc-citation:hover { text-decoration: underline; }
        .cc-source { scroll-margin-top: 1rem; }
        .cc-source-title { font-weight: 700; }
        .cc-quote { border-left: 3px solid var(--cc-accent); padding-left: 0.8rem; }
        button[kind="primaryFormSubmit"], button[data-testid="stBaseButton-primaryFormSubmit"] {
            background: var(--cc-accent); border-color: var(--cc-accent); color: white;
        }
        button[kind="primaryFormSubmit"]:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background: #0e6965; border-color: #0e6965; color: white;
        }
        [data-testid="stForm"] {
            background: var(--cc-surface); border: 1px solid var(--cc-line);
            padding: 1rem; border-radius: 10px;
        }
        [data-testid="stExpander"] {
            background: var(--cc-surface); border: 1px solid var(--cc-line); border-radius: 9px;
        }
        @media (max-width: 640px) {
            .block-container { padding: 1.4rem 0.85rem 2.5rem; }
            h1 { font-size: 1.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[Source]) -> None:
    """Render each fetched source and its supporting quote."""
    st.markdown('<div class="cc-section-label">Sources</div>', unsafe_allow_html=True)
    if not sources:
        st.info("No relevant sources were retrieved for this answer.")
        return
    for source in sources:
        safe_title = html.escape(source.title or source.url)
        safe_url = html.escape(source.url, quote=True)
        st.markdown(
            f'<div id="source-{source.id}" class="cc-source">'
            f'<span class="cc-badge cc-neutral">[{source.id}]</span> '
            f'<span class="cc-source-title"><a href="{safe_url}" target="_blank">'
            f"{safe_title}</a></span></div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Supporting quote [{source.id}]"):
            st.markdown(
                f'<div class="cc-quote">{html.escape(source.quote)}</div>',
                unsafe_allow_html=True,
            )


def _render_results(question: str, answer: Answer, traces: list[dict[str, Any]]) -> None:
    """Render a structured answer, source links, and the agent trace."""
    st.markdown('<div class="cc-section-label">Question</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"### {html.escape(question)}")

    st.markdown('<div class="cc-section-label">Answer</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(answer.summary)
        source_map = {source.id: source for source in answer.sources}
        for section in answer.sections:
            st.markdown(f"#### {html.escape(section.heading)}")
            for point in section.points:
                st.markdown(render_point_html(point, source_map), unsafe_allow_html=True)
        if not answer.sections:
            st.info("No relevant points could be supported by fetched sources.")

    _render_sources(answer.sources)
    with st.expander("Agent trace"):
        if traces:
            for trace in traces:
                st.markdown(
                    f"**Step {trace['step']} · {html.escape(trace['tool'])}** "
                    f"({trace['latency_ms']:.0f} ms)"
                )
                st.code(str(trace["args"]), language="json")
        else:
            st.caption("No trace steps were recorded for this run.")


st.set_page_config(page_title="CiteCheck", page_icon="C", layout="centered")
_inject_styles()
st.title("CiteCheck")
st.markdown(
    '<div class="cc-subtitle">Research answers grounded in sources you can inspect.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="cc-section-label">Question</div>', unsafe_allow_html=True)
with st.form("research_question", clear_on_submit=False):
    question_input = st.text_input(
        "Research question",
        placeholder="Ask a focused research question",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", type="primary")

defaults = Settings.from_environment()
with st.expander("Advanced (experiment settings)"):
    model = st.text_input("Model", value=defaults.model)
    embed_model = st.text_input("Embedding model", value=defaults.embed_model)
    max_steps = st.number_input(
        "Maximum steps", min_value=1, max_value=30, value=defaults.max_steps
    )
    top_k = st.number_input("Evidence passages", min_value=1, max_value=20, value=defaults.top_k)
    retrieval_enabled = st.toggle("Semantic retrieval", value=defaults.retrieval_enabled)

if submitted:
    if not question_input.strip():
        st.warning("Enter a research question to begin.")
    else:
        settings = Settings(
            model=model,
            embed_model=embed_model,
            max_steps=int(max_steps),
            top_k=int(top_k),
            retrieval_enabled=retrieval_enabled,
            verifier_enabled=True,
        )
        trace_rows: list[dict[str, Any]] = []
        current_step = st.empty()

        def display_trace(path: str, trace: StepTrace) -> None:
            """Capture JSONL-compatible trace data and show the active operation."""
            trace_data = trace.model_dump(mode="json")
            trace_rows.append(
                {
                    "step": trace_data["step"],
                    "tool": trace_data["tool"],
                    "args": trace_data["args"],
                    "latency_ms": trace_data["latency_ms"],
                    "result_length": trace_data["result_length"],
                }
            )
            current_step.caption(f"Current step: {trace.tool} · {trace.latency_ms:.0f} ms")

        try:
            with st.spinner("Researching and checking evidence…"):
                result = run_agent(question_input.strip(), settings, trace_fn=display_trace)
            st.session_state["citecheck_result"] = result
            st.session_state["citecheck_question"] = question_input.strip()
            st.session_state["citecheck_trace"] = trace_rows
        except Exception as exc:
            message = str(exc)
            if "connect" in message.casefold() or "ollama" in message.casefold():
                st.error(
                    "Could not reach Ollama. Start Ollama, confirm it is listening locally, "
                    "and check that the selected model is installed."
                )
            else:
                st.error(f"Research failed: {message}")

if "citecheck_result" in st.session_state and "citecheck_question" in st.session_state:
    _render_results(
        st.session_state["citecheck_question"],
        st.session_state["citecheck_result"],
        st.session_state.get("citecheck_trace", []),
    )
