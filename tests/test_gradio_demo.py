"""The Gradio page formats a result without calling the model."""

from types import SimpleNamespace

import pytest

pytest.importorskip("spaces")
pytest.importorskip("gradio")

from app import ask, demo, format_result  # noqa: E402


def test_format_result_includes_citation_and_trace():
    result = SimpleNamespace(
        answer="CHAS subsidises GP care [1].",
        refused=False,
        intent="lookup",
        latency_ms=1200.4,
        citation_verification={"valid": True},
        citations=[{"index": 1, "title": "CHAS", "url": "https://www.healthhub.sg/a", "domain": "healthhub.sg"}],
        trace_text="User\n ↓\nFinal Answer",
    )
    answer, meta, citations, trace = format_result(result)
    assert "CHAS subsidises" in answer
    assert "refused: False" in meta
    assert "intent: lookup" in meta
    assert "healthhub.sg" in citations
    assert "Final Answer" in trace


def test_short_question_does_not_load_the_index():
    answer, meta, citations, trace = ask("no", "pipeline")
    assert "longer" in answer
    assert meta == ""
    assert citations == ""
    assert trace == ""


def test_demo_registers_an_ask_handler():
    assert demo is not None
