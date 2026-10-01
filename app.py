"""Gradio demo for a free Hugging Face ZeroGPU Space.

Answers run on CPU and call OpenAI. The GPU button is registered only because
ZeroGPU refuses to start without one @spaces.GPU handler. Asking a question
does not request a GPU.
"""

from __future__ import annotations

import os

import spaces
import gradio as gr

_pipeline = None


def _load():
    """Load the index and reranker once. On Spaces this runs during startup."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import RAGPipeline
        from src.retrieval.rerank import _get_cross_encoder

        _pipeline = RAGPipeline()
        _get_cross_encoder()
    return _pipeline


def format_result(result) -> tuple[str, str, str, str]:
    check = result.citation_verification or {}
    meta = (
        f"refused: {result.refused} · intent: {result.intent or '—'} · "
        f"{result.latency_ms:.0f} ms · citations valid: {check.get('valid')}"
    )
    lines = []
    for cite in result.citations or []:
        index = cite.get("index")
        title = cite.get("title") or cite.get("domain") or "source"
        domain = cite.get("domain") or ""
        url = cite.get("url")
        label = f"[{index}] {title}"
        if url:
            lines.append(f"- [{label}]({url}) ({domain})")
        else:
            lines.append(f"- {label} ({domain})")
    return result.answer or "", meta, "\n".join(lines), result.trace_text or ""


def ask(question: str, mode: str) -> tuple[str, str, str, str]:
    text = (question or "").strip()
    if len(text) < 3:
        return "Ask a slightly longer question.", "", "", ""
    try:
        pipeline = _load()
        if mode == "agent":
            from src.agent.loop import PolicyAgent

            result = PolicyAgent(retriever=pipeline.retriever).query(text)
        elif mode == "multi":
            from src.agent.workflow import MultiAgentWorkflow

            result = MultiAgentWorkflow(retriever=pipeline.retriever).query(text)
        else:
            result = pipeline.query(text)
    except Exception as exc:  # noqa: BLE001 — show the failure on the page
        return f"Could not answer: {exc}", "", "", ""
    return format_result(result)


@spaces.GPU(duration=15)
def gpu_registered():
    """Registered so ZeroGPU will boot. Not used for answers."""
    return "registered"


with gr.Blocks(title="CarePolicy") as demo:
    gr.Markdown(
        """
# CarePolicy

Ask about Singapore public healthcare policy. Answers come from HealthHub, MOH, HPB, and PDPC pages, with citations I can check.

Educational project. Not medical or legal advice. `multi` can take about 20 seconds because it searches more than once. The first question after a cold start also waits while the index loads.
        """
    )
    question = gr.Textbox(
        label="Question",
        value="What is CHAS and who is eligible?",
        lines=3,
    )
    mode = gr.Radio(
        choices=[
            ("pipeline — one search", "pipeline"),
            ("agent — the model picks the searches", "agent"),
            ("multi — route, then specialists", "multi"),
        ],
        value="multi",
        label="Mode",
    )
    ask_btn = gr.Button("Ask")
    meta = gr.Markdown()
    answer = gr.Markdown()
    citations = gr.Markdown()
    trace = gr.Code(label="Trace", language=None)
    gr.Examples(
        examples=[
            ["What is CHAS and who is eligible?", "pipeline"],
            ["Compare CHAS and MediShield Life: who is eligible for each?", "multi"],
            ["What is the stock price of Apple Inc today?", "pipeline"],
        ],
        inputs=[question, mode],
    )
    ask_btn.click(
        ask,
        inputs=[question, mode],
        outputs=[answer, meta, citations, trace],
        concurrency_limit=2,
    )
    # Hidden. ZeroGPU scans event handlers and requires one GPU function.
    probe = gr.Button(visible=False)
    probe_out = gr.Textbox(visible=False)
    probe.click(gpu_registered, outputs=probe_out)

if os.environ.get("SPACE_ID"):
    _load()

if __name__ == "__main__":
    demo.launch()
