"""Generate the final answer with an LLM, grounded in the retrieved chunks."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.citations import format_context
from src.generation.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE, REFUSAL_MESSAGE
from src.llm_provider import get_chat_model

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@traceable(name="generate_answer", run_type="llm")
def generate_answer(question: str, chunks: list[dict]) -> str:
    """Ask the LLM to answer using only the given chunks, citing them as [1], [2], ..."""
    if not chunks:
        return REFUSAL_MESSAGE

    context = format_context(chunks)
    prompt = ANSWER_USER_TEMPLATE.format(context=context, question=question)

    model = get_chat_model()
    if model is None:
        return _extractive_fallback(chunks)

    response = model.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return response.content.strip()


def _extractive_fallback(chunks: list[dict]) -> str:
    """Local fallback when no LLM provider is configured."""
    top = chunks[0]
    snippet = top["text"][:500].rsplit(" ", 1)[0]
    return f"Based on the available sources, {snippet}... [1]"
