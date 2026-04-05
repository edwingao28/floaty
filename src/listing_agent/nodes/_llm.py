"""Shared LLM utilities for nodes."""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from listing_agent.config import get_config


def invoke_with_fallback(prompt: str) -> str:
    """Try primary model, fall back to ANTHROPIC_FALLBACK_MODEL on error."""
    config = get_config()
    try:
        llm = ChatAnthropic(model=config.ANTHROPIC_MODEL, temperature=0)
        return str(llm.invoke(prompt).content)
    except Exception:
        llm = ChatAnthropic(model=config.ANTHROPIC_FALLBACK_MODEL, temperature=0)
        return str(llm.invoke(prompt).content)
