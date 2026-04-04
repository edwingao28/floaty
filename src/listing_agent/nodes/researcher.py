import os
from pathlib import Path
from typing import Any

from listing_agent.rag.retriever import PlatformRetriever
from listing_agent.state import AgentState

_KNOWLEDGE_DIR = Path(__file__).parent / ".." / "rag" / "knowledge_base"
_retriever: PlatformRetriever | None = None


def _get_retriever() -> PlatformRetriever:
    global _retriever
    if _retriever is None:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
        _retriever = PlatformRetriever(
            knowledge_dir=str(_KNOWLEDGE_DIR),
            persist_dir=persist_dir,
        )
    return _retriever


def research_platforms(state: AgentState) -> dict[str, Any]:
    """LangGraph node: for each target platform, RAG-retrieve relevant listing rules."""
    retriever = _get_retriever()

    # Build product description from attributes or raw data
    attrs = state.get("product_attributes")
    if attrs:
        product_desc = f"{attrs.title} - {', '.join(attrs.features)}"
    else:
        raw = state["raw_product_data"]
        product_desc = raw.get("description", str(raw))

    platform_rules: dict[str, str] = {}

    for platform in state["target_platforms"]:
        try:
            platform_rules[platform] = retriever.get_rules(platform, product_desc)
        except Exception as e:
            platform_rules[platform] = f"No rules available: {e}"

    return {"platform_rules": platform_rules}
