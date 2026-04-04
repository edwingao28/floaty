import os
from typing import Any

from listing_agent.rag.retriever import PlatformRetriever
from listing_agent.state import AgentState

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "rag", "knowledge_base")
_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", ".chroma")


def research_platforms(state: AgentState) -> dict[str, Any]:
    """LangGraph node: for each target platform, RAG-retrieve relevant listing rules."""
    retriever = PlatformRetriever(
        knowledge_dir=_KNOWLEDGE_DIR,
        persist_dir=_PERSIST_DIR,
    )

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
        except ValueError as e:
            platform_rules[platform] = f"No rules available: {e}"

    return {"platform_rules": platform_rules}
