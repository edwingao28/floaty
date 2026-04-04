import json
from typing import Any

from langchain_anthropic import ChatAnthropic

from listing_agent.state import AgentState


_ANALYZE_PROMPT = """You are a product categorization expert.

Analyze this product description and extract structured attributes as JSON.

Product: {product_input}

Return ONLY valid JSON with this exact structure:
{{
  "name": "product name (clean, title-cased)",
  "category": "one of: home_and_kitchen, clothing, electronics, beauty, sports, toys, books, other",
  "features": ["feature 1", "feature 2", "..."],
  "materials": ["material 1", "..."],
  "target_audience": "describe in one sentence",
  "price_range": "budget (<$15), mid-range ($15-$50), premium ($50-$150), luxury (>$150)"
}}"""


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def analyze_product(state: AgentState) -> dict[str, Any]:
    """LangGraph node: parse raw_product_data → product_attributes."""
    llm = get_llm()
    # Extract description from raw_product_data
    raw = state["raw_product_data"]
    product_input = raw.get("description", str(raw))
    prompt = _ANALYZE_PROMPT.format(product_input=product_input)

    try:
        response = llm.invoke(prompt)
        attributes = json.loads(response.content)
        return {"product_attributes": attributes}
    except (json.JSONDecodeError, KeyError) as e:
        return {"errors": [f"Failed to parse product attributes: {e}"]}
