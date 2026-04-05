import json
from typing import Any

from pydantic import ValidationError

from listing_agent.nodes._llm import invoke_with_fallback
from listing_agent.state import AgentState, ProductAttributes


_ANALYZE_PROMPT_TEMPLATE = """You are a product categorization expert.

Analyze this product description and extract structured attributes as JSON.

Product: PRODUCT_INPUT_PLACEHOLDER

Return ONLY valid JSON with this exact structure:
{
  "title": "product name (clean, title-cased)",
  "category": "one of: home_and_kitchen, clothing, electronics, beauty, sports, toys, books, other",
  "features": ["feature 1", "feature 2", "..."],
  "materials": ["material 1", "..."],
  "target_audience": "describe in one sentence",
  "price_range": "budget (<$15), mid-range ($15-$50), premium ($50-$150), luxury (>$150)",
  "brand": "brand name or empty string",
  "keywords": ["keyword1", "keyword2", "..."]
}"""


def analyze_product(state: AgentState) -> dict[str, Any]:
    """LangGraph node: parse raw_product_data → product_attributes."""
    try:
        raw = state["raw_product_data"]
        product_input = raw.get("description", str(raw))
        prompt = _ANALYZE_PROMPT_TEMPLATE.replace("PRODUCT_INPUT_PLACEHOLDER", product_input)
        content = invoke_with_fallback(prompt)
        attributes = json.loads(content)
        attributes["raw_input"] = product_input
        pa = ProductAttributes(**attributes)
        return {"product_attributes": pa}
    except (json.JSONDecodeError, KeyError, ValidationError) as e:
        return {"errors": [f"Failed to parse product attributes: {e}"]}
    except Exception as e:
        return {"errors": [f"Failed to analyze product: {e}"]}
