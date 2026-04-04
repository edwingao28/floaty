import json
from typing import Any

from listing_agent.nodes.analyzer import get_llm
from listing_agent.state import AgentState, GeneratedListing


_PLATFORM_INSTRUCTIONS = {
    "shopify": (
        "Generate an HTML-formatted product description with proper headings and paragraphs. "
        "Include an SEO-optimized title (seo_title) and meta description (seo_description)."
    ),
    "amazon": (
        "Use a brand-first title format (e.g. 'BrandName - Product Description'). "
        "Write exactly 5 bullet points highlighting key features and benefits. "
        "Include backend_keywords (max 249 bytes, comma-separated, no brand/ASIN)."
    ),
    "etsy": (
        "Front-load the most important keywords in the title. "
        "Provide up to 13 tags, each max 20 characters."
    ),
}

_GENERATE_PROMPT_TEMPLATE = """You are an expert e-commerce listing copywriter.

Generate a product listing for the PLATFORM_PLACEHOLDER platform.

## Platform Rules
RULES_PLACEHOLDER

## Product Attributes
- Title: TITLE_PLACEHOLDER
- Category: CATEGORY_PLACEHOLDER
- Features: FEATURES_PLACEHOLDER
- Materials: MATERIALS_PLACEHOLDER
- Target Audience: AUDIENCE_PLACEHOLDER
- Price Range: PRICE_PLACEHOLDER
- Brand: BRAND_PLACEHOLDER
- Keywords: KEYWORDS_PLACEHOLDER

## Platform-Specific Instructions
INSTRUCTIONS_PLACEHOLDER

REFINEMENT_PLACEHOLDER

Return ONLY valid JSON with this exact structure:
{
  "platform": "PLATFORM_PLACEHOLDER",
  "title": "listing title",
  "description": "full product description",
  "bullet_points": ["point 1", "point 2"],
  "tags": ["tag1", "tag2"],
  "seo_title": "SEO title if applicable",
  "seo_description": "meta description if applicable",
  "backend_keywords": "backend keywords if applicable",
  "category_id": ""
}"""

_REFINEMENT_TEMPLATE = """## Previous Listing (needs improvement)
PREVIOUS_LISTING_PLACEHOLDER

## Feedback
FEEDBACK_PLACEHOLDER

Incorporate the feedback above to improve the listing."""


def generate_listings(state: AgentState) -> dict[str, Any]:
    """LangGraph node: generate platform-specific listings via LLM."""
    listings: list[GeneratedListing] = []
    errors: list[str] = []

    attrs = state.get("product_attributes")
    if attrs is None:
        return {"errors": ["No product_attributes in state"]}

    platform_rules = state.get("platform_rules", {})
    existing_listings = state.get("listings", [])
    iteration = state.get("refinement_count", 0)

    for platform in state["target_platforms"]:
        try:
            llm = get_llm()
            rules = platform_rules.get(platform, "No specific rules available.")  # type: ignore[union-attr]
            instructions = _PLATFORM_INSTRUCTIONS.get(platform, "")

            # Build refinement block if previous listing + feedback exist
            refinement_block = ""
            for prev in existing_listings:
                if prev.platform == platform and prev.feedback:
                    prev_json = prev.model_dump_json(indent=2)
                    refinement_block = (
                        _REFINEMENT_TEMPLATE
                        .replace("PREVIOUS_LISTING_PLACEHOLDER", prev_json)
                        .replace("FEEDBACK_PLACEHOLDER", prev.feedback)
                    )
                    break

            prompt = (
                _GENERATE_PROMPT_TEMPLATE
                .replace("PLATFORM_PLACEHOLDER", platform)
                .replace("RULES_PLACEHOLDER", str(rules))
                .replace("TITLE_PLACEHOLDER", attrs.title)
                .replace("CATEGORY_PLACEHOLDER", attrs.category)
                .replace("FEATURES_PLACEHOLDER", ", ".join(attrs.features))
                .replace("MATERIALS_PLACEHOLDER", ", ".join(attrs.materials) if attrs.materials else "N/A")
                .replace("AUDIENCE_PLACEHOLDER", attrs.target_audience or "General")
                .replace("PRICE_PLACEHOLDER", attrs.price_range or "Not specified")
                .replace("BRAND_PLACEHOLDER", attrs.brand or "Unbranded")
                .replace("KEYWORDS_PLACEHOLDER", ", ".join(attrs.keywords))
                .replace("INSTRUCTIONS_PLACEHOLDER", instructions)
                .replace("REFINEMENT_PLACEHOLDER", refinement_block)
            )

            response = llm.invoke(prompt)
            data = json.loads(response.content)
            data["platform"] = platform
            data["iteration"] = iteration
            listing = GeneratedListing(**data)
            listings.append(listing)
        except Exception as e:
            errors.append(f"Failed to generate listing for {platform}: {e}")

    result: dict[str, Any] = {}
    if listings:
        result["listings"] = listings
    if errors:
        result["errors"] = errors
    return result
