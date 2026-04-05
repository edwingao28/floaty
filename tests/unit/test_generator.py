import json

import pytest
from unittest.mock import MagicMock, patch

from listing_agent.nodes.generator import generate_listings
from listing_agent.state import AgentState, GeneratedListing, ProductAttributes


@pytest.fixture
def product_attrs() -> ProductAttributes:
    return ProductAttributes(
        title="Handmade Ceramic Mug",
        category="home_and_kitchen",
        features=["12oz capacity", "dishwasher safe", "minimalist design"],
        materials=["ceramic"],
        target_audience="coffee lovers",
        price_range="mid-range ($20-$45)",
        brand="StudioCraft",
        keywords=["ceramic mug", "handmade", "minimalist"],
        raw_input="handmade ceramic mug",
    )


@pytest.fixture
def base_state(product_attrs) -> AgentState:
    return {
        "raw_product_data": {"description": "handmade ceramic mug"},
        "target_platforms": ["shopify"],
        "product_attributes": product_attrs,
        "platform_rules": {"shopify": "Use SEO-friendly titles. HTML descriptions allowed."},
    }


def _mock_llm_json(platform: str = "shopify") -> str:
    return json.dumps({
        "platform": platform,
        "title": "Handmade Ceramic Mug - 12oz Minimalist Design",
        "description": "<p>Beautiful handmade ceramic mug.</p>",
        "bullet_points": ["12oz capacity", "Dishwasher safe"],
        "tags": ["ceramic", "handmade"],
        "seo_title": "Handmade Ceramic Mug | StudioCraft",
        "seo_description": "Shop our handmade ceramic mug.",
        "backend_keywords": "",
        "category_id": "",
    })


def test_generate_listings_happy_path(base_state):
    with patch("listing_agent.nodes.generator.invoke_with_fallback", return_value=_mock_llm_json("shopify")):
        result = generate_listings(base_state)

    assert "listings" in result
    assert len(result["listings"]) == 1
    listing = result["listings"][0]
    assert isinstance(listing, GeneratedListing)
    assert listing.platform == "shopify"
    assert listing.title == "Handmade Ceramic Mug - 12oz Minimalist Design"
    assert "errors" not in result


def test_generate_listings_malformed_json(base_state):
    with patch("listing_agent.nodes.generator.invoke_with_fallback", return_value="this is not valid json {{{}"):
        result = generate_listings(base_state)

    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Failed to generate listing for shopify" in result["errors"][0]
    assert "listings" not in result


def test_generate_listings_refinement_mode(base_state, product_attrs):
    previous_listing = GeneratedListing(
        platform="shopify",
        title="Old Title",
        description="Old description",
        feedback="Make the title more descriptive and add more keywords",
        iteration=0,
    )
    base_state["listings"] = [previous_listing]

    captured_prompts = []

    def capture_prompt(prompt: str) -> str:
        captured_prompts.append(prompt)
        return _mock_llm_json("shopify")

    with patch("listing_agent.nodes.generator.invoke_with_fallback", side_effect=capture_prompt):
        result = generate_listings(base_state)

    assert captured_prompts
    call_prompt = captured_prompts[0]
    assert "Make the title more descriptive" in call_prompt
    assert "Old Title" in call_prompt
    assert "Previous Listing" in call_prompt

    assert "listings" in result
    assert isinstance(result["listings"][0], GeneratedListing)


def test_generate_listings_no_attributes():
    state: AgentState = {
        "raw_product_data": {"description": "test"},
        "target_platforms": ["shopify"],
    }
    result = generate_listings(state)
    assert "errors" in result
    assert "No product_attributes" in result["errors"][0]


def test_generate_listings_multiple_platforms(base_state):
    base_state["target_platforms"] = ["shopify", "amazon", "etsy"]
    base_state["platform_rules"]["amazon"] = "Brand-first titles."
    base_state["platform_rules"]["etsy"] = "Keyword-rich tags."

    call_count = 0
    platforms = ["shopify", "amazon", "etsy"]

    def make_response(prompt: str) -> str:
        nonlocal call_count
        platform = platforms[call_count]
        call_count += 1
        return _mock_llm_json(platform)

    with patch("listing_agent.nodes.generator.invoke_with_fallback", side_effect=make_response):
        result = generate_listings(base_state)

    assert len(result["listings"]) == 3
    result_platforms = {l.platform for l in result["listings"]}
    assert result_platforms == {"shopify", "amazon", "etsy"}


def test_generate_listings_uses_fallback_on_primary_failure(base_state):
    """Falls back to secondary model when primary raises."""
    fallback_content = _mock_llm_json("shopify")
    mock_cfg = MagicMock()
    mock_cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
    mock_cfg.ANTHROPIC_FALLBACK_MODEL = "claude-haiku-4-5-20251001"
    with patch("listing_agent.nodes._llm.get_config", return_value=mock_cfg), \
         patch("listing_agent.nodes._llm.ChatAnthropic") as MockLLM:
        primary_instance = MagicMock()
        primary_instance.invoke.side_effect = Exception("Rate limit")
        fallback_instance = MagicMock()
        fallback_instance.invoke.return_value = MagicMock(content=fallback_content)
        MockLLM.side_effect = [primary_instance, fallback_instance]

        result = generate_listings(base_state)

    assert "listings" in result
    assert result["listings"][0].platform == "shopify"
