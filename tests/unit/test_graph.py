"""Unit tests for the LangGraph graph wiring (graph.py)."""
from unittest.mock import MagicMock, patch

import pytest

from listing_agent.graph import build_graph

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_ANALYZER_JSON = (
    '{"title": "Handmade Ceramic Mug", "category": "home_and_kitchen",'
    ' "features": ["12oz", "handmade"], "materials": ["ceramic"],'
    ' "target_audience": "coffee lovers", "price_range": "mid-range",'
    ' "brand": "", "keywords": ["ceramic", "mug"]}'
)

# Long enough title (>=50) + long description (>=200 chars, >=30 words) + >=5 tags
# + seo_title present → score >= 0.7, loop exits on first pass.
_GENERATOR_JSON = (
    '{"title": "Handmade Ceramic Coffee Mug 12oz | Minimalist Design",'
    ' "description": "A beautifully handcrafted ceramic mug perfect for coffee lovers.'
    " 12oz capacity, dishwasher safe. Made with care in our artisan studio."
    ' The minimalist design complements any kitchen decor perfectly.",'
    ' "tags": ["ceramic mug", "handmade", "coffee cup", "minimalist", "dishwasher safe"],'
    ' "bullet_points": ["12oz capacity", "Dishwasher safe", "Handmade craftsmanship"],'
    ' "seo_title": "Handmade Ceramic Mug | StudioCraft",'
    ' "seo_description": "Shop our handmade ceramic mug."}'
)

# Short listing → scores below 0.7 → loop keeps refining.
_GENERATOR_JSON_SHORT = (
    '{"title": "Mug",'
    ' "description": "A mug.",'
    ' "tags": [],'
    ' "bullet_points": [],'
    ' "seo_title": "",'
    ' "seo_description": ""}'
)

_INITIAL_STATE = {
    "raw_product_data": {"description": "handmade ceramic mug"},
    "target_platforms": ["shopify"],
}


def _make_llm_mock(json_str: str) -> MagicMock:
    """Return a mock that behaves like ChatAnthropic.invoke()."""
    llm = MagicMock()
    response = MagicMock()
    response.content = json_str
    llm.invoke.return_value = response
    return llm


def _make_retriever_mock() -> MagicMock:
    retriever = MagicMock()
    retriever.get_rules.return_value = "No specific rules."
    return retriever


# ---------------------------------------------------------------------------
# Test 1: Happy-path end-to-end
# ---------------------------------------------------------------------------


def test_graph_end_to_end_happy_path():
    """Graph runs to completion with valid mocks; final state has scored listings."""
    analyzer_llm = _make_llm_mock(_ANALYZER_JSON)
    generator_llm = _make_llm_mock(_GENERATOR_JSON)
    retriever_mock = _make_retriever_mock()

    with (
        patch("listing_agent.nodes.analyzer.get_llm", return_value=analyzer_llm),
        patch("listing_agent.nodes.generator.get_llm", return_value=generator_llm),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
    ):
        graph = build_graph()
        result = graph.invoke(_INITIAL_STATE)

    assert "listings" in result
    listings = result["listings"]
    assert len(listings) == 1
    listing = listings[0]
    assert listing.platform == "shopify"
    assert listing.score is not None
    assert listing.score >= 0.7, f"Expected score >= 0.7, got {listing.score}"
    # Loop exited cleanly — errors should be absent or empty
    assert result.get("errors", []) == []


# ---------------------------------------------------------------------------
# Test 2: Refinement loop reaches max_refinements
# ---------------------------------------------------------------------------


def test_graph_refinement_loop_reaches_max():
    """Short listing scores below 0.7; loop runs until max_refinements=2."""
    analyzer_llm = _make_llm_mock(_ANALYZER_JSON)
    generator_llm = _make_llm_mock(_GENERATOR_JSON_SHORT)
    retriever_mock = _make_retriever_mock()

    initial_state = {**_INITIAL_STATE, "max_refinements": 2}

    with (
        patch("listing_agent.nodes.analyzer.get_llm", return_value=analyzer_llm),
        patch("listing_agent.nodes.generator.get_llm", return_value=generator_llm),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
    ):
        graph = build_graph()
        result = graph.invoke(initial_state)

    assert result.get("refinement_count", 0) == 2


# ---------------------------------------------------------------------------
# Test 3: Error propagation from analyzer
# ---------------------------------------------------------------------------


def test_graph_error_propagation_invalid_analyzer_json():
    """Analyzer returns invalid JSON → errors list is populated, graph still terminates."""
    bad_llm = _make_llm_mock("not valid json {{{")
    retriever_mock = _make_retriever_mock()

    # Generator is never reached (product_attributes will be None), but we
    # still need a mock in place to avoid real LLM calls if something leaks.
    generator_llm = _make_llm_mock(_GENERATOR_JSON)

    with (
        patch("listing_agent.nodes.analyzer.get_llm", return_value=bad_llm),
        patch("listing_agent.nodes.generator.get_llm", return_value=generator_llm),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
    ):
        graph = build_graph()
        result = graph.invoke(_INITIAL_STATE)

    errors = result.get("errors", [])
    assert len(errors) >= 1
    assert any("Failed to" in e for e in errors)
