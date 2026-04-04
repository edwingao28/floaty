import operator
from typing import Annotated, get_type_hints

import pytest
from pydantic import ValidationError

from listing_agent.state import (
    AgentState,
    GeneratedListing,
    PlatformRules,
    ProductAttributes,
)


# --- ProductAttributes ---


def test_product_attributes_valid():
    pa = ProductAttributes(
        title="Organic Cotton T-Shirt",
        category="Apparel",
        features=["breathable", "eco-friendly"],
        materials=["organic cotton"],
        target_audience="eco-conscious millennials",
        price_range="mid",
        brand="EcoWear",
        keywords=["organic", "cotton", "t-shirt"],
        raw_input="Organic cotton t-shirt for eco-conscious buyers",
    )
    assert pa.title == "Organic Cotton T-Shirt"
    assert pa.category == "Apparel"
    assert pa.features == ["breathable", "eco-friendly"]
    assert pa.materials == ["organic cotton"]
    assert pa.target_audience == "eco-conscious millennials"
    assert pa.price_range == "mid"
    assert pa.brand == "EcoWear"
    assert pa.keywords == ["organic", "cotton", "t-shirt"]
    assert pa.raw_input == "Organic cotton t-shirt for eco-conscious buyers"


def test_product_attributes_defaults():
    pa = ProductAttributes(
        title="Widget",
        category="Gadgets",
        features=["fast"],
        keywords=["widget"],
        raw_input="A fast widget",
    )
    assert pa.materials == []
    assert pa.target_audience == ""
    assert pa.price_range == ""
    assert pa.brand == ""


def test_product_attributes_rejects_missing():
    with pytest.raises(ValidationError):
        ProductAttributes(title="Widget", category="Gadgets")  # missing features, keywords, raw_input


# --- PlatformRules ---


def test_platform_rules_valid():
    pr = PlatformRules(
        platform="amazon",
        title_constraints={"max_length": 200},
        description_constraints={"format": "HTML"},
        keyword_rules={"density": 0.02},
        category_taxonomy={"browse_node": "12345"},
        additional_rules=["Use A+ content"],
    )
    assert pr.platform == "amazon"
    assert pr.title_constraints == {"max_length": 200}
    assert pr.description_constraints == {"format": "HTML"}
    assert pr.keyword_rules == {"density": 0.02}
    assert pr.category_taxonomy == {"browse_node": "12345"}
    assert pr.additional_rules == ["Use A+ content"]


# --- GeneratedListing ---


def test_generated_listing_valid():
    gl = GeneratedListing(
        platform="shopify",
        title="Premium Widget",
        description="A premium widget for professionals.",
        bullet_points=["durable", "lightweight"],
        tags=["widget", "premium"],
        seo_title="Buy Premium Widget",
        seo_description="Shop the best premium widget.",
        backend_keywords="widget premium buy",
        category_id="cat-123",
        score=0.92,
        feedback="Excellent listing",
        iteration=2,
    )
    assert gl.platform == "shopify"
    assert gl.title == "Premium Widget"
    assert gl.score == 0.92
    assert gl.feedback == "Excellent listing"
    assert gl.iteration == 2


def test_generated_listing_optional_score_feedback():
    gl = GeneratedListing(
        platform="etsy",
        title="Handmade Mug",
        description="A beautiful handmade mug.",
    )
    assert gl.score is None
    assert gl.feedback is None


def test_generated_listing_defaults():
    gl = GeneratedListing(
        platform="amazon",
        title="Gadget",
        description="A useful gadget.",
    )
    assert gl.bullet_points == []
    assert gl.tags == []
    assert gl.seo_title == ""
    assert gl.seo_description == ""
    assert gl.backend_keywords == ""
    assert gl.category_id == ""
    assert gl.iteration == 0


# --- AgentState ---


def test_agent_state_structure():
    state: AgentState = {
        "raw_product_data": {"name": "Widget"},
        "product_attributes": None,
        "target_platforms": ["shopify", "amazon", "etsy"],
        "platform_rules": [],
        "listings": [],
        "approved_listings": [],
        "refinement_count": 0,
        "max_refinements": 3,
        "quality_threshold": 0.8,
        "publish_results": {},
        "errors": [],
    }
    assert state["raw_product_data"] == {"name": "Widget"}
    assert state["product_attributes"] is None
    assert state["target_platforms"] == ["shopify", "amazon", "etsy"]
    assert state["listings"] == []
    assert state["errors"] == []
    assert state["max_refinements"] == 3
    assert state["quality_threshold"] == 0.8


def test_agent_state_listings_reducer():
    hints = get_type_hints(AgentState, include_extras=True)
    metadata = hints["listings"].__metadata__
    reducer = metadata[0]
    assert reducer is operator.add

    list_a = [
        GeneratedListing(platform="shopify", title="A", description="desc A"),
    ]
    list_b = [
        GeneratedListing(platform="amazon", title="B", description="desc B"),
    ]
    merged = reducer(list_a, list_b)
    assert len(merged) == 2
    assert merged[0].platform == "shopify"
    assert merged[1].platform == "amazon"


def test_agent_state_errors_reducer():
    hints = get_type_hints(AgentState, include_extras=True)
    metadata = hints["errors"].__metadata__
    reducer = metadata[0]
    assert reducer is operator.add

    merged = reducer(["error 1"], ["error 2", "error 3"])
    assert merged == ["error 1", "error 2", "error 3"]


# --- Rejection tests ---


def test_platform_rules_rejects_missing_platform():
    """PlatformRules without platform raises ValidationError."""
    with pytest.raises(ValidationError):
        PlatformRules()


def test_generated_listing_rejects_missing_required():
    """GeneratedListing without title and description raises ValidationError."""
    with pytest.raises(ValidationError):
        GeneratedListing(platform="shopify")


def test_generated_listing_rejects_invalid_platform():
    """GeneratedListing with invalid platform raises ValidationError."""
    with pytest.raises(ValidationError):
        GeneratedListing(platform="ebay", title="x", description="x")


def test_generated_listing_rejects_score_out_of_range():
    """score must be 0.0-1.0."""
    with pytest.raises(ValidationError):
        GeneratedListing(platform="shopify", title="x", description="x", score=2.0)
    with pytest.raises(ValidationError):
        GeneratedListing(platform="shopify", title="x", description="x", score=-0.5)
