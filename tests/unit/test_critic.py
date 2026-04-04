import pytest

from listing_agent.nodes.critic import critique_listings, should_refine, _score_listing
from listing_agent.state import AgentState, GeneratedListing


# --- fixtures ---

def good_shopify_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Handmade Ceramic Coffee Mug Minimalist Design Studio Pottery",
        description=(
            "This beautiful handmade ceramic mug is crafted with care in our studio. "
            "It holds 12 ounces and features a minimalist design perfect for coffee or tea lovers. "
            "Dishwasher safe and microwave safe. A great gift for any occasion."
        ),
        tags=["ceramic", "mug", "handmade", "minimalist", "coffee"],
        seo_title="Handmade Ceramic Mug | Minimalist Studio Pottery",
    )


def bad_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Mug",
        description="A mug.",
        tags=[],
    )


def good_amazon_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="amazon",
        title="Handmade Ceramic Coffee Mug Minimalist Design Dishwasher Safe",
        description=(
            "Premium handmade ceramic mug with 12oz capacity. "
            "Crafted with attention to detail in a small studio. "
            "Features minimalist design suitable for modern kitchens. "
            "Dishwasher and microwave safe. Perfect gift for coffee enthusiasts and home decor lovers."
        ),
        tags=["ceramic", "mug", "handmade", "coffee", "gift"],
        bullet_points=["12oz capacity", "Dishwasher safe", "Handmade in studio"],
    )


def good_etsy_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="etsy",
        title="Handmade Ceramic Coffee Mug Minimalist Design Studio Pottery",
        description=(
            "Beautiful handmade ceramic mug perfect for your morning coffee or afternoon tea. "
            "Crafted with love in our studio using high-quality clay materials. "
            "Minimalist design fits any kitchen aesthetic. Makes a wonderful gift."
        ),
        tags=["ceramic", "mug", "handmade", "minimalist", "coffee", "gift", "pottery"],
    )


# --- _score_listing tests ---

def test_good_listing_scores_above_threshold():
    score, feedback = _score_listing(good_shopify_listing())
    assert score >= 0.7, f"Expected >= 0.7, got {score}. Feedback: {feedback}"


def test_bad_listing_scores_below_threshold():
    score, feedback = _score_listing(bad_listing())
    assert score < 0.7, f"Expected < 0.7, got {score}. Feedback: {feedback}"
    assert "Title too short" in feedback
    assert "Needs more tags" in feedback


def test_score_capped_at_one():
    # A perfect listing should not exceed 1.0
    score, _ = _score_listing(good_shopify_listing())
    assert score <= 1.0


def test_amazon_platform_specific_bullet_points():
    listing_no_bullets = GeneratedListing(
        platform="amazon",
        title="Handmade Ceramic Coffee Mug Minimalist Design Dishwasher Safe",
        description=(
            "Premium handmade ceramic mug with 12oz capacity. "
            "Crafted with attention to detail in a small studio. "
            "Features minimalist design suitable for modern kitchens. "
            "Dishwasher and microwave safe. Perfect gift."
        ),
        tags=["ceramic", "mug", "handmade", "coffee", "gift"],
        bullet_points=[],
    )
    score_no_bullets, feedback_no_bullets = _score_listing(listing_no_bullets)
    score_with_bullets, _ = _score_listing(good_amazon_listing())
    assert score_with_bullets > score_no_bullets
    assert "bullet points" in feedback_no_bullets


def test_shopify_platform_specific_seo_title():
    listing_no_seo = GeneratedListing(
        platform="shopify",
        title="Handmade Ceramic Coffee Mug Minimalist Design Studio Pottery",
        description=(
            "This beautiful handmade ceramic mug is crafted with care in our studio. "
            "It holds 12 ounces and features a minimalist design perfect for coffee or tea lovers. "
            "Dishwasher safe and microwave safe. A great gift for any occasion."
        ),
        tags=["ceramic", "mug", "handmade", "minimalist", "coffee"],
        seo_title="",
    )
    score_no_seo, feedback = _score_listing(listing_no_seo)
    score_with_seo, _ = _score_listing(good_shopify_listing())
    assert score_with_seo > score_no_seo
    assert "seo_title" in feedback


def test_etsy_platform_specific_tags():
    listing_few_tags = GeneratedListing(
        platform="etsy",
        title="Handmade Ceramic Coffee Mug Minimalist Design Studio Pottery",
        description=(
            "Beautiful handmade ceramic mug perfect for your morning coffee or afternoon tea. "
            "Crafted with love in our studio using high-quality clay materials. "
            "Minimalist design fits any kitchen aesthetic. Makes a wonderful gift."
        ),
        tags=["ceramic", "mug", "handmade"],
    )
    score_few, feedback = _score_listing(listing_few_tags)
    score_many, _ = _score_listing(good_etsy_listing())
    assert score_many > score_few
    assert "etsy" in feedback.lower() or "tags" in feedback.lower()


# --- critique_listings node tests ---

def test_critique_listings_populates_score_and_feedback():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [good_shopify_listing(), bad_listing()],
        "refinement_count": 0,
    }
    result = critique_listings(state)
    assert "listings" in result
    assert "refinement_count" in result
    assert result["refinement_count"] == 1
    for listing in result["listings"]:
        assert listing.score is not None
        assert listing.feedback is not None


def test_critique_listings_increments_refinement_count():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [good_shopify_listing()],
        "refinement_count": 2,
    }
    result = critique_listings(state)
    assert result["refinement_count"] == 3


def test_critique_listings_empty_listings():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": [],
        "listings": [],
    }
    result = critique_listings(state)
    assert result["listings"] == []
    assert result["refinement_count"] == 1


# --- should_refine tests ---

def test_should_refine_done_when_all_scores_above_threshold():
    scored_listing = good_shopify_listing().model_copy(update={"score": 0.9})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "quality_threshold": 0.7,
        "max_refinements": 3,
    }
    assert should_refine(state) == "done"


def test_should_refine_refine_when_scores_below_threshold():
    scored_listing = bad_listing().model_copy(update={"score": 0.2})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "quality_threshold": 0.7,
        "max_refinements": 3,
    }
    assert should_refine(state) == "refine"


def test_should_refine_done_when_max_refinements_reached():
    # Even with low scores, stop when max reached
    scored_listing = bad_listing().model_copy(update={"score": 0.1})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 3,
        "quality_threshold": 0.7,
        "max_refinements": 3,
    }
    assert should_refine(state) == "done"


def test_should_refine_uses_state_defaults():
    # No explicit quality_threshold or max_refinements — use defaults
    scored_listing = bad_listing().model_copy(update={"score": 0.2})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
    }
    assert should_refine(state) == "refine"


def test_should_refine_done_when_listings_empty():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [],
        "refinement_count": 1,
    }
    assert should_refine(state) == "done"


def test_should_refine_done_when_refinement_count_exceeds_max():
    scored_listing = bad_listing().model_copy(update={"score": 0.1})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 5,
        "max_refinements": 3,
    }
    assert should_refine(state) == "done"
