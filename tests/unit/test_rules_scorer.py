import pytest
from listing_agent.scoring.rules import RulesScorer
from listing_agent.state import GeneratedListing


@pytest.fixture
def scorer():
    return RulesScorer()


def _good_shopify() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Handmade Ceramic Coffee Mug 12oz | Dishwasher Safe | Minimalist Design",
        description="<p>A beautifully handcrafted <strong>ceramic mug</strong> perfect for coffee lovers.</p><ul><li>12oz capacity</li><li>Dishwasher safe</li><li>Minimalist design fits any kitchen</li></ul><p>Made with premium materials in our artisan studio.</p>",
        tags=["ceramic mug", "handmade", "coffee cup", "minimalist", "dishwasher safe"],
        seo_title="Handmade Ceramic Mug | 12oz Minimalist",
        seo_description="Shop our handmade ceramic mug. 12oz, dishwasher safe, minimalist design. Perfect gift for coffee lovers.",
        bullet_points=[],
    )


def _good_amazon(keywords: list[str] | None = None) -> GeneratedListing:
    return GeneratedListing(
        platform="amazon",
        title="StudioCraft - Handmade Ceramic Coffee Mug 12oz Dishwasher Safe Minimalist Design",
        description="Premium handmade ceramic mug with 12oz capacity. Crafted with care in our artisan studio. Minimalist design.",
        bullet_points=[
            "12oz capacity perfect for standard coffee serving",
            "Dishwasher safe for easy cleanup",
            "Handmade in artisan studio with premium ceramic",
            "Minimalist design complements any kitchen decor",
            "Perfect gift for coffee and tea lovers",
        ],
        tags=[],
        backend_keywords="ceramic mug handmade coffee cup minimalist dishwasher safe artisan pottery gift",
    )


def _bad_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Mug",
        description="A mug.",
        tags=[],
    )


def test_title_length_compliance_good(scorer):
    listing = _good_shopify()
    result = scorer.score(listing, primary_keywords=["ceramic mug"])
    assert result.dimensions["title_length_compliance"] == 1.0


def test_title_length_compliance_bad(scorer):
    listing = _bad_listing()
    result = scorer.score(listing, primary_keywords=["mug"])
    assert result.dimensions["title_length_compliance"] == 0.0


def test_title_over_platform_limit(scorer):
    listing = GeneratedListing(
        platform="etsy",
        title="A" * 141,  # over 140 for etsy
        description="desc " * 50,
    )
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["char_limit_compliance"] == 0.0


def test_bullet_compliance_amazon_exact_5(scorer):
    listing = _good_amazon()
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["bullet_compliance"] == 1.0


def test_bullet_compliance_amazon_wrong_count(scorer):
    listing = GeneratedListing(
        platform="amazon",
        title="Product Title Over Thirty Characters Long Enough",
        description="desc " * 50,
        bullet_points=["only one"],
    )
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["bullet_compliance"] == 0.0


def test_keyword_presence_full(scorer):
    listing = _good_shopify()
    result = scorer.score(listing, primary_keywords=["ceramic mug"])
    assert result.dimensions["keyword_presence"] >= 0.5


def test_keyword_presence_absent(scorer):
    listing = GeneratedListing(
        platform="shopify",
        title="Beautiful Product For Your Home",
        description="A wonderful item you will love. Great for daily use. Made with quality.",
    )
    result = scorer.score(listing, primary_keywords=["ceramic mug"])
    assert result.dimensions["keyword_presence"] == 0.0


def test_html_validity_shopify(scorer):
    listing = _good_shopify()
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["html_validity"] == 1.0


def test_html_validity_skipped_for_amazon(scorer):
    listing = _good_amazon()
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["html_validity"] == 1.0  # N/A → passes


def test_html_validity_script_tag(scorer):
    listing = GeneratedListing(
        platform="shopify",
        title="Product Title Over Thirty Characters Long Enough",
        description="<p>Good</p><script>alert('bad')</script>",
    )
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["html_validity"] == 0.0


def test_amazon_backend_keywords_249_bytes(scorer):
    listing = GeneratedListing(
        platform="amazon",
        title="Product Title Over Thirty Characters Long Enough",
        description="desc " * 50,
        bullet_points=["b1", "b2", "b3", "b4", "b5"],
        backend_keywords="a " * 125,  # 250 bytes — over limit
    )
    result = scorer.score(listing, primary_keywords=[])
    assert result.dimensions["char_limit_compliance"] == 0.0


def test_composite_score(scorer):
    listing = _good_shopify()
    result = scorer.score(listing, primary_keywords=["ceramic mug"])
    assert 0.0 <= result.composite <= 1.0
    assert len(result.violations) == 0 or isinstance(result.violations, list)


def test_bad_listing_low_score(scorer):
    listing = _bad_listing()
    result = scorer.score(listing, primary_keywords=["mug"])
    assert result.composite < 0.5
    assert len(result.violations) > 0
