from unittest.mock import MagicMock, patch

from listing_agent.nodes.publisher import publish_listings
from listing_agent.state import AgentState, GeneratedListing


def _approved_listing(platform: str = "shopify", price: float | None = None) -> GeneratedListing:
    return GeneratedListing(
        platform=platform,
        title="Widget",
        description="<p>A widget</p>",
        tags=["widget"],
        seo_title="Widget | Brand",
        seo_description="Buy a widget.",
        score=0.9,
        price=price,
    )


def test_publish_shopify():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "approved_listings": [_approved_listing("shopify")],
    }
    with patch("listing_agent.nodes.publisher.shopify_create_product") as mock_tool:
        mock_tool.invoke.return_value = {"status": "success", "product_id": "123"}
        result = publish_listings(state)

    assert result["publish_results"]["shopify"]["status"] == "success"


def test_publish_partial_failure():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify", "amazon"],
        "approved_listings": [
            _approved_listing("shopify"),
            _approved_listing("amazon"),
        ],
    }
    with patch("listing_agent.nodes.publisher.shopify_create_product") as mock_shopify, \
         patch("listing_agent.nodes.publisher.amazon_put_listing") as mock_amazon:
        mock_shopify.invoke.return_value = {"status": "success", "product_id": "123"}
        mock_amazon.invoke.side_effect = Exception("API down")
        result = publish_listings(state)

    assert result["publish_results"]["shopify"]["status"] == "success"
    assert result["publish_results"]["amazon"]["status"] == "error"


def test_publish_empty_approved():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "approved_listings": [],
    }
    result = publish_listings(state)
    assert result["publish_results"] == {}


def test_publish_etsy_with_price():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["etsy"],
        "approved_listings": [_approved_listing("etsy", price=28.0)],
    }
    with patch("listing_agent.nodes.publisher.etsy_create_listing") as mock_tool:
        mock_tool.invoke.return_value = {"status": "success", "listing_id": "456"}
        result = publish_listings(state)

    assert result["publish_results"]["etsy"]["status"] == "success"


def test_publish_etsy_requires_price():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["etsy"],
        "approved_listings": [_approved_listing("etsy")],  # no price set
    }
    result = publish_listings(state)
    assert result["publish_results"]["etsy"]["status"] == "error"
    assert "price" in result["publish_results"]["etsy"]["error"]
