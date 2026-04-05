"""Publisher node: publish approved listings to platform APIs."""

from __future__ import annotations

from typing import Any

from listing_agent.state import AgentState, GeneratedListing
from listing_agent.tools.shopify import shopify_create_product
from listing_agent.tools.amazon import amazon_put_listing
from listing_agent.tools.etsy import etsy_create_listing


def _publish_shopify(listing: GeneratedListing) -> dict[str, Any]:
    return shopify_create_product.invoke({
        "title": listing.title,
        "description_html": listing.description,
        "tags": listing.tags,
        "seo_title": listing.seo_title or None,
        "seo_description": listing.seo_description or None,
    })


def _publish_amazon(listing: GeneratedListing) -> dict[str, Any]:
    sku = f"listing-agent-{hash(listing.title) % 10**8:08d}"
    return amazon_put_listing.invoke({
        "sku": sku,
        "title": listing.title,
        "bullet_points": listing.bullet_points,
        "description": listing.description,
        "backend_keywords": listing.backend_keywords,
    })


def _publish_etsy(listing: GeneratedListing) -> dict[str, Any]:
    return etsy_create_listing.invoke({
        "title": listing.title,
        "description": listing.description,
        "price": 0.0,  # price must come from product data
        "quantity": 1,
        "tags": listing.tags,
    })


_PUBLISHERS = {
    "shopify": _publish_shopify,
    "amazon": _publish_amazon,
    "etsy": _publish_etsy,
}


def publish_listings(state: AgentState) -> dict[str, Any]:
    """LangGraph node: publish each approved listing to its platform."""
    approved: list[GeneratedListing] = state.get("approved_listings", [])
    results: dict[str, Any] = {}

    for listing in approved:
        publisher = _PUBLISHERS.get(listing.platform)
        if not publisher:
            results[listing.platform] = {"status": "error", "error": f"No publisher for {listing.platform}"}
            continue
        try:
            results[listing.platform] = publisher(listing)
        except Exception as e:
            results[listing.platform] = {"status": "error", "error": str(e)}

    return {"publish_results": results}
