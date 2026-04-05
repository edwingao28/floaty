"""Amazon SP-API tools for listing management."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from listing_agent.config import get_config


def _get_listings_api():
    """Get SP-API Listings client. Lazy import to avoid hard dep in tests."""
    from sp_api.api import ListingsItems
    from sp_api.base import Marketplaces

    config = get_config()
    region_map = {"NA": Marketplaces.US, "EU": Marketplaces.UK, "FE": Marketplaces.JP}
    marketplace = region_map.get(config.AMAZON_REGION, Marketplaces.US)

    return ListingsItems(
        marketplace=marketplace,
        refresh_token=config.AMAZON_REFRESH_TOKEN,
        lwa_app_id=config.AMAZON_LWA_CLIENT_ID,
        lwa_client_secret=config.AMAZON_LWA_CLIENT_SECRET,
    )


def _build_attributes(
    title: str,
    bullet_points: list[str],
    description: str,
    backend_keywords: str,
    marketplace_id: str,
) -> dict[str, Any]:
    """Build SP-API attribute format."""
    attrs: dict[str, Any] = {
        "item_name": [{"value": title, "language_tag": "en_US", "marketplace_id": marketplace_id}],
        "product_description": [{"value": description, "language_tag": "en_US", "marketplace_id": marketplace_id}],
    }
    if bullet_points:
        attrs["bullet_point"] = [
            {"value": bp, "language_tag": "en_US", "marketplace_id": marketplace_id}
            for bp in bullet_points
        ]
    if backend_keywords:
        attrs["generic_keyword"] = [
            {"value": backend_keywords, "language_tag": "en_US", "marketplace_id": marketplace_id}
        ]
    return attrs


@tool
def amazon_put_listing(
    sku: str,
    title: str,
    bullet_points: list[str],
    description: str,
    backend_keywords: str = "",
) -> dict[str, Any]:
    """Create or replace an Amazon listing via putListingsItem."""
    if len(backend_keywords.encode("utf-8")) > 249:
        return {
            "status": "error",
            "error": f"backend_keywords exceeds 249 bytes ({len(backend_keywords.encode('utf-8'))} bytes). Entire field would be non-indexed.",
        }

    config = get_config()
    attrs = _build_attributes(title, bullet_points, description, backend_keywords, config.AMAZON_MARKETPLACE_ID)
    try:
        api = _get_listings_api()
        response = api.put_listings_item(
            sellerId=config.AMAZON_SELLER_ID,
            sku=sku,
            marketplaceIds=[config.AMAZON_MARKETPLACE_ID],
            body={"productType": "PRODUCT", "attributes": attrs},
        )
        return {"status": "success", "sku": sku, "response": str(response.payload)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def amazon_patch_listing(
    sku: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Partial update to an Amazon listing via patchListingsItem."""
    config = get_config()
    try:
        api = _get_listings_api()
        patches = [{"op": "REPLACE", "path": f"/attributes/{k}", "value": v} for k, v in updates.items()]
        response = api.patch_listings_item(
            sellerId=config.AMAZON_SELLER_ID,
            sku=sku,
            marketplaceIds=[config.AMAZON_MARKETPLACE_ID],
            body={"productType": "PRODUCT", "patches": patches},
        )
        return {"status": "success", "sku": sku, "response": str(response.payload)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
