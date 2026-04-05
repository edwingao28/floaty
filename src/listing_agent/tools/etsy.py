"""Etsy Open API v3 tools for listing management."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool

from listing_agent.config import get_config

_BASE_URL = "https://openapi.etsy.com/v3"
_ETSY_TAG_MAX = 13
_ETSY_TAG_CHAR_MAX = 20


def _headers() -> dict[str, str]:
    config = get_config()
    return {
        "x-api-key": config.ETSY_API_KEY,
        "Authorization": f"Bearer {config.ETSY_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def _validate_tags(tags: list[str]) -> str | None:
    """Return error message if tags invalid, None if OK."""
    if len(tags) > _ETSY_TAG_MAX:
        return f"Etsy max {_ETSY_TAG_MAX} tags (got {len(tags)})"
    for tag in tags:
        if len(tag) > _ETSY_TAG_CHAR_MAX:
            return f"Etsy tag '{tag}' exceeds {_ETSY_TAG_CHAR_MAX} chars"
    return None


@tool
def etsy_create_listing(
    title: str,
    description: str,
    price: float,
    quantity: int,
    tags: list[str] | None = None,
    taxonomy_id: int | None = None,
    who_made: str = "i_did",
    when_made: str = "made_to_order",
) -> dict[str, Any]:
    """Create a listing on Etsy."""
    if tags:
        err = _validate_tags(tags)
        if err:
            return {"status": "error", "error": err}

    config = get_config()
    body: dict[str, Any] = {
        "title": title,
        "description": description,
        "price": price,
        "quantity": quantity,
        "who_made": who_made,
        "when_made": when_made,
    }
    if tags:
        body["tags"] = tags
    if taxonomy_id:
        body["taxonomy_id"] = taxonomy_id

    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{_BASE_URL}/application/shops/{config.ETSY_SHOP_ID}/listings",
                json=body,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return {"status": "success", "listing_id": data.get("listing_id")}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def etsy_update_listing(
    listing_id: int,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    price: float | None = None,
) -> dict[str, Any]:
    """Update an existing Etsy listing."""
    if tags:
        err = _validate_tags(tags)
        if err:
            return {"status": "error", "error": err}

    config = get_config()
    body: dict[str, Any] = {}
    if title:
        body["title"] = title
    if description:
        body["description"] = description
    if tags is not None:
        body["tags"] = tags
    if price is not None:
        body["price"] = price

    try:
        with httpx.Client() as client:
            resp = client.patch(
                f"{_BASE_URL}/application/shops/{config.ETSY_SHOP_ID}/listings/{listing_id}",
                json=body,
                headers=_headers(),
            )
            resp.raise_for_status()
            return {"status": "success", "listing_id": listing_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}
