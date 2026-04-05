"""Shopify Admin API tools using httpx + GraphQL."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool

from listing_agent.config import get_config

_CREATE_MUTATION = """
mutation productCreate($input: ProductInput!) {
  productCreate(input: $input) {
    product { id handle title }
    userErrors { field message }
  }
}
"""

_UPDATE_MUTATION = """
mutation productUpdate($input: ProductInput!) {
  productUpdate(input: $input) {
    product { id handle title }
    userErrors { field message }
  }
}
"""


def _graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    config = get_config()
    url = f"https://{config.SHOPIFY_SHOP_URL}/admin/api/{config.SHOPIFY_API_VERSION}/graphql.json"
    headers = {
        "X-Shopify-Access-Token": config.SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json",
    }
    with httpx.Client() as client:
        resp = client.post(url, json={"query": query, "variables": variables}, headers=headers)
        resp.raise_for_status()
        return resp.json()


@tool
def shopify_create_product(
    title: str,
    description_html: str,
    tags: list[str] | None = None,
    seo_title: str | None = None,
    seo_description: str | None = None,
    vendor: str | None = None,
) -> dict[str, Any]:
    """Create a product on Shopify via GraphQL Admin API."""
    product_input: dict[str, Any] = {
        "title": title,
        "descriptionHtml": description_html,
        "status": "DRAFT",
    }
    if tags:
        product_input["tags"] = tags
    if vendor:
        product_input["vendor"] = vendor
    if seo_title or seo_description:
        product_input["seo"] = {}
        if seo_title:
            product_input["seo"]["title"] = seo_title
        if seo_description:
            product_input["seo"]["description"] = seo_description

    data = _graphql(_CREATE_MUTATION, {"input": product_input})
    result = data.get("data", {}).get("productCreate", {})
    user_errors = result.get("userErrors", [])
    if user_errors:
        return {"status": "error", "errors": user_errors}
    product = result.get("product", {})
    return {
        "status": "success",
        "product_id": product.get("id"),
        "handle": product.get("handle"),
    }


@tool
def shopify_update_product(
    product_id: str,
    title: str | None = None,
    description_html: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Update an existing Shopify product."""
    product_input: dict[str, Any] = {"id": product_id}
    if title:
        product_input["title"] = title
    if description_html:
        product_input["descriptionHtml"] = description_html
    if tags is not None:
        product_input["tags"] = tags

    data = _graphql(_UPDATE_MUTATION, {"input": product_input})
    result = data.get("data", {}).get("productUpdate", {})
    user_errors = result.get("userErrors", [])
    if user_errors:
        return {"status": "error", "errors": user_errors}
    return {"status": "success", "product_id": product_id}
