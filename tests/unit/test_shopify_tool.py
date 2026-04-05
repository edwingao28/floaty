import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from listing_agent.tools.shopify import shopify_create_product, shopify_update_product


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.SHOPIFY_SHOP_URL = "test-shop.myshopify.com"
    config.SHOPIFY_ACCESS_TOKEN = "shpat_test"
    config.SHOPIFY_API_VERSION = "2026-01"
    return config


def _mock_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    return resp


def test_create_product_success(mock_config):
    response_data = {
        "data": {
            "productCreate": {
                "product": {"id": "gid://shopify/Product/123", "handle": "test-mug"},
                "userErrors": [],
            }
        },
        "extensions": {"cost": {"throttleStatus": {"currentlyAvailable": 900}}},
    }
    with patch("listing_agent.tools.shopify.get_config", return_value=mock_config), \
         patch("listing_agent.tools.shopify.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(response_data)
        mock_httpx.Client.return_value = mock_client

        result = shopify_create_product.invoke({
            "title": "Test Mug",
            "description_html": "<p>A mug</p>",
            "tags": ["mug"],
        })

    assert result["status"] == "success"
    assert result["product_id"] == "gid://shopify/Product/123"


def test_create_product_user_errors(mock_config):
    response_data = {
        "data": {
            "productCreate": {
                "product": None,
                "userErrors": [{"field": ["handle"], "message": "Handle already exists"}],
            }
        },
        "extensions": {"cost": {"throttleStatus": {"currentlyAvailable": 900}}},
    }
    with patch("listing_agent.tools.shopify.get_config", return_value=mock_config), \
         patch("listing_agent.tools.shopify.httpx") as mock_httpx:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(response_data)
        mock_httpx.Client.return_value = mock_client

        result = shopify_create_product.invoke({
            "title": "Test Mug",
            "description_html": "<p>A mug</p>",
        })

    assert result["status"] == "error"
    assert "Handle already exists" in str(result["errors"])
