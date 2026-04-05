from unittest.mock import MagicMock, patch

from listing_agent.tools.etsy import etsy_create_listing, etsy_update_listing


def test_create_listing_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"listing_id": 12345, "title": "Test Mug"}

    with patch("listing_agent.tools.etsy.get_config") as mock_config, \
         patch("listing_agent.tools.etsy.httpx") as mock_httpx:
        mock_config.return_value = MagicMock(
            ETSY_API_KEY="key",
            ETSY_SHOP_ID="shop1",
            ETSY_ACCESS_TOKEN="token",
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_httpx.Client.return_value = mock_client

        result = etsy_create_listing.invoke({
            "title": "Handmade Ceramic Mug",
            "description": "A beautiful mug.",
            "price": 28.0,
            "quantity": 10,
            "tags": ["ceramic", "mug", "handmade"],
            "taxonomy_id": 1234,
        })

    assert result["status"] == "success"
    assert result["listing_id"] == 12345


def test_create_listing_tag_validation():
    with patch("listing_agent.tools.etsy.get_config") as mock_config:
        mock_config.return_value = MagicMock(
            ETSY_API_KEY="key",
            ETSY_SHOP_ID="shop1",
            ETSY_ACCESS_TOKEN="token",
        )
        result = etsy_create_listing.invoke({
            "title": "Test",
            "description": "desc",
            "price": 10.0,
            "quantity": 1,
            "tags": ["a" * 21],  # exceeds 20 chars
            "taxonomy_id": 1234,
        })

    assert result["status"] == "error"
    assert "20 chars" in result["error"]


def test_create_listing_too_many_tags():
    with patch("listing_agent.tools.etsy.get_config") as mock_config:
        mock_config.return_value = MagicMock(
            ETSY_API_KEY="key",
            ETSY_SHOP_ID="shop1",
            ETSY_ACCESS_TOKEN="token",
        )
        result = etsy_create_listing.invoke({
            "title": "Test",
            "description": "desc",
            "price": 10.0,
            "quantity": 1,
            "tags": [f"tag{i}" for i in range(14)],  # 14 > 13 max
            "taxonomy_id": 1234,
        })

    assert result["status"] == "error"
    assert "13 tags" in result["error"]
