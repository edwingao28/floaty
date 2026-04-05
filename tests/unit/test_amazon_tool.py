from unittest.mock import MagicMock, patch

from listing_agent.tools.amazon import amazon_put_listing, amazon_patch_listing


def test_put_listing_success():
    mock_listings_api = MagicMock()
    mock_listings_api.put_listings_item.return_value = MagicMock(
        payload={"status": "ACCEPTED", "sku": "test-sku-001"}
    )
    with patch("listing_agent.tools.amazon._get_listings_api", return_value=mock_listings_api), \
         patch("listing_agent.tools.amazon.get_config") as mock_config:
        mock_config.return_value = MagicMock(
            AMAZON_SELLER_ID="SELLER1",
            AMAZON_MARKETPLACE_ID="ATVPDKIKX0DER",
        )
        result = amazon_put_listing.invoke({
            "sku": "test-sku-001",
            "title": "Test Product",
            "bullet_points": ["b1", "b2", "b3", "b4", "b5"],
            "description": "A test product.",
            "backend_keywords": "test product widget",
        })

    assert result["status"] == "success"


def test_put_listing_keywords_over_249_bytes():
    with patch("listing_agent.tools.amazon.get_config") as mock_config:
        mock_config.return_value = MagicMock(
            AMAZON_SELLER_ID="SELLER1",
            AMAZON_MARKETPLACE_ID="ATVPDKIKX0DER",
        )
        result = amazon_put_listing.invoke({
            "sku": "test-sku",
            "title": "Test",
            "bullet_points": [],
            "description": "desc",
            "backend_keywords": "a " * 130,  # > 249 bytes
        })

    assert result["status"] == "error"
    assert "249 bytes" in result["error"]


def test_patch_listing_success():
    mock_listings_api = MagicMock()
    mock_listings_api.patch_listings_item.return_value = MagicMock(
        payload={"status": "ACCEPTED"}
    )
    with patch("listing_agent.tools.amazon._get_listings_api", return_value=mock_listings_api), \
         patch("listing_agent.tools.amazon.get_config") as mock_config:
        mock_config.return_value = MagicMock(
            AMAZON_SELLER_ID="SELLER1",
            AMAZON_MARKETPLACE_ID="ATVPDKIKX0DER",
        )
        result = amazon_patch_listing.invoke({
            "sku": "test-sku-001",
            "updates": {"item_name": [{"value": "New Title"}]},
        })

    assert result["status"] == "success"
