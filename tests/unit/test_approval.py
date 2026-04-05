from unittest.mock import patch

from listing_agent.nodes.approval import approve_listings
from listing_agent.state import AgentState, GeneratedListing


def _scored_listing(platform: str = "shopify", score: float = 0.85) -> GeneratedListing:
    return GeneratedListing(
        platform=platform,
        title=f"{platform.capitalize()} Widget Title",
        description="A great widget.",
        score=score,
        feedback="Looks good.",
    )


def test_approve_listings_calls_interrupt():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [_scored_listing()],
    }
    with patch("listing_agent.nodes.approval.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"decision": "approve_all"}
        result = approve_listings(state)

    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert "listings" in payload
    assert payload["listings"][0]["platform"] == "shopify"
    assert result["approved_listings"] == state["listings"]


def test_approve_selective():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify", "amazon"],
        "listings": [_scored_listing("shopify"), _scored_listing("amazon")],
    }
    with patch("listing_agent.nodes.approval.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {
            "decision": "approve_selective",
            "platforms": ["shopify"],
        }
        result = approve_listings(state)

    assert len(result["approved_listings"]) == 1
    assert result["approved_listings"][0].platform == "shopify"


def test_reject_all():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [_scored_listing()],
    }
    with patch("listing_agent.nodes.approval.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"decision": "reject_all"}
        result = approve_listings(state)

    assert result["approved_listings"] == []
