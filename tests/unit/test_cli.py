from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from listing_agent.cli import app
from listing_agent.state import GeneratedListing

runner = CliRunner()


def test_help_smoke():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "listing-agent" in result.output


def _make_listing(platform: str, score: float = 0.8) -> GeneratedListing:
    return GeneratedListing(
        platform=platform,
        title=f"{platform.capitalize()} Widget",
        description="A great widget for everyone.",
        tags=["widget", "great"],
        score=score,
    )


def _mock_graph(listings, errors=None, refinement_count=1):
    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = {
        "listings": listings,
        "errors": errors or [],
        "refinement_count": refinement_count,
    }
    mock_graph = MagicMock(return_value=mock_compiled)
    return mock_graph


def test_generate_rich_output():
    listings = [_make_listing("shopify", score=0.85)]
    mock_build = _mock_graph(listings, refinement_count=2)

    with patch("listing_agent.cli.build_graph", mock_build):
        result = runner.invoke(app, ["generate", "A cool widget"])

    assert result.exit_code == 0
    assert "SHOPIFY" in result.output
    assert "Shopify Widget" in result.output
    assert "Refinements: 2" in result.output


def test_generate_json_output():
    listings = [_make_listing("amazon", score=0.6)]
    mock_build = _mock_graph(listings)

    with patch("listing_agent.cli.build_graph", mock_build):
        result = runner.invoke(app, ["generate", "A widget", "--json"])

    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["platform"] == "amazon"
    assert data[0]["title"] == "Amazon Widget"


def test_generate_exits_on_errors():
    mock_build = _mock_graph([], errors=["LLM call failed"])

    with patch("listing_agent.cli.build_graph", mock_build):
        result = runner.invoke(app, ["generate", "broken product"])

    assert result.exit_code == 1
    assert "LLM call failed" in result.output


def test_generate_platforms_option():
    listings = [_make_listing("etsy", score=0.4)]
    mock_build = _mock_graph(listings)

    with patch("listing_agent.cli.build_graph", mock_build) as mb:
        result = runner.invoke(app, ["generate", "widget", "--platforms", "etsy"])

    assert result.exit_code == 0
    mock_compiled = mb.return_value
    call_args = mock_compiled.invoke.call_args[0][0]
    assert call_args["target_platforms"] == ["etsy"]
    assert call_args["raw_product_data"] == {"description": "widget"}


def test_ingest_command():
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "Ingesting" in result.output
