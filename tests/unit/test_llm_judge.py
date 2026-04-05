import json
from unittest.mock import MagicMock, patch

from listing_agent.scoring.llm_judge import LLMJudge, JudgeResult
from listing_agent.state import GeneratedListing, ProductAttributes


def _listing() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Handmade Ceramic Coffee Mug 12oz | Dishwasher Safe",
        description="A beautifully crafted ceramic mug for coffee lovers.",
        tags=["ceramic", "mug", "handmade"],
    )


def _attrs() -> ProductAttributes:
    return ProductAttributes(
        title="Handmade Ceramic Mug",
        category="home_and_kitchen",
        features=["12oz", "dishwasher safe"],
        keywords=["ceramic", "mug"],
        raw_input="handmade ceramic mug",
    )


_JUDGE_JSON = json.dumps({
    "persuasiveness": {"score": 4, "justification": "Clear benefits stated"},
    "brand_voice": {"score": 3, "justification": "Neutral tone"},
    "usp_clarity": {"score": 5, "justification": "Differentiator obvious"},
    "competitive_positioning": {"score": 3, "justification": "Average positioning"},
    "improvements": ["Add urgency", "Mention warranty"],
})


def test_judge_returns_result():
    mock_response = MagicMock()
    mock_response.content = _JUDGE_JSON

    with patch("listing_agent.scoring.llm_judge._get_judge_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get.return_value = mock_llm

        judge = LLMJudge()
        result = judge.evaluate(_listing(), _attrs())

    assert isinstance(result, JudgeResult)
    assert 0.0 <= result.composite <= 1.0
    assert "persuasiveness" in result.dimensions
    assert len(result.improvements) == 2


def test_judge_handles_malformed_json():
    mock_response = MagicMock()
    mock_response.content = "not json"

    with patch("listing_agent.scoring.llm_judge._get_judge_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get.return_value = mock_llm

        judge = LLMJudge()
        result = judge.evaluate(_listing(), _attrs())

    assert result.composite == 0.0
    assert len(result.errors) > 0


def test_judge_normalizes_1_5_to_0_1():
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "persuasiveness": {"score": 5, "justification": "x"},
        "brand_voice": {"score": 5, "justification": "x"},
        "usp_clarity": {"score": 5, "justification": "x"},
        "competitive_positioning": {"score": 5, "justification": "x"},
        "improvements": [],
    })
    with patch("listing_agent.scoring.llm_judge._get_judge_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get.return_value = mock_llm

        judge = LLMJudge()
        result = judge.evaluate(_listing(), _attrs())

    assert result.composite == 1.0
