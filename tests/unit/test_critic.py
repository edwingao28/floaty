from unittest.mock import MagicMock, patch

import pytest

from listing_agent.nodes.critic import critique_listings, should_refine
from listing_agent.scoring.llm_judge import JudgeResult
from listing_agent.scoring.rules import RulesResult
from listing_agent.scoring.rubric import CompositeResult
from listing_agent.state import AgentState, GeneratedListing


# --- fixtures ---

def good_shopify_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Handmade Ceramic Coffee Mug Minimalist Design Studio Pottery",
        description=(
            "This beautiful handmade ceramic mug is crafted with care in our studio. "
            "It holds 12 ounces and features a minimalist design perfect for coffee or tea lovers. "
            "Dishwasher safe and microwave safe. A great gift for any occasion."
        ),
        tags=["ceramic", "mug", "handmade", "minimalist", "coffee"],
        seo_title="Handmade Ceramic Mug | Minimalist Studio Pottery",
    )


def bad_listing() -> GeneratedListing:
    return GeneratedListing(
        platform="shopify",
        title="Mug",
        description="A mug.",
        tags=[],
    )


def _make_config_mock(
    rules_weight=0.6,
    llm_weight=0.4,
    convergence_delta=0.03,
    oscillation_window=2,
    llm_judge_sample_rate=1.0,
    max_refinements=3,
    quality_threshold=0.8,
):
    cfg = MagicMock()
    cfg.RULES_WEIGHT = rules_weight
    cfg.LLM_WEIGHT = llm_weight
    cfg.CONVERGENCE_DELTA = convergence_delta
    cfg.OSCILLATION_WINDOW = oscillation_window
    cfg.LLM_JUDGE_SAMPLE_RATE = llm_judge_sample_rate
    cfg.MAX_REFINEMENTS = max_refinements
    cfg.QUALITY_THRESHOLD = quality_threshold
    return cfg


def _make_rules_mock(composite=0.8):
    mock = MagicMock()
    mock.score.return_value = RulesResult(
        dimensions={"title_length_compliance": 1.0},
        composite=composite,
        violations=[],
        suggestions=[],
    )
    return mock


def _make_judge_mock(composite=0.7):
    mock = MagicMock()
    mock.evaluate.return_value = JudgeResult(
        dimensions={"persuasiveness": 0.75},
        composite=composite,
        improvements=[],
    )
    return mock


def _make_rubric_mock(overall=0.75):
    mock = MagicMock()
    instance = MagicMock()
    instance.composite.return_value = CompositeResult(
        overall_score=overall,
        rules_score=0.8,
        llm_score=0.7,
        violations=[],
        improvements=[],
    )
    instance.is_converged.return_value = False
    instance.is_oscillating.return_value = False
    mock.return_value = instance
    return mock


# --- critique_listings tests ---

def test_critique_uses_rules_scorer():
    """RulesScorer.score() is called for each listing."""
    listing = good_shopify_listing()
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [listing],
        "refinement_count": 0,
    }

    mock_rules_cls = MagicMock()
    mock_rules_inst = _make_rules_mock()
    mock_rules_cls.return_value = mock_rules_inst

    with (
        patch("listing_agent.nodes.critic.RulesScorer", mock_rules_cls),
        patch("listing_agent.nodes.critic.LLMJudge", MagicMock()),
        patch("listing_agent.nodes.critic.ScoringRubric", _make_rubric_mock()),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
    ):
        critique_listings(state)

    mock_rules_inst.score.assert_called_once()


def test_critique_uses_llm_judge_on_first_iteration():
    """Judge runs when refinement_count=0."""
    listing = good_shopify_listing()
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [listing],
        "refinement_count": 0,
    }

    mock_judge_cls = MagicMock()
    mock_judge_inst = _make_judge_mock()
    mock_judge_cls.return_value = mock_judge_inst

    with (
        patch("listing_agent.nodes.critic.RulesScorer", return_value=_make_rules_mock()),
        patch("listing_agent.nodes.critic.LLMJudge", mock_judge_cls),
        patch("listing_agent.nodes.critic.ScoringRubric", _make_rubric_mock()),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
    ):
        critique_listings(state)

    mock_judge_cls.assert_called_once()
    mock_judge_inst.evaluate.assert_called_once()


def test_critique_listings_populates_score_and_feedback():
    listing = good_shopify_listing()
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [listing, bad_listing()],
        "refinement_count": 0,
    }

    with (
        patch("listing_agent.nodes.critic.RulesScorer", return_value=_make_rules_mock()),
        patch("listing_agent.nodes.critic.LLMJudge", MagicMock(return_value=_make_judge_mock())),
        patch("listing_agent.nodes.critic.ScoringRubric", _make_rubric_mock()),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
    ):
        result = critique_listings(state)

    assert "listings" in result
    assert "refinement_count" in result
    assert result["refinement_count"] == 1
    for l in result["listings"]:
        assert l.score is not None
        assert l.feedback is not None


def test_critique_listings_increments_refinement_count():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [good_shopify_listing()],
        "refinement_count": 2,
    }

    with (
        patch("listing_agent.nodes.critic.RulesScorer", return_value=_make_rules_mock()),
        patch("listing_agent.nodes.critic.LLMJudge", MagicMock(return_value=_make_judge_mock())),
        patch("listing_agent.nodes.critic.ScoringRubric", _make_rubric_mock()),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
    ):
        result = critique_listings(state)

    assert result["refinement_count"] == 3


def test_critique_listings_empty_listings():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": [],
        "listings": [],
    }

    with (
        patch("listing_agent.nodes.critic.RulesScorer", return_value=_make_rules_mock()),
        patch("listing_agent.nodes.critic.LLMJudge", MagicMock(return_value=_make_judge_mock())),
        patch("listing_agent.nodes.critic.ScoringRubric", _make_rubric_mock()),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
    ):
        result = critique_listings(state)

    assert result["listings"] == []
    assert result["refinement_count"] == 1


def test_critique_appends_score_history():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [good_shopify_listing()],
        "refinement_count": 0,
        "score_history": [0.6],
    }

    with (
        patch("listing_agent.nodes.critic.RulesScorer", return_value=_make_rules_mock()),
        patch("listing_agent.nodes.critic.LLMJudge", MagicMock(return_value=_make_judge_mock())),
        patch("listing_agent.nodes.critic.ScoringRubric", _make_rubric_mock(overall=0.75)),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
    ):
        result = critique_listings(state)

    assert len(result["score_history"]) == 2
    assert result["score_history"][0] == 0.6


# --- should_refine tests ---

def test_should_refine_done_when_all_scores_above_threshold():
    scored_listing = good_shopify_listing().model_copy(update={"score": 0.9})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "quality_threshold": 0.7,
        "max_refinements": 3,
        "score_history": [0.9],
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()):
        assert should_refine(state) == "done"


def test_should_refine_refine_when_scores_below_threshold():
    scored_listing = bad_listing().model_copy(update={"score": 0.2})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "quality_threshold": 0.7,
        "max_refinements": 3,
        "score_history": [0.2],
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()):
        assert should_refine(state) == "refine"


def test_should_refine_done_when_max_refinements_reached():
    scored_listing = bad_listing().model_copy(update={"score": 0.1})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 3,
        "quality_threshold": 0.7,
        "max_refinements": 3,
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()):
        assert should_refine(state) == "done"


def test_should_refine_uses_state_defaults():
    scored_listing = bad_listing().model_copy(update={"score": 0.2})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()):
        assert should_refine(state) == "refine"


def test_should_refine_done_when_listings_empty():
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [],
        "refinement_count": 1,
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()):
        assert should_refine(state) == "done"


def test_should_refine_done_when_refinement_count_exceeds_max():
    scored_listing = bad_listing().model_copy(update={"score": 0.1})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 5,
        "max_refinements": 3,
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()):
        assert should_refine(state) == "done"


def test_should_refine_convergence():
    """score_history with small delta → done."""
    scored_listing = bad_listing().model_copy(update={"score": 0.5})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "quality_threshold": 0.9,
        "max_refinements": 10,
        "score_history": [0.50, 0.51],  # delta 0.01 < 0.03
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock(convergence_delta=0.03)):
        assert should_refine(state) == "done"


def test_should_refine_oscillation():
    """Oscillating scores → done."""
    scored_listing = bad_listing().model_copy(update={"score": 0.5})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "quality_threshold": 0.9,
        "max_refinements": 10,
        "score_history": [0.70, 0.80, 0.71],  # oscillating
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock(oscillation_window=2)):
        assert should_refine(state) == "done"


def test_should_refine_threshold_0_8():
    """Default quality_threshold is 0.8."""
    scored_listing = good_shopify_listing().model_copy(update={"score": 0.75})
    state: AgentState = {
        "raw_product_data": {},
        "target_platforms": ["shopify"],
        "listings": [scored_listing],
        "refinement_count": 1,
        "score_history": [0.75],
    }
    with patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock(quality_threshold=0.8)):
        # 0.75 < 0.8 → should refine
        assert should_refine(state) == "refine"
