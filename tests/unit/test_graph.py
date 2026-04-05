"""Unit tests for the LangGraph graph wiring (graph.py)."""
from unittest.mock import MagicMock, patch

from listing_agent.graph import build_graph


def _make_config_mock():
    cfg = MagicMock()
    cfg.RULES_WEIGHT = 0.6
    cfg.LLM_WEIGHT = 0.4
    cfg.CONVERGENCE_DELTA = 0.03
    cfg.OSCILLATION_WINDOW = 2
    cfg.LLM_JUDGE_SAMPLE_RATE = 1.0
    cfg.MAX_REFINEMENTS = 3
    cfg.QUALITY_THRESHOLD = 0.7
    return cfg

# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

_ANALYZER_JSON = (
    '{"title": "Handmade Ceramic Mug", "category": "home_and_kitchen",'
    ' "features": ["12oz", "handmade"], "materials": ["ceramic"],'
    ' "target_audience": "coffee lovers", "price_range": "mid-range",'
    ' "brand": "", "keywords": ["ceramic", "mug"]}'
)

# Long enough title (>=50) + long description (>=200 chars, >=30 words) + >=5 tags
# + seo_title present → score >= 0.7, loop exits on first pass.
_GENERATOR_JSON = (
    '{"title": "Handmade Ceramic Coffee Mug 12oz | Minimalist Design",'
    ' "description": "A beautifully handcrafted ceramic mug perfect for coffee lovers.'
    " 12oz capacity, dishwasher safe. Made with care in our artisan studio."
    ' The minimalist design complements any kitchen decor perfectly.",'
    ' "tags": ["ceramic mug", "handmade", "coffee cup", "minimalist", "dishwasher safe"],'
    ' "bullet_points": ["12oz capacity", "Dishwasher safe", "Handmade craftsmanship"],'
    ' "seo_title": "Handmade Ceramic Mug | StudioCraft",'
    ' "seo_description": "Shop our handmade ceramic mug."}'
)

# Short listing → scores below 0.7 → loop keeps refining.
_GENERATOR_JSON_SHORT = (
    '{"title": "Mug",'
    ' "description": "A mug.",'
    ' "tags": [],'
    ' "bullet_points": [],'
    ' "seo_title": "",'
    ' "seo_description": ""}'
)

_INITIAL_STATE = {
    "raw_product_data": {"description": "handmade ceramic mug"},
    "target_platforms": ["shopify"],
}


def _make_retriever_mock() -> MagicMock:
    retriever = MagicMock()
    retriever.get_rules.return_value = "No specific rules."
    return retriever


# ---------------------------------------------------------------------------
# Test 1: Happy-path end-to-end
# ---------------------------------------------------------------------------


def test_graph_end_to_end_happy_path():
    """Graph runs to completion with valid mocks; final state has scored listings."""
    retriever_mock = _make_retriever_mock()

    # First call (analyzer) → analyzer JSON; subsequent calls (generator) → generator JSON
    call_count = {"n": 0}

    def fake_analyzer(prompt: str) -> str:
        return _ANALYZER_JSON

    def fake_generator(prompt: str) -> str:
        call_count["n"] += 1
        return _GENERATOR_JSON

    with (
        patch("listing_agent.nodes.analyzer.invoke_with_fallback", side_effect=fake_analyzer),
        patch("listing_agent.nodes.generator.invoke_with_fallback", side_effect=fake_generator),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
        patch("listing_agent.nodes.critic.LLMJudge") as mock_judge_cls,
        patch("listing_agent.nodes.approval.interrupt", return_value={"decision": "approve_all"}),
        patch("listing_agent.nodes.publisher.shopify_create_product") as mock_shopify,
    ):
        mock_judge_inst = MagicMock()
        from listing_agent.scoring.llm_judge import JudgeResult
        mock_judge_inst.evaluate.return_value = JudgeResult(composite=0.8, improvements=[])
        mock_judge_cls.return_value = mock_judge_inst
        mock_shopify.invoke.return_value = {"status": "success", "product_id": "123"}
        graph = build_graph()
        result = graph.invoke(_INITIAL_STATE)

    assert "listings" in result
    listings = result["listings"]
    assert len(listings) == 1
    listing = listings[0]
    assert listing.platform == "shopify"
    assert listing.score is not None
    assert listing.score >= 0.0
    # Loop exited cleanly — errors should be absent or empty
    assert result.get("errors", []) == []
    assert result.get("publish_results", {}).get("shopify", {}).get("status") == "success"


# ---------------------------------------------------------------------------
# Test 2: Refinement loop reaches max_refinements
# ---------------------------------------------------------------------------


def test_graph_refinement_loop_reaches_max():
    """Short listing scores below threshold; loop runs until max_refinements=2."""
    retriever_mock = _make_retriever_mock()

    initial_state = {**_INITIAL_STATE, "max_refinements": 2}

    with (
        patch("listing_agent.nodes.analyzer.invoke_with_fallback", return_value=_ANALYZER_JSON),
        patch("listing_agent.nodes.generator.invoke_with_fallback", return_value=_GENERATOR_JSON_SHORT),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
        patch("listing_agent.nodes.critic.LLMJudge") as mock_judge_cls,
        patch("listing_agent.nodes.approval.interrupt", return_value={"decision": "approve_all"}),
        patch("listing_agent.nodes.publisher.shopify_create_product") as mock_shopify,
    ):
        mock_judge_inst = MagicMock()
        from listing_agent.scoring.llm_judge import JudgeResult
        mock_judge_inst.evaluate.return_value = JudgeResult(composite=0.1, improvements=[])
        mock_judge_cls.return_value = mock_judge_inst
        mock_shopify.invoke.return_value = {"status": "success", "product_id": "123"}
        graph = build_graph()
        result = graph.invoke(initial_state)

    assert result.get("refinement_count", 0) == 2


# ---------------------------------------------------------------------------
# Test 3: Error propagation from analyzer
# ---------------------------------------------------------------------------


def test_graph_error_propagation_invalid_analyzer_json():
    """Analyzer returns invalid JSON → errors list is populated, graph still terminates."""
    retriever_mock = _make_retriever_mock()

    with (
        patch("listing_agent.nodes.analyzer.invoke_with_fallback", return_value="not valid json {{{}"),
        patch("listing_agent.nodes.generator.invoke_with_fallback", return_value=_GENERATOR_JSON),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
        patch("listing_agent.nodes.critic.LLMJudge") as mock_judge_cls,
        patch("listing_agent.nodes.approval.interrupt", return_value={"decision": "approve_all"}),
        patch("listing_agent.nodes.publisher.shopify_create_product") as mock_shopify,
    ):
        mock_judge_inst = MagicMock()
        from listing_agent.scoring.llm_judge import JudgeResult
        mock_judge_inst.evaluate.return_value = JudgeResult(composite=0.5, improvements=[])
        mock_judge_cls.return_value = mock_judge_inst
        mock_shopify.invoke.return_value = {"status": "success", "product_id": "123"}
        graph = build_graph()
        result = graph.invoke(_INITIAL_STATE)

    errors = result.get("errors", [])
    assert len(errors) >= 1
    assert any("Failed to" in e for e in errors)


# ---------------------------------------------------------------------------
# Test 4: Selective refinement
# ---------------------------------------------------------------------------


def test_selective_refinement_preserves_passing():
    """Only below-threshold listings are re-generated."""
    from listing_agent.state import GeneratedListing, ProductAttributes

    # Two platforms: shopify passes (score >= 0.8), amazon fails (score < 0.8)
    shopify_passing = GeneratedListing(
        platform="shopify",
        title="Handmade Ceramic Coffee Mug 12oz | Minimalist Design",
        description="A beautifully handcrafted ceramic mug perfect for coffee lovers. 12oz capacity.",
        tags=["ceramic mug", "handmade"],
        seo_title="Handmade Ceramic Mug",
        score=0.9,  # above threshold — should NOT be regenerated
        feedback="Looks good.",
    )
    amazon_failing = GeneratedListing(
        platform="amazon",
        title="Mug",
        description="A mug.",
        score=0.2,  # below threshold — should be regenerated
        feedback="VIOLATIONS: Title too short",
    )

    attrs = ProductAttributes(
        title="Handmade Ceramic Mug",
        category="home_and_kitchen",
        features=["12oz"],
        keywords=["ceramic mug"],
        raw_input="handmade ceramic mug",
    )

    state = {
        "raw_product_data": {"description": "handmade ceramic mug"},
        "target_platforms": ["shopify", "amazon"],
        "product_attributes": attrs,
        "platform_rules": {},
        "listings": [shopify_passing, amazon_failing],
        "quality_threshold": 0.8,
        "refinement_count": 1,
    }

    invoke_calls = []

    def fake_invoke(prompt: str) -> str:
        invoke_calls.append(prompt)
        return _GENERATOR_JSON

    with patch("listing_agent.nodes.generator.invoke_with_fallback", side_effect=fake_invoke):
        from listing_agent.nodes.generator import generate_listings
        result = generate_listings(state)

    assert "listings" in result
    result_listings = result["listings"]

    # shopify was preserved (not regenerated)
    shopify_result = next((l for l in result_listings if l.platform == "shopify"), None)
    assert shopify_result is not None
    assert shopify_result.score == 0.9  # preserved original score

    # LLM was only called once (for amazon)
    assert len(invoke_calls) == 1


# ---------------------------------------------------------------------------
# Test 5: checkpointer parameter
# ---------------------------------------------------------------------------


def test_graph_accepts_checkpointer():
    graph = build_graph(checkpointer=None)
    assert graph is not None


# ---------------------------------------------------------------------------
# Test 6: include_publishing=False routes to END
# ---------------------------------------------------------------------------


def test_graph_no_publishing_routes_to_end():
    """build_graph(include_publishing=False) skips approval+publisher nodes."""
    retriever_mock = _make_retriever_mock()

    with (
        patch("listing_agent.nodes.analyzer.invoke_with_fallback", return_value=_ANALYZER_JSON),
        patch("listing_agent.nodes.generator.invoke_with_fallback", return_value=_GENERATOR_JSON),
        patch("listing_agent.nodes.researcher._retriever", None),
        patch(
            "listing_agent.nodes.researcher.PlatformRetriever",
            return_value=retriever_mock,
        ),
        patch("listing_agent.nodes.critic.get_config", return_value=_make_config_mock()),
        patch("listing_agent.nodes.critic.LLMJudge") as mock_judge_cls,
    ):
        mock_judge_inst = MagicMock()
        from listing_agent.scoring.llm_judge import JudgeResult
        mock_judge_inst.evaluate.return_value = JudgeResult(composite=0.8, improvements=[])
        mock_judge_cls.return_value = mock_judge_inst
        graph = build_graph(include_publishing=False)
        result = graph.invoke(_INITIAL_STATE)

    assert "listings" in result
    # No publish_results since publisher was skipped
    assert "publish_results" not in result or result.get("publish_results") is None
