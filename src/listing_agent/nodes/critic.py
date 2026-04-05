"""Critic node: hybrid scoring (rules + LLM judge) with structured feedback."""

from __future__ import annotations

from typing import Any

from listing_agent.config import get_config
from listing_agent.scoring.llm_judge import LLMJudge
from listing_agent.scoring.rubric import ScoringRubric
from listing_agent.scoring.rules import RulesScorer
from listing_agent.state import AgentState, GeneratedListing


def _build_feedback(result) -> str:
    """Build structured feedback string from CompositeResult."""
    parts: list[str] = []
    if result.violations:
        parts.append("VIOLATIONS: " + "; ".join(result.violations))
    if result.improvements:
        parts.append("IMPROVEMENTS: " + "; ".join(result.improvements))
    if result.keep:
        parts.append("KEEP: " + "; ".join(result.keep))
    return " | ".join(parts) if parts else "Looks good."


def critique_listings(state: AgentState) -> dict[str, Any]:
    """LangGraph node: score each listing with rules + optional LLM judge."""
    config = get_config()
    rules_scorer = RulesScorer()
    rubric = ScoringRubric(
        rules_weight=config.RULES_WEIGHT,
        llm_weight=config.LLM_WEIGHT,
        convergence_delta=config.CONVERGENCE_DELTA,
        oscillation_window=config.OSCILLATION_WINDOW,
    )

    refinement_count = state.get("refinement_count", 0)
    use_judge = refinement_count == 0 or config.LLM_JUDGE_SAMPLE_RATE >= 1.0
    judge = LLMJudge() if use_judge else None

    attrs = state.get("product_attributes")
    listings: list[GeneratedListing] = state.get("listings", [])
    scored: list[GeneratedListing] = []

    keywords = list(attrs.keywords) if attrs else []

    for listing in listings:
        rules_result = rules_scorer.score(listing, primary_keywords=keywords)
        judge_result = judge.evaluate(listing, attrs) if judge else None
        composite = rubric.composite(rules_result, judge_result)

        feedback = _build_feedback(composite)
        scored.append(listing.model_copy(update={
            "score": composite.overall_score,
            "feedback": feedback,
        }))

    avg_score = sum(l.score or 0.0 for l in scored) / max(len(scored), 1)
    history = list(state.get("score_history", []))
    history.append(avg_score)

    return {
        "listings": scored,
        "refinement_count": refinement_count + 1,
        "score_history": history,
    }


def should_refine(state: AgentState) -> str:
    """Conditional edge: 'done' or 'refine'."""
    config = get_config()
    max_refinements: int = state.get("max_refinements", config.MAX_REFINEMENTS)
    quality_threshold: float = state.get("quality_threshold", config.QUALITY_THRESHOLD)
    refinement_count: int = state.get("refinement_count", 0)

    if refinement_count >= max_refinements:
        return "done"

    listings: list[GeneratedListing] = state.get("listings", [])
    if not listings:
        return "done"

    if all((listing.score or 0.0) >= quality_threshold for listing in listings):
        return "done"

    rubric = ScoringRubric(
        convergence_delta=config.CONVERGENCE_DELTA,
        oscillation_window=config.OSCILLATION_WINDOW,
    )
    history = state.get("score_history", [])
    if rubric.is_converged(history):
        return "done"
    if rubric.is_oscillating(history):
        return "done"

    return "refine"
