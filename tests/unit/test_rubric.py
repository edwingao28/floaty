from listing_agent.scoring.rubric import ScoringRubric, CompositeResult
from listing_agent.scoring.rules import RulesResult
from listing_agent.scoring.llm_judge import JudgeResult


def test_composite_rules_only():
    rubric = ScoringRubric(rules_weight=0.6, llm_weight=0.4)
    rules = RulesResult(dimensions={"title_length_compliance": 1.0}, composite=0.9)
    result = rubric.composite(rules_result=rules, judge_result=None)
    assert isinstance(result, CompositeResult)
    assert result.overall_score == 0.9


def test_composite_both():
    rubric = ScoringRubric(rules_weight=0.6, llm_weight=0.4)
    rules = RulesResult(composite=0.8)
    judge = JudgeResult(composite=0.6, improvements=["Add urgency"])
    result = rubric.composite(rules_result=rules, judge_result=judge)
    expected = 0.8 * 0.6 + 0.6 * 0.4
    assert abs(result.overall_score - expected) < 0.01


def test_composite_feedback_structured():
    rubric = ScoringRubric()
    rules = RulesResult(
        composite=0.5,
        violations=["Title too short"],
        suggestions=["Add keyword 'mug'"],
    )
    judge = JudgeResult(composite=0.7, improvements=["Add urgency"])
    result = rubric.composite(rules_result=rules, judge_result=judge)
    assert "Title too short" in result.violations
    assert "Add keyword 'mug'" in result.improvements
    assert "Add urgency" in result.improvements


def test_convergence_detected():
    rubric = ScoringRubric(convergence_delta=0.03)
    assert rubric.is_converged(scores=[0.70, 0.72]) is True
    assert rubric.is_converged(scores=[0.70, 0.80]) is False


def test_oscillation_detected():
    rubric = ScoringRubric(oscillation_window=2)
    assert rubric.is_oscillating(scores=[0.70, 0.80, 0.71]) is True
    assert rubric.is_oscillating(scores=[0.70, 0.80, 0.90]) is False
    assert rubric.is_oscillating(scores=[0.70]) is False
