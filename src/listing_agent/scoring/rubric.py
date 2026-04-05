"""Composite scoring with configurable weights and early-stop detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from listing_agent.scoring.rules import RulesResult
from listing_agent.scoring.llm_judge import JudgeResult


@dataclass
class CompositeResult:
    overall_score: float = 0.0
    rules_score: float = 0.0
    llm_score: float | None = None
    violations: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)


class ScoringRubric:
    def __init__(
        self,
        rules_weight: float = 0.6,
        llm_weight: float = 0.4,
        convergence_delta: float = 0.03,
        oscillation_window: int = 2,
    ):
        self.rules_weight = rules_weight
        self.llm_weight = llm_weight
        self.convergence_delta = convergence_delta
        self.oscillation_window = oscillation_window

    def composite(
        self,
        rules_result: RulesResult,
        judge_result: JudgeResult | None = None,
    ) -> CompositeResult:
        violations = list(rules_result.violations)
        improvements = list(rules_result.suggestions)
        keep: list[str] = [dim for dim, score in rules_result.dimensions.items() if score >= 1.0]

        if judge_result is not None and not judge_result.errors:
            overall = (
                rules_result.composite * self.rules_weight
                + judge_result.composite * self.llm_weight
            )
            improvements.extend(judge_result.improvements)
            return CompositeResult(
                overall_score=overall,
                rules_score=rules_result.composite,
                llm_score=judge_result.composite,
                violations=violations,
                improvements=improvements,
                keep=keep,
            )
        return CompositeResult(
            overall_score=rules_result.composite,
            rules_score=rules_result.composite,
            llm_score=None,
            violations=violations,
            improvements=improvements,
            keep=keep,
        )

    def is_converged(self, scores: list[float]) -> bool:
        if len(scores) < 2:
            return False
        delta = abs(scores[-1] - scores[-2])
        return delta < self.convergence_delta

    def is_oscillating(self, scores: list[float]) -> bool:
        w = self.oscillation_window
        if len(scores) < w + 1:
            return False
        return abs(scores[-1] - scores[-(w + 1)]) < 0.02
