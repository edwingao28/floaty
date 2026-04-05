"""LLM-as-Judge scorer using Claude Haiku — 4 subjective dimensions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic

from listing_agent.state import GeneratedListing, ProductAttributes

_JUDGE_WEIGHTS = {
    "persuasiveness": 0.12,
    "brand_voice": 0.10,
    "usp_clarity": 0.10,
    "competitive_positioning": 0.08,
}

_JUDGE_PROMPT = """You are a listing quality judge. Score this e-commerce listing on 4 dimensions.

## Listing
Platform: PLATFORM_PH
Title: TITLE_PH
Description: DESCRIPTION_PH
Bullet points: BULLETS_PH
Tags: TAGS_PH

## Product Info
Name: PRODUCT_TITLE_PH
Category: CATEGORY_PH
Features: FEATURES_PH

## Scoring Rubric (1-5 scale per dimension)

**Persuasiveness** — Does the listing compel purchase?
- 1: Generic, no value proposition
- 3: Adequate, mentions benefits but not compelling
- 5: Compelling, clear call-to-action, benefits-first language

**Brand Voice** — Professional, consistent tone?
- 1: Robotic or inconsistent tone
- 3: Neutral, professional but generic
- 5: Distinctive, appropriate for product category

**USP Clarity** — Is the primary differentiator obvious within first 2 sentences?
- 1: No clear differentiator
- 3: Differentiator present but buried
- 5: Differentiator immediately clear

**Competitive Positioning** — Does it stand out from generic listings?
- 1: Could describe any product in this category
- 3: Some unique elements
- 5: Clearly differentiated from competitors

Return ONLY valid JSON:
{
  "persuasiveness": {"score": N, "justification": "..."},
  "brand_voice": {"score": N, "justification": "..."},
  "usp_clarity": {"score": N, "justification": "..."},
  "competitive_positioning": {"score": N, "justification": "..."},
  "improvements": ["specific suggestion 1", "specific suggestion 2"]
}"""


@dataclass
class JudgeResult:
    dimensions: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    improvements: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _get_judge_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)


class LLMJudge:
    def evaluate(
        self,
        listing: GeneratedListing,
        attrs: ProductAttributes | None = None,
    ) -> JudgeResult:
        try:
            llm = _get_judge_llm()
            prompt = (
                _JUDGE_PROMPT
                .replace("PLATFORM_PH", listing.platform)
                .replace("TITLE_PH", listing.title)
                .replace("DESCRIPTION_PH", listing.description[:1000])
                .replace("BULLETS_PH", "; ".join(listing.bullet_points) if listing.bullet_points else "N/A")
                .replace("TAGS_PH", ", ".join(listing.tags) if listing.tags else "N/A")
                .replace("PRODUCT_TITLE_PH", attrs.title if attrs else "Unknown")
                .replace("CATEGORY_PH", attrs.category if attrs else "Unknown")
                .replace("FEATURES_PH", ", ".join(attrs.features) if attrs else "Unknown")
            )
            response = llm.invoke(prompt)
            data = json.loads(response.content)
            dims: dict[str, float] = {}
            for dim in _JUDGE_WEIGHTS:
                raw_score = data.get(dim, {}).get("score", 1)
                dims[dim] = (raw_score - 1) / 4.0  # normalize 1-5 → 0-1
            composite = sum(dims[k] * _JUDGE_WEIGHTS[k] for k in _JUDGE_WEIGHTS)
            composite = min(composite / sum(_JUDGE_WEIGHTS.values()), 1.0)
            return JudgeResult(
                dimensions=dims,
                composite=composite,
                improvements=data.get("improvements", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return JudgeResult(errors=[f"Judge parse error: {e}"])
        except Exception as e:
            return JudgeResult(errors=[f"Judge error: {e}"])
