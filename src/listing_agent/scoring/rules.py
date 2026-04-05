"""Deterministic rules-based scorer — 6 dimensions from 04-quality-system.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from listing_agent.state import GeneratedListing

# Platform constraints
_TITLE_MAX = {"shopify": 255, "amazon": 200, "etsy": 140}
_TITLE_MIN = 30
_DESC_MAX = {"shopify": 100_000, "amazon": 2000, "etsy": 65_535}
_AMAZON_BULLET_COUNT = 5
_AMAZON_BULLET_MAX_CHARS = 500
_AMAZON_KEYWORD_MAX_BYTES = 249
_ETSY_TAG_MAX = 13
_ETSY_TAG_CHAR_MAX = 20
_SHOPIFY_SEO_TITLE_MAX = 60
_SHOPIFY_SEO_DESC_MAX = 160

_PROHIBITED_HTML = {"script", "iframe", "style", "link", "object", "embed"}


@dataclass
class RulesResult:
    """Output of RulesScorer.score()."""
    dimensions: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


# Dimension weights
_WEIGHTS = {
    "title_length_compliance": 0.08,
    "bullet_compliance": 0.05,
    "keyword_presence": 0.15,
    "readability": 0.10,
    "char_limit_compliance": 0.12,
    "html_validity": 0.10,
}


class RulesScorer:
    """Deterministic rules-based listing scorer."""

    def score(
        self,
        listing: GeneratedListing,
        primary_keywords: list[str] | None = None,
    ) -> RulesResult:
        dims: dict[str, float] = {}
        violations: list[str] = []
        suggestions: list[str] = []
        kws = primary_keywords or []

        dims["title_length_compliance"] = self._title_length(listing, violations)
        dims["bullet_compliance"] = self._bullet_compliance(listing, violations)
        dims["keyword_presence"] = self._keyword_presence(listing, kws, violations, suggestions)
        dims["readability"] = self._readability(listing)
        dims["char_limit_compliance"] = self._char_limits(listing, violations)
        dims["html_validity"] = self._html_validity(listing, violations)

        composite = sum(dims[k] * _WEIGHTS[k] for k in _WEIGHTS)
        # Normalize to 0-1 (weights sum to 0.60)
        composite = min(composite / sum(_WEIGHTS.values()), 1.0)

        return RulesResult(
            dimensions=dims,
            composite=composite,
            violations=violations,
            suggestions=suggestions,
        )

    def _title_length(self, listing: GeneratedListing, violations: list[str]) -> float:
        title_len = len(listing.title)
        limit = _TITLE_MAX.get(listing.platform, 255)
        if title_len > limit:
            violations.append(f"Title exceeds {listing.platform} limit ({title_len}/{limit} chars).")
            return 0.0
        if title_len < _TITLE_MIN:
            violations.append(f"Title too short ({title_len} chars, min {_TITLE_MIN}).")
            return 0.0
        return 1.0

    def _bullet_compliance(self, listing: GeneratedListing, violations: list[str]) -> float:
        if listing.platform != "amazon":
            return 1.0  # N/A for non-Amazon
        count = len(listing.bullet_points)
        if count != _AMAZON_BULLET_COUNT:
            violations.append(f"Amazon requires exactly 5 bullets (got {count}).")
            return 0.0
        for i, bp in enumerate(listing.bullet_points):
            if len(bp) > _AMAZON_BULLET_MAX_CHARS:
                violations.append(f"Bullet {i+1} exceeds 500 chars ({len(bp)}).")
                return 0.0
        return 1.0

    def _keyword_presence(
        self,
        listing: GeneratedListing,
        keywords: list[str],
        violations: list[str],
        suggestions: list[str],
    ) -> float:
        if not keywords:
            return 1.0  # no keywords to check
        # Require minimum content for keyword presence to matter
        word_count = len(listing.description.split())
        if len(listing.title) < _TITLE_MIN and word_count < 10:
            suggestions.append("Add more content to enable keyword optimization.")
            return 0.0
        title_lower = listing.title.lower()
        desc_lower = listing.description.lower()
        first_content = ""
        if listing.bullet_points:
            first_content = listing.bullet_points[0].lower()
        elif desc_lower:
            first_content = desc_lower[:500]

        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            in_title = kw_lower in title_lower
            in_first = kw_lower in first_content
            if in_title:
                score += 0.3 / len(keywords)
            if in_first:
                score += 0.2 / len(keywords)
            if in_title or kw_lower in desc_lower:
                score += 0.5 / len(keywords)
            else:
                suggestions.append(f"Add keyword '{kw}' to title or description.")
        return min(score, 1.0)

    def _readability(self, listing: GeneratedListing) -> float:
        """Simplified Flesch-Kincaid approximation."""
        text = listing.description
        clean = re.sub(r"<[^>]+>", " ", text)
        words = clean.split()
        if len(words) < 5:
            return 0.0
        if len(words) < 10:
            return 0.3
        sentences = max(len(re.split(r"[.!?]+", clean)) - 1, 1)
        syllables = sum(self._count_syllables(w) for w in words)
        grade = 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59
        if 8 <= grade <= 10:
            return 1.0
        elif 6 <= grade <= 12:
            return 0.7
        elif 4 <= grade <= 14:
            return 0.4
        return 0.2

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        if len(word) <= 2:
            return 1
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and not word.endswith("le"):
            count -= 1
        return max(count, 1)

    def _char_limits(self, listing: GeneratedListing, violations: list[str]) -> float:
        platform = listing.platform
        desc_max = _DESC_MAX.get(platform, 100_000)
        if len(listing.description) > desc_max:
            violations.append(f"Description exceeds {platform} limit ({len(listing.description)}/{desc_max}).")
            return 0.0
        if platform == "amazon" and len(listing.backend_keywords.encode("utf-8")) > _AMAZON_KEYWORD_MAX_BYTES:
            byte_count = len(listing.backend_keywords.encode("utf-8"))
            violations.append(f"Amazon backend_keywords exceeds 249 bytes ({byte_count} bytes).")
            return 0.0
        if platform == "etsy":
            if len(listing.tags) > _ETSY_TAG_MAX:
                violations.append(f"Etsy max 13 tags (got {len(listing.tags)}).")
                return 0.0
            for tag in listing.tags:
                if len(tag) > _ETSY_TAG_CHAR_MAX:
                    violations.append(f"Etsy tag '{tag}' exceeds 20 chars.")
                    return 0.0
        if platform == "shopify":
            if listing.seo_title and len(listing.seo_title) > _SHOPIFY_SEO_TITLE_MAX:
                violations.append(f"Shopify seo_title exceeds 60 chars ({len(listing.seo_title)}).")
                return 0.0
            if listing.seo_description and len(listing.seo_description) > _SHOPIFY_SEO_DESC_MAX:
                violations.append(f"Shopify seo_description exceeds 160 chars ({len(listing.seo_description)}).")
                return 0.0
        return 1.0

    def _html_validity(self, listing: GeneratedListing, violations: list[str]) -> float:
        if listing.platform != "shopify":
            return 1.0  # N/A
        desc = listing.description
        for tag in _PROHIBITED_HTML:
            if f"<{tag}" in desc.lower():
                violations.append(f"Prohibited HTML tag <{tag}> found in description.")
                return 0.0
        open_tags = re.findall(r"<(\w+)[^>]*>", desc)
        close_tags = re.findall(r"</(\w+)>", desc)
        self_closing = {"br", "hr", "img", "input", "meta"}
        for tag in open_tags:
            if tag.lower() not in self_closing and tag.lower() not in [c.lower() for c in close_tags]:
                violations.append(f"Unclosed HTML tag <{tag}>.")
                return 0.0
        return 1.0
