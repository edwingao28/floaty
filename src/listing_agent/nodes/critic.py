from typing import Any

from listing_agent.state import AgentState, GeneratedListing

_PLATFORM_TITLE_LIMITS = {"shopify": 255, "amazon": 200, "etsy": 140}


def _score_listing(listing: GeneratedListing) -> tuple[float, str]:
    score = 0.0
    issues: list[str] = []

    # Title (0–0.3)
    title_len = len(listing.title)
    if title_len >= 30:
        score += 0.1
    else:
        issues.append("Title too short (< 30 chars).")
    if title_len >= 50:
        score += 0.1
    else:
        issues.append("Title too short (< 50 chars).")
    limit = _PLATFORM_TITLE_LIMITS.get(listing.platform, 255)
    if title_len <= limit:
        score += 0.1
    else:
        issues.append(f"Title exceeds {listing.platform} limit ({limit} chars).")

    # Description (0–0.4)
    desc_len = len(listing.description)
    word_count = len(listing.description.split())
    if desc_len >= 100:
        score += 0.15
    else:
        issues.append("Description too short (< 100 chars).")
    if desc_len >= 200:
        score += 0.1
    else:
        issues.append("Description too short (< 200 chars).")
    if word_count >= 30:
        score += 0.15
    else:
        issues.append(f"Description needs more words (< 30 words, got {word_count}).")

    # Tags (0–0.2)
    tag_count = len(listing.tags)
    if tag_count >= 3:
        score += 0.1
    else:
        issues.append("Needs more tags (< 3).")
    if tag_count >= 5:
        score += 0.1
    else:
        issues.append("Needs more tags (< 5).")

    # Platform-specific (0–0.1)
    if listing.platform == "amazon":
        if len(listing.bullet_points) >= 3:
            score += 0.1
        else:
            issues.append("Amazon listing needs >= 3 bullet points.")
    elif listing.platform == "shopify":
        if listing.seo_title:
            score += 0.1
        else:
            issues.append("Shopify listing needs a non-empty seo_title.")
    elif listing.platform == "etsy":
        if tag_count >= 5:
            score += 0.1
        else:
            issues.append("Etsy listing needs >= 5 tags.")

    score = min(score, 1.0)
    feedback = " ".join(issues) if issues else "Looks good."
    return score, feedback


def critique_listings(state: AgentState) -> dict[str, Any]:
    """LangGraph node: score each listing with deterministic heuristics."""
    listings: list[GeneratedListing] = state.get("listings", [])
    scored: list[GeneratedListing] = []
    for listing in listings:
        score, feedback = _score_listing(listing)
        scored.append(listing.model_copy(update={"score": score, "feedback": feedback}))
    return {
        "listings": scored,
        "refinement_count": state.get("refinement_count", 0) + 1,
    }


def should_refine(state: AgentState) -> str:
    """Conditional edge: 'done' or 'refine'."""
    max_refinements: int = state.get("max_refinements", 3)
    quality_threshold: float = state.get("quality_threshold", 0.7)
    refinement_count: int = state.get("refinement_count", 0)

    if refinement_count >= max_refinements:
        return "done"

    listings: list[GeneratedListing] = state.get("listings", [])
    if listings and all(
        (l.score or 0.0) >= quality_threshold for l in listings
    ):
        return "done"

    return "refine"
