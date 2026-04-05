"""Approval node: pauses for human review via LangGraph interrupt()."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from listing_agent.state import AgentState, GeneratedListing


def approve_listings(state: AgentState) -> dict[str, Any]:
    """LangGraph node: interrupt for human approval before publishing."""
    listings: list[GeneratedListing] = state.get("listings", [])

    previews = []
    for listing in listings:
        previews.append({
            "platform": listing.platform,
            "title": listing.title,
            "description": listing.description[:200],
            "score": listing.score,
        })

    decision = interrupt({"listings": previews})

    if not isinstance(decision, dict):
        return {"approved_listings": listings}

    action = decision.get("decision", "approve_all")

    if action == "reject_all":
        return {"approved_listings": []}

    if action == "approve_selective":
        approved_platforms = set(decision.get("platforms", []))
        approved = [listing for listing in listings if listing.platform in approved_platforms]
        return {"approved_listings": approved}

    # approve_all (default)
    return {"approved_listings": listings}
