from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ProductAttributes(BaseModel):
    """Structured output from the Analyzer node."""

    title: str
    category: str
    features: list[str]
    materials: list[str] = []
    target_audience: str = ""
    price_range: str = ""
    brand: str = ""
    keywords: list[str]
    raw_input: str


class PlatformRules(BaseModel):
    """Retrieved SEO rules per platform."""

    platform: Literal["shopify", "amazon", "etsy"]
    title_constraints: dict[str, Any] = {}
    description_constraints: dict[str, Any] = {}
    keyword_rules: dict[str, Any] = {}
    category_taxonomy: dict[str, Any] = {}
    additional_rules: list[str] = []


class GeneratedListing(BaseModel):
    """A single platform listing."""

    platform: Literal["shopify", "amazon", "etsy"]
    title: str
    description: str
    bullet_points: list[str] = []
    tags: list[str] = []
    seo_title: str = ""
    seo_description: str = ""
    backend_keywords: str = ""
    category_id: str = ""
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    feedback: str | None = None
    iteration: int = Field(default=0, ge=0)


class _AgentStateRequired(TypedDict):
    raw_product_data: dict[str, Any]
    target_platforms: list[str]


class AgentState(_AgentStateRequired, total=False):
    """LangGraph root state flowing through every node."""

    product_attributes: ProductAttributes | None
    platform_rules: dict[str, str]
    listings: list[GeneratedListing]
    approved_listings: list[GeneratedListing]
    refinement_count: int
    max_refinements: int
    quality_threshold: float
    publish_results: dict[str, Any]
    score_history: list[float]  # per-iteration avg scores for convergence detection
    errors: Annotated[list[str], operator.add]
