from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel


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

    platform: str  # "shopify" | "amazon" | "etsy"
    title_constraints: dict = {}
    description_constraints: dict = {}
    keyword_rules: dict = {}
    category_taxonomy: dict = {}
    additional_rules: list[str] = []


class GeneratedListing(BaseModel):
    """A single platform listing."""

    platform: str
    title: str
    description: str
    bullet_points: list[str] = []
    tags: list[str] = []
    seo_title: str = ""
    seo_description: str = ""
    backend_keywords: str = ""
    category_id: str = ""
    score: float | None = None
    feedback: str | None = None
    iteration: int = 0


class AgentState(TypedDict, total=False):
    """LangGraph root state flowing through every node."""

    raw_product_data: dict
    product_attributes: ProductAttributes | None
    target_platforms: list[str]
    platform_rules: list[PlatformRules]
    listings: Annotated[list[GeneratedListing], operator.add]
    approved_listings: list[GeneratedListing]
    refinement_count: int
    max_refinements: int
    quality_threshold: float
    publish_results: dict
    errors: Annotated[list[str], operator.add]
