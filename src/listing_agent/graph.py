from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from listing_agent.nodes.analyzer import analyze_product
from listing_agent.nodes.approval import approve_listings
from listing_agent.nodes.critic import critique_listings, should_refine
from listing_agent.nodes.generator import generate_listings
from listing_agent.nodes.publisher import publish_listings
from listing_agent.nodes.researcher import research_platforms
from listing_agent.state import AgentState


def build_graph(checkpointer: Any = None, include_publishing: bool = True) -> CompiledStateGraph:
    """Build and compile the listing-agent StateGraph.

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver, SqliteSaver, etc.)
                      None for no persistence.
        include_publishing: When True, routes through approval + publisher after
                            critique. When False, routes directly to END.
    """
    graph = StateGraph(AgentState)

    graph.add_node("analyze_product", analyze_product)
    graph.add_node("research_platforms", research_platforms)
    graph.add_node("generate_listings", generate_listings)
    graph.add_node("critique_listings", critique_listings)

    graph.add_edge(START, "analyze_product")
    graph.add_edge("analyze_product", "research_platforms")
    graph.add_edge("research_platforms", "generate_listings")
    graph.add_edge("generate_listings", "critique_listings")

    if include_publishing:
        graph.add_node("approve_listings", approve_listings)
        graph.add_node("publish_listings", publish_listings)
        graph.add_conditional_edges(
            "critique_listings",
            should_refine,
            {"refine": "generate_listings", "done": "approve_listings"},
        )
        graph.add_edge("approve_listings", "publish_listings")
        graph.add_edge("publish_listings", END)
    else:
        graph.add_conditional_edges(
            "critique_listings",
            should_refine,
            {"refine": "generate_listings", "done": END},
        )

    return graph.compile(checkpointer=checkpointer)
