from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from listing_agent.nodes.analyzer import analyze_product
from listing_agent.nodes.critic import critique_listings, should_refine
from listing_agent.nodes.generator import generate_listings
from listing_agent.nodes.researcher import research_platforms
from listing_agent.state import AgentState


def build_graph() -> CompiledStateGraph:
    """Build and compile the listing-agent StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("analyze_product", analyze_product)
    graph.add_node("research_platforms", research_platforms)
    graph.add_node("generate_listings", generate_listings)
    graph.add_node("critique_listings", critique_listings)

    graph.add_edge(START, "analyze_product")
    graph.add_edge("analyze_product", "research_platforms")
    graph.add_edge("research_platforms", "generate_listings")
    graph.add_edge("generate_listings", "critique_listings")

    graph.add_conditional_edges(
        "critique_listings",
        should_refine,
        {"refine": "generate_listings", "done": END},
    )

    return graph.compile()
