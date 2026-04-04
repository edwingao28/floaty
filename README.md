# listing-agent

> Generate platform-optimized e-commerce listings using LangGraph + RAG

CLI agent that takes a product description and generates optimized listings for Shopify, Amazon, and Etsy in one command.

## Architecture

```
Product Input
    |
    v
+-------------+      +-------------------+
|   Analyzer  | ---> | Platform Research  |
|  (LLM node) |      | (RAG: ChromaDB +  |
+-------------+      | SentenceTransf.)  |
                      +-------------------+
    |                        |
    v                        v
+--------------------------------------+
|        Generator Node (LLM)          |
| Shopify listing | Amazon listing     |
| Etsy listing    | ...                |
+--------------------------------------+
    |
    v
+--------------+    quality < 0.7?
| Critic Node  | ------------------> loop back (max 3x)
| (heuristic)  |
+--------------+
    | quality >= 0.7
    v
  Final Listings
```

## Tech Stack

- **LangGraph** -- stateful agent orchestration with self-refinement loop
- **Claude Sonnet** -- product analysis and listing generation
- **ChromaDB + Sentence Transformers** -- local RAG for platform SEO rules
- **Typer + Rich** -- CLI interface

## Setup

```bash
cd listing-agent
uv pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Generate for all platforms
listing-agent generate "handmade ceramic mug, 12oz, dishwasher safe"

# Specific platforms
listing-agent generate "vintage denim jacket" --platforms shopify,etsy

# JSON output
listing-agent generate "watercolor painting set" --json > listings.json
```

## Development

```bash
# Run tests
uv run pytest tests/unit/ -v

# Run the agent
uv run listing-agent generate "your product description"
```

## How It Works

1. **Analyzer** -- LLM extracts structured product attributes (title, category, features, materials, keywords)
2. **Researcher** -- RAG retrieves platform-specific SEO rules from ChromaDB (title limits, tag rules, keyword guidelines)
3. **Generator** -- LLM generates per-platform listings following retrieved rules
4. **Critic** -- Deterministic heuristic scorer evaluates title length, description quality, tags, platform-specific fields
5. **Refinement loop** -- If any listing scores below 0.7, the generator re-runs with critic feedback (up to 3 iterations)
