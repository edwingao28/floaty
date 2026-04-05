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
+--------------+    quality < 0.8?
| Critic Node  | ------------------> loop back (max 3x)
| RulesScorer  |    (selective: only re-generates
| + LLMJudge   |     below-threshold listings)
| + ScoringRubric|
+--------------+
    | quality >= 0.8 (or converged)
    v
+----------------+
| Approval Node  |  <-- human-in-the-loop via interrupt()
| (optional)     |      only with --publish
+----------------+
    |
    v
+----------------+
| Publisher Node |  <-- Shopify GraphQL / Amazon SP-API / Etsy v3
| (optional)     |
+----------------+
    |
    v
  Final Listings
```

## Tech Stack

- **LangGraph** — stateful agent orchestration with self-refinement loop and `interrupt()` for human approval
- **Claude Sonnet** — product analysis and listing generation (with Haiku fallback)
- **Claude Haiku** — LLM judge for subjective quality scoring
- **ChromaDB + Sentence Transformers** — local RAG for platform SEO rules
- **Typer + Rich** — CLI interface

## Setup

```bash
cd listing-agent
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### 1. Ingest knowledge base (first run)

```bash
uv run listing-agent ingest

# Custom knowledge base path
uv run listing-agent ingest --path /path/to/docs
```

### 2. Generate listings

```bash
# All platforms (shopify, amazon, etsy)
uv run listing-agent generate "handmade ceramic mug, 12oz, dishwasher safe"

# Specific platforms
uv run listing-agent generate "vintage denim jacket" --platforms shopify,etsy

# JSON output
uv run listing-agent generate "watercolor painting set" --json > listings.json

# Full pipeline: generate + human approval + publish
uv run listing-agent generate "handmade mug" --publish
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run the agent
uv run listing-agent generate "your product description"
```

## How It Works

1. **Analyzer** — LLM extracts structured product attributes (title, category, features, materials, keywords)
2. **Researcher** — RAG retrieves platform-specific SEO rules from ChromaDB (title limits, tag rules, keyword guidelines)
3. **Generator** — LLM generates per-platform listings following retrieved rules; on refinement rounds, only re-generates below-threshold listings
4. **Critic** — Composite scorer: `RulesScorer` (6 deterministic dimensions, weight 0.6) + `LLMJudge` (4 subjective dimensions via Haiku, weight 0.4). Detects convergence and oscillation to avoid infinite loops
5. **Refinement loop** — If any listing scores below 0.8, the generator re-runs with critic feedback (up to 3 iterations)
6. **Approval** *(with `--publish`)* — Graph pauses via `interrupt()` for human review; resume with approve/reject decision
7. **Publisher** *(with `--publish`)* — Posts approved listings to Shopify GraphQL Admin API, Amazon SP-API, or Etsy Open API v3

## Scoring

| Dimension | Type | Weight |
|-----------|------|--------|
| Title length compliance | Rules | 0.60 combined |
| Bullet compliance | Rules | |
| Keyword presence | Rules | |
| Readability | Rules | |
| Char limit compliance | Rules | |
| HTML validity | Rules | |
| Persuasiveness | LLM (Haiku) | 0.40 combined |
| Brand voice | LLM (Haiku) | |
| USP clarity | LLM (Haiku) | |
| Competitive positioning | LLM (Haiku) | |

Quality threshold: **0.8** (configurable via `QUALITY_THRESHOLD` env var)

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Primary LLM |
| `ANTHROPIC_FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Fallback + LLM judge |
| `CHROMA_DB_PATH` | `.chroma` | ChromaDB storage path |
| `QUALITY_THRESHOLD` | `0.8` | Min score to pass critic |
| `RULES_WEIGHT` | `0.6` | Weight for rules scorer |
| `LLM_WEIGHT` | `0.4` | Weight for LLM judge |
| `SHOPIFY_STORE_URL` | — | Shopify store URL |
| `SHOPIFY_ACCESS_TOKEN` | — | Shopify Admin API token |
| `AMAZON_REFRESH_TOKEN` | — | Amazon SP-API refresh token |
| `ETSY_API_KEY` | — | Etsy Open API v3 key |
