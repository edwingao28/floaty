import json
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from listing_agent.graph import build_graph

app = typer.Typer(name="listing-agent", help="E-commerce listing optimization agent")
console = Console()


@app.command()
def generate(
    product: str,
    platforms: Annotated[
        str,
        typer.Option("--platforms", help="Comma-separated platforms"),
    ] = "shopify,amazon,etsy",
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Output raw JSON"),
    ] = False,
) -> None:
    """Generate optimized listings for a product."""
    target_platforms = [p.strip() for p in platforms.split(",")]
    initial_state = {
        "raw_product_data": {"description": product},
        "target_platforms": target_platforms,
    }

    result = build_graph().invoke(initial_state)

    errors = result.get("errors", [])
    if errors:
        for err in errors:
            rprint(f"[red]Error:[/red] {err}")
        raise typer.Exit(1)

    listings = result.get("listings", [])

    if output_json:
        rprint(json.dumps([l.model_dump() for l in listings], indent=2))
        return

    for listing in listings:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("Title", listing.title)
        table.add_row("Tags", ", ".join(listing.tags))
        table.add_row("Description", listing.description[:200])

        score = listing.score
        if score is None:
            score_str = "N/A"
        elif score >= 0.7:
            score_str = f"[green]{score:.2f}[/green]"
        elif score >= 0.5:
            score_str = f"[yellow]{score:.2f}[/yellow]"
        else:
            score_str = f"[red]{score:.2f}[/red]"
        table.add_row("Quality score", score_str)

        console.print(Panel(table, title=listing.platform.upper()))

    refinement_count = result.get("refinement_count", 0)
    rprint(f"\nRefinements: {refinement_count}")


@app.command()
def ingest() -> None:
    """Ingest platform knowledge base documents."""
    rprint("Ingesting knowledge base...")


if __name__ == "__main__":
    app()
