import typer
from rich import print as rprint

app = typer.Typer(name="listing-agent", help="E-commerce listing optimization agent")


@app.command()
def generate(product: str) -> None:
    """Generate optimized listings for a product."""
    rprint(f"Generating listings for: {product}")


@app.command()
def ingest() -> None:
    """Ingest platform knowledge base documents."""
    rprint("Ingesting knowledge base...")


if __name__ == "__main__":
    app()
