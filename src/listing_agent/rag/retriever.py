from listing_agent.rag.loader import SUPPORTED_PLATFORMS, build_knowledge_base


class PlatformRetriever:
    def __init__(self, knowledge_dir: str, persist_dir: str = ".chroma"):
        self._collection = build_knowledge_base(knowledge_dir, persist_dir)

    def get_rules(self, platform: str, product_description: str, n_results: int = 3) -> str:
        """Return relevant platform rules for the given product."""
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Unknown platform '{platform}'. Supported: {SUPPORTED_PLATFORMS}")

        results = self._collection.query(
            query_texts=[f"{platform} listing rules for {product_description}"],
            n_results=n_results,
            where={"platform": platform},
        )
        sections = results["documents"][0]
        return "\n\n".join(sections)
