from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


SUPPORTED_PLATFORMS = {"shopify", "amazon", "etsy"}


def build_knowledge_base(knowledge_dir: str, persist_dir: str) -> chromadb.Collection:
    """Index platform markdown docs into ChromaDB. Returns collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name="platform_rules",
        embedding_function=ef,
    )

    if collection.count() > 0:
        return collection  # already indexed

    knowledge_path = Path(knowledge_dir)
    documents, metadatas, ids = [], [], []

    for platform in SUPPORTED_PLATFORMS:
        doc_path = knowledge_path / f"{platform}.md"
        if not doc_path.exists():
            raise FileNotFoundError(f"Missing knowledge doc: {doc_path}")

        content = doc_path.read_text()
        # Split into sections for finer-grained retrieval
        sections = [s.strip() for s in content.split("##") if s.strip()]
        for i, section in enumerate(sections):
            documents.append(section)
            metadatas.append({"platform": platform, "section": i})
            ids.append(f"{platform}_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection
