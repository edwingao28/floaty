import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


SUPPORTED_PLATFORMS = {"shopify", "amazon", "etsy"}


def _compute_knowledge_hash(knowledge_path: Path) -> str:
    """Hash all platform doc contents to detect changes."""
    h = hashlib.sha256()
    for platform in sorted(SUPPORTED_PLATFORMS):
        doc_path = knowledge_path / f"{platform}.md"
        if not doc_path.exists():
            raise FileNotFoundError(f"Missing knowledge doc: {doc_path}")
        h.update(doc_path.read_bytes())
    return h.hexdigest()


def build_knowledge_base(knowledge_dir: str, persist_dir: str) -> chromadb.Collection:
    """Index platform markdown docs into ChromaDB. Returns collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    knowledge_path = Path(knowledge_dir)
    current_hash = _compute_knowledge_hash(knowledge_path)

    collection = client.get_or_create_collection(
        name="platform_rules",
        embedding_function=ef,
    )

    stored_hash = (collection.metadata or {}).get("knowledge_hash", "")
    if collection.count() > 0 and stored_hash == current_hash:
        return collection  # already indexed, no changes

    # Hash changed or collection empty — delete and re-index
    client.delete_collection("platform_rules")
    collection = client.create_collection(
        name="platform_rules",
        embedding_function=ef,
        metadata={"knowledge_hash": current_hash},
    )

    documents, metadatas, ids = [], [], []

    for platform in SUPPORTED_PLATFORMS:
        doc_path = knowledge_path / f"{platform}.md"
        if not doc_path.exists():
            raise FileNotFoundError(f"Missing knowledge doc: {doc_path}")

        content = doc_path.read_text()
        # Split on ## and skip sections[0] (H1 preamble before first ##)
        raw_sections = content.split("##")
        sections = [s.strip() for s in raw_sections[1:] if s.strip()]
        for i, section in enumerate(sections):
            documents.append(section)
            metadatas.append({"platform": platform, "section": i})
            ids.append(f"{platform}_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection
