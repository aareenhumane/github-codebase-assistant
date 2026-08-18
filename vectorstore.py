import os
import hashlib
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

PERSIST_ROOT = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )


def repo_persist_dir(repo_url: str) -> str:
    """Deterministic local path for a given repo URL's vector store."""
    key = hashlib.sha256(repo_url.encode()).hexdigest()[:16]
    return os.path.join(PERSIST_ROOT, key)


def vectorstore_exists(repo_url: str) -> bool:
    persist_dir = repo_persist_dir(repo_url)
    return os.path.isdir(persist_dir) and len(os.listdir(persist_dir)) > 0


def build_vectorstore(chunks: list[Document], repo_url: str) -> Chroma:
    """Embed chunks and store them in a local Chroma vector store, keyed by repo URL."""
    persist_dir = repo_persist_dir(repo_url)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(repo_url: str) -> Chroma:
    """Load an existing Chroma vector store for a repo URL from disk."""
    persist_dir = repo_persist_dir(repo_url)
    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())

def sync_vectorstore(chunks: list[Document], repo_url: str) -> Chroma:
    """Update a repo's vector store incrementally: only re-embed files whose content changed."""
    persist_dir = repo_persist_dir(repo_url)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())

    existing = vectorstore.get(include=["metadatas"])
    existing_hash_by_source: dict[str, str] = {}
    existing_ids_by_source: dict[str, list[str]] = {}
    for id_, meta in zip(existing["ids"], existing["metadatas"]):
        source = meta.get("source")
        existing_hash_by_source[source] = meta.get("file_hash")
        existing_ids_by_source.setdefault(source, []).append(id_)

    new_hash_by_source: dict[str, str] = {}
    for c in chunks:
        new_hash_by_source[c.metadata["source"]] = c.metadata["file_hash"]

    changed_or_new_sources = {
        source for source, h in new_hash_by_source.items()
        if existing_hash_by_source.get(source) != h
    }
    deleted_sources = set(existing_hash_by_source) - set(new_hash_by_source)

    sources_to_purge = changed_or_new_sources | deleted_sources
    ids_to_delete = [
        id_ for source in sources_to_purge
        for id_ in existing_ids_by_source.get(source, [])
    ]
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    chunks_to_add = [c for c in chunks if c.metadata["source"] in changed_or_new_sources]
    if chunks_to_add:
        vectorstore.add_documents(chunks_to_add)

    print(f"Sync: {len(changed_or_new_sources)} files changed/new, {len(deleted_sources)} removed, "
          f"{len(chunks) - len(chunks_to_add)} chunks unchanged")
    return vectorstore


if __name__ == "__main__":
    import sys
    from ingest import clone_repo, load_repo
    from chunk import chunk_documents, format_location

    repo_url = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "how does this project work"

    if vectorstore_exists(repo_url):
        print(f"Found cached index for {repo_url}, loading...")
        vectorstore = load_vectorstore(repo_url)
    else:
        path = clone_repo(repo_url)
        docs, _ = load_repo(path)
        chunks = chunk_documents(docs)
        print(f"Embedding {len(chunks)} chunks...")
        vectorstore = build_vectorstore(chunks, repo_url)

    print("Testing a similarity search...")
    results = vectorstore.similarity_search(query, k=3)
    for r in results:
        print(format_location(r.metadata))
        print(r.page_content[:150].replace("\n", " "))
        print("---")