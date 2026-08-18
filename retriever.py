import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Generic import patterns across languages — used only to build the reference graph,
# not to guess frameworks or entry-point names.
IMPORT_PATTERNS = [
    re.compile(r'^\s*import\s+([\w\.]+)', re.MULTILINE),          # python, java
    re.compile(r'^\s*from\s+([\w\.]+)\s+import', re.MULTILINE),   # python
    re.compile(r'require\([\'"]([^\'"]+)[\'"]\)'),                 # node
    re.compile(r'^\s*import\s+.*?[\'"]([^\'"]+)[\'"]', re.MULTILINE),  # js/ts
]


def get_retriever(vectorstore: Chroma, k: int = 6, fetch_k: int = 20):
    """Return an MMR-based retriever for diverse, relevant results."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )


def _module_name(source: str) -> str:
    """Turn a file path into a bare module-ish name for import matching, e.g. src/app.py -> app."""
    fname = source.replace("\\", "/").rsplit("/", 1)[-1]
    return fname.rsplit(".", 1)[0]


def get_unimported_files(vectorstore: Chroma) -> set[str]:
    """Return source files that no other file in the repo appears to import — likely entry points."""
    collection = vectorstore.get(include=["metadatas", "documents"])

    # Reconstruct full file contents by source (chunks fragment the text)
    file_texts: dict[str, str] = {}
    for meta, text in zip(collection["metadatas"], collection["documents"]):
        source = meta.get("source", "")
        file_texts[source] = file_texts.get(source, "") + "\n" + text

    all_sources = list(file_texts.keys())
    module_names = {s: _module_name(s) for s in all_sources}

    referenced = set()
    for source, text in file_texts.items():
        for pattern in IMPORT_PATTERNS:
            for match in pattern.findall(text):
                imported_leaf = match.replace(".", "/").rsplit("/", 1)[-1]
                for other_source, other_module in module_names.items():
                    if other_source != source and other_module == imported_leaf:
                        referenced.add(other_source)

    return set(all_sources) - referenced


def get_entry_point_chunks(vectorstore: Chroma) -> list[Document]:
    """Fetch all chunks from files nothing else in the repo imports."""
    unimported = get_unimported_files(vectorstore)
    collection = vectorstore.get(include=["metadatas", "documents"])
    return [
        Document(page_content=text, metadata=meta)
        for meta, text in zip(collection["metadatas"], collection["documents"])
        if meta.get("source") in unimported
    ]

def get_hybrid_retriever(vectorstore: Chroma, k: int = 6, fetch_k: int = 20):
    """Combine semantic (MMR) search with BM25 keyword search, over-fetching so downstream
    filtering (e.g. code-vs-doc prioritization) has real candidates to choose from."""
    vector_retriever = get_retriever(vectorstore, k=fetch_k, fetch_k=fetch_k * 3)

    collection = vectorstore.get(include=["metadatas", "documents"])
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(collection["documents"], collection["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = fetch_k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

def retrieve(vectorstore: Chroma, query: str, k: int = 6, include_entry_points: bool = True) -> list[Document]:
    """Retrieve relevant chunks, prioritizing source code over documentation, with entry points always included."""
    hybrid_retriever = get_hybrid_retriever(vectorstore, k=k, fetch_k=max(20, k * 3))
    all_results = hybrid_retriever.invoke(query)

    code_results = [d for d in all_results if not d.metadata.get("is_doc")]
    doc_results = [d for d in all_results if d.metadata.get("is_doc")]

    results = code_results[:k]
    if len(results) < k:
        results += doc_results[: (k - len(results))]

    if include_entry_points:
        entry_chunks = get_entry_point_chunks(vectorstore)
        seen = {(d.metadata.get("source"), d.page_content) for d in results}
        for chunk in entry_chunks:
            key = (chunk.metadata.get("source"), chunk.page_content)
            if key not in seen:
                results.append(chunk)
                seen.add(key)

    return results


if __name__ == "__main__":
    import sys
    from vectorstore import load_vectorstore, vectorstore_exists
    from chunk import format_location

    repo_url = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "how does this app work"

    if not vectorstore_exists(repo_url):
        print(f"No cached index found for {repo_url}. Run vectorstore.py on it first.")
        sys.exit(1)

    vectorstore = load_vectorstore(repo_url)
    results = retrieve(vectorstore, query)

    print(f"Query: {query}\n")
    for r in results:
        print(format_location(r.metadata))
        print(r.page_content[:200].replace("\n", " "))
        print("---")