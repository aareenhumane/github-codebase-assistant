from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
import ast

# Map file extensions to LangChain's Language enum for syntax-aware splitting
EXT_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".md": Language.MARKDOWN,
    ".rst": Language.RST,
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def _chunk_notebook_cells(doc: Document) -> list[Document]:
    """Chunk a flattened notebook by cell, citing cell index instead of line numbers."""
    source = doc.metadata["source"]
    file_hash = doc.metadata["file_hash"]
    is_doc = doc.metadata["is_doc"]                    # <-- added
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    raw_cells = doc.page_content.split("\n\n# [")
    chunks = []
    for i, raw in enumerate(raw_cells):
        cell_text = raw if i == 0 else "# [" + raw
        if not cell_text.strip():
            continue
        pieces = splitter.split_text(cell_text)
        for piece in pieces:
            chunks.append(Document(
                page_content=piece,
                metadata={"source": source, "cell": i + 1, "file_hash": file_hash, "is_doc": is_doc}  # <-- added
            ))
    return chunks

def _get_splitter(ext: str) -> RecursiveCharacterTextSplitter:
    language = EXT_TO_LANGUAGE.get(ext)
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    # Fallback for unknown/plain text files
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _find_line_range(full_text: str, chunk_text: str, search_from: int) -> tuple[int, int, int]:
    """Locate a chunk within the original text and return (start_line, end_line, next_search_from)."""
    idx = full_text.find(chunk_text, search_from)
    if idx == -1:
        idx = full_text.find(chunk_text)  # fallback: search from the start
    if idx == -1:
        return (1, 1, search_from)  # couldn't locate; shouldn't normally happen

    start_line = full_text.count("\n", 0, idx) + 1
    end_line = start_line + chunk_text.count("\n")
    return (start_line, end_line, idx + 1)  # advance slightly to avoid re-matching same spot

def _chunk_python_ast(doc: Document) -> list[Document] | None:
    """Chunk a Python file by top-level function/class boundaries using the AST.
    Returns None if the file fails to parse (falls back to generic splitting)."""
    source = doc.page_content
    file_hash = doc.metadata["file_hash"]
    is_doc = doc.metadata["is_doc"]                              # <-- ADD THIS LINE
    file_path = doc.metadata["source"]

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines()
    chunks = []
    covered_lines = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line)
            chunk_text = "\n".join(lines[start_line - 1:end_line])

            # Oversized functions/classes still get sub-split so we don't blow context limits
            if len(chunk_text) > CHUNK_SIZE * 2:
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                for piece in sub_splitter.split_text(chunk_text):
                    chunks.append(Document(
                        page_content=piece,
                        metadata={"source": file_path, "start_line": start_line,
                                  "end_line": end_line, "file_hash": file_hash,
                                  "is_doc": is_doc}                # <-- ADD THIS KEY (spot #1)
                    ))
            else:
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"source": file_path, "start_line": start_line,
                              "end_line": end_line, "file_hash": file_hash,
                              "is_doc": is_doc}                    # <-- ADD THIS KEY (spot #2)
                ))
            covered_lines.update(range(start_line, end_line + 1))

    # Capture top-level code NOT inside any function/class (imports, constants, script logic)
    leftover_lines = [
        (i + 1, line) for i, line in enumerate(lines)
        if (i + 1) not in covered_lines and line.strip()
    ]
    if leftover_lines:
        leftover_text = "\n".join(line for _, line in leftover_lines)
        start_line = leftover_lines[0][0]
        end_line = leftover_lines[-1][0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for piece in splitter.split_text(leftover_text):
            chunks.append(Document(
                page_content=piece,
                metadata={"source": file_path, "start_line": start_line,
                          "end_line": end_line, "file_hash": file_hash,
                          "is_doc": is_doc}                        # <-- ADD THIS KEY (spot #3)
            ))

    return chunks if chunks else None

def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        source = doc.metadata["source"]
        file_hash = doc.metadata["file_hash"]
        is_doc = doc.metadata["is_doc"]                # <-- added

        if source.endswith(".ipynb"):
            chunks.extend(_chunk_notebook_cells(doc))
            continue

        if source.endswith(".py"):
            ast_chunks = _chunk_python_ast(doc)
            if ast_chunks is not None:
                chunks.extend(ast_chunks)
                continue

        ext = "." + source.rsplit(".", 1)[-1] if "." in source else ""
        splitter = _get_splitter(ext)

        pieces = splitter.split_text(doc.page_content)
        search_from = 0
        for piece in pieces:
            start_line, end_line, search_from = _find_line_range(doc.page_content, piece, search_from)
            chunks.append(Document(
                page_content=piece,
                metadata={
                    "source": source,
                    "start_line": start_line,
                    "end_line": end_line,
                    "file_hash": file_hash,
                    "is_doc": is_doc,                    # <-- added
                }
            ))
    return chunks


def format_location(meta: dict) -> str:
    """Human-readable location string for a chunk, handling both line-based and cell-based citations."""
    if "cell" in meta:
        return f"{meta['source']} (cell {meta['cell']})"
    return f"{meta['source']} (lines {meta['start_line']}-{meta['end_line']})"

def format_github_link(repo_url: str, branch: str, meta: dict) -> str:
    """Build a GitHub URL pointing to the exact file (and line range, if available) for a chunk."""
    base = repo_url.rstrip("/")
    if base.endswith(".git"):
        base = base[:-4]
    source = meta["source"].replace("\\", "/")

    if "cell" in meta:
        # GitHub doesn't support deep-linking into notebook cells; link to the file itself
        return f"{base}/blob/{branch}/{source}"

    start = meta.get("start_line")
    end = meta.get("end_line")
    if start and end:
        if start == end:
            return f"{base}/blob/{branch}/{source}#L{start}"
        return f"{base}/blob/{branch}/{source}#L{start}-L{end}"
    return f"{base}/blob/{branch}/{source}"

if __name__ == "__main__":
    import sys
    from ingest import clone_repo, load_repo

    path = sys.argv[1]
    if path.startswith("http"):
        path = clone_repo(path)
    docs, _ = load_repo(path)
    chunks = chunk_documents(docs)
    print(f"{len(docs)} files -> {len(chunks)} chunks")
    for c in chunks[:5]:
        print(format_location(c.metadata))
        print(c.page_content[:150].replace("\n", " "))
        print("---")