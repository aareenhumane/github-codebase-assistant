import os
import json
import subprocess
import tempfile
from langchain_core.documents import Document
import hashlib
from urllib.parse import urlparse, urlunparse


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rs", ".php", ".md", ".rst"
}
NOTEBOOK_EXTENSION = ".ipynb"
SKIP_DIRS = {".git", "node_modules", "venv", "__pycache__", "dist", "build", ".next", ".ipynb_checkpoints"}
DOC_EXTENSIONS = {".md", ".rst"}                      # <-- add this


def _is_doc_file(rel_path: str, ext: str) -> bool:    # <-- add this function
    """Heuristic: docs live in doc-like extensions, optionally under a docs/ folder."""
    return ext in DOC_EXTENSIONS

def file_hash(text: str) -> str:
    """Content hash for a file's text, used to detect changes between indexing runs."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clone_repo(github_url: str, token: str | None = None) -> str:
    """Clone a GitHub repo to a temp dir and return the local path.
    If a token is provided, it's injected into the clone URL for private repo access."""
    tmp_dir = tempfile.mkdtemp(prefix="repo_")
    clone_url = _inject_token(github_url, token) if token else github_url

    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, tmp_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Strip any token from the error message so it's never displayed/logged
        safe_error = result.stderr.replace(token, "***") if token else result.stderr
        raise RuntimeError(f"Failed to clone repo: {safe_error.strip()}")

    return tmp_dir

def _inject_token(github_url: str, token: str) -> str:
    """Embed a GitHub token into an HTTPS clone URL for authenticated access."""
    parsed = urlparse(github_url)
    if parsed.scheme != "https" or "github.com" not in parsed.netloc:
        return github_url  # only supports https github.com URLs; leave others untouched
    authed_netloc = f"{token}@{parsed.netloc}"
    return urlunparse(parsed._replace(netloc=authed_netloc))

def _notebook_to_text(fpath: str) -> str:
    """Flatten a Jupyter notebook's code + markdown cells into plain text."""
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        nb = json.load(f)

    parts = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        cell_type = cell.get("cell_type", "code")
        if cell_type == "markdown":
            parts.append(f"# [markdown cell]\n{source}")
        else:
            parts.append(f"# [code cell]\n{source}")
    return "\n\n".join(parts)


MAX_FILE_SIZE_BYTES = 500_000       # skip individual files larger than ~500KB (likely generated/data files)
MAX_TOTAL_FILES = 500               # hard cap on files indexed per repo


def load_repo(repo_path: str) -> list[Document]:
    """Walk a local repo directory and load code/text/notebook files as Documents,
    respecting size limits to avoid indexing huge or generated files."""
    docs = []
    skipped_large = []
    truncated = False

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if len(docs) >= MAX_TOTAL_FILES:
                truncated = True
                break

            ext = os.path.splitext(fname)[1]
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_path).replace("\\", "/")

            if ext not in CODE_EXTENSIONS and ext != NOTEBOOK_EXTENSION:
                continue

            try:
                file_size = os.path.getsize(fpath)
                if file_size > MAX_FILE_SIZE_BYTES:
                    skipped_large.append(rel_path)
                    continue

                if ext == NOTEBOOK_EXTENSION:
                    text = _notebook_to_text(fpath)
                else:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
            except Exception:
                continue

            if not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": rel_path,
                    "file_hash": file_hash(text),
                    "is_doc": _is_doc_file(rel_path, ext),
                }
            ))
        if truncated:
            break

    return docs, {"skipped_large": skipped_large, "truncated": truncated}

def build_file_tree(repo_path: str) -> str:
    """Generate a simple indented file tree string for repo structure context."""
    lines = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel_root = os.path.relpath(root, repo_path)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        indent = "  " * depth
        if rel_root != ".":
            lines.append(f"{indent}{os.path.basename(root)}/")
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1]
            if ext in CODE_EXTENSIONS or ext == NOTEBOOK_EXTENSION:
                lines.append(f"{indent}  {fname}")
    return "\n".join(lines)

def get_default_branch(repo_path: str) -> str:
    """Return the current branch name of a cloned repo (e.g. 'main' or 'master')."""
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True
    )
    branch = result.stdout.strip()
    return branch if branch else "main"

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    if path.startswith("http"):
        path = clone_repo(path)
    docs, _ = load_repo(path)
    print(f"Loaded {len(docs)} files")
    for d in docs:
        print(d.metadata["source"])