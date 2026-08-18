import streamlit as st
from ingest import clone_repo, load_repo, build_file_tree, get_default_branch, MAX_TOTAL_FILES
from chunk import chunk_documents, format_location, format_github_link
from vectorstore import build_vectorstore, load_vectorstore, vectorstore_exists, sync_vectorstore
from retriever import retrieve
from qa import answer_question, rewrite_query

_LANG_BY_EXT = {
    ".py": "python", ".ipynb": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go",
    ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".cs": "csharp", ".rs": "rust", ".php": "php", ".md": "markdown", ".rst": "text",
}

def _guess_lang(source: str) -> str:
    ext = "." + source.rsplit(".", 1)[-1] if "." in source else ""
    return _LANG_BY_EXT.get(ext, "text")

st.set_page_config(page_title="GitHub Codebase Assistant", layout="wide")
st.title("GitHub Codebase Assistant")

# --- Repo loading ---
repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repo")

with st.expander("Private repo? Add a GitHub token"):
    github_token = st.text_input(
        "Personal access token",
        type="password",
        help="Only needed for private repos. Generate one at github.com/settings/tokens with 'repo' scope.",
    )

with st.sidebar:
    st.header("Repository Structure")
    if "file_tree" in st.session_state:
        st.caption(st.session_state.get("repo_url", ""))
        st.code(st.session_state["file_tree"], language="text")
    else:
        st.caption("Load a repo to see its file structure here.")

if st.button("Load Repo") and repo_url:
    # Basic URL sanity check before attempting anything expensive
    if not repo_url.startswith(("https://github.com/", "http://github.com/")):
        st.error("Please enter a valid GitHub URL, e.g. https://github.com/user/repo")
        st.stop()

    try:
        with st.spinner("Cloning repo..."):
            repo_path = clone_repo(repo_url, token=github_token or None)
    except RuntimeError as e:
        st.error(f"Couldn't clone this repo: {e}")
        st.info("Check that the URL is correct and, if it's private, that your token has 'repo' scope.")
        st.stop()

    try:
        with st.spinner("Loading files..."):
            docs, load_info = load_repo(repo_path)
    except Exception as e:
        st.error(f"Something went wrong reading the repo's files: {e}")
        st.stop()

    if not docs:
        st.warning(
            "No indexable code files were found in this repo. "
            "It may be empty, use unsupported file types, or contain only binary/data files."
        )
        st.stop()

    if load_info["truncated"]:
        st.warning(
            f"This repo has more than {MAX_TOTAL_FILES} indexable files. "
            f"Only the first {MAX_TOTAL_FILES} were loaded — results may be incomplete."
        )
    if load_info["skipped_large"]:
        with st.expander(f"{len(load_info['skipped_large'])} large files were skipped"):
            for f in load_info["skipped_large"]:
                st.text(f)

    try:
        with st.spinner(f"Chunking {len(docs)} files..."):
            chunks = chunk_documents(docs)
    except Exception as e:
        st.error(f"Something went wrong splitting the code into chunks: {e}")
        st.stop()

    try:
        with st.spinner("Syncing index (only changed files are re-embedded)..."):
            vectorstore = sync_vectorstore(chunks, repo_url)
    except Exception as e:
        st.error(f"Something went wrong building the search index: {e}")
        st.info("If this keeps happening, try a smaller repo or check your GPU/CUDA setup.")
        st.stop()

    branch = get_default_branch(repo_path)
    file_tree = build_file_tree(repo_path)
    st.session_state["file_tree"] = file_tree
    st.session_state["branch"] = branch
    st.session_state["vectorstore"] = vectorstore
    st.session_state["repo_url"] = repo_url
    st.session_state["messages"] = []
    st.success(f"Indexed {len(docs)} files -> {len(chunks)} chunks from {repo_url}")

# --- Chat interface ---
if "vectorstore" in st.session_state:
    st.divider()
    st.caption(f"Chatting with: {st.session_state['repo_url']}")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("source_docs"):
                with st.expander(f"Sources ({len(msg['source_docs'])})"):
                    for doc in msg["source_docs"]:
                        link = format_github_link(
                            st.session_state["repo_url"],
                            st.session_state["branch"],
                            doc.metadata,
                        )
                        st.markdown(f"**[{format_location(doc.metadata)}]({link})**")
                        lang = _guess_lang(doc.metadata.get("source", ""))
                        st.code(doc.page_content, language=lang)

    question = st.chat_input("Ask a question about the codebase...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                history = st.session_state["messages"][:-1]

                with st.spinner("Understanding question..."):
                    search_query = rewrite_query(question, history)

                with st.spinner("Retrieving relevant code..."):
                    docs = retrieve(st.session_state["vectorstore"], search_query)

                with st.spinner("Generating answer..."):
                    answer = answer_question(question, docs, st.session_state.get("file_tree", ""), history=history)

                st.markdown(answer)

                with st.expander(f"Sources ({len(docs)})"):
                    if search_query != question:
                        st.caption(f"Searched for: \"{search_query}\"")
                    for doc in docs:
                        link = format_github_link(st.session_state["repo_url"], st.session_state["branch"], doc.metadata)
                        st.markdown(f"**[{format_location(doc.metadata)}]({link})**")
                        lang = _guess_lang(doc.metadata.get("source", ""))
                        st.code(doc.page_content, language=lang)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "source_docs": docs,
                })
            except Exception as e:
                error_msg = f"Sorry, something went wrong answering that: {e}"
                st.error(error_msg)
                st.session_state["messages"].append({"role": "assistant", "content": error_msg, "source_docs": []})
else:
    st.info("Load a GitHub repo above to start chatting.")