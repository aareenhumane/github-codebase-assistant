import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from chunk import format_location

load_dotenv()

LLM_MODEL = "gemma-4-31b-it"

SYSTEM_PROMPT = """You are a codebase assistant. Answer the user's question about the repository \
using ONLY the provided code context below.

Rules:
- Base your answer strictly on the given context. If the context doesn't contain enough \
information, say so explicitly instead of guessing.
- Every claim about specific code (functions, classes, logic) must include an inline citation \
in the form [source] right after the claim, using the exact source labels given in the context.
- Be concise and technical. Prefer short explanations over long ones.
- If asked about architecture, synthesize across multiple files/chunks rather than listing them.
- Use the conversation history only to understand what the user is referring to (e.g. "that function", \
"the one you mentioned"). Never answer from memory of the prior conversation alone — always ground \
new claims in the retrieved context below.

Repository file structure (for context on how files relate; not all files below have retrieved content):
{file_tree}

Retrieved code context:
{context}
"""
REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up \
into a standalone question that includes any necessary context from the history. \
If the follow-up question is already standalone (doesn't depend on prior turns), return it unchanged. \
Output ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question: {question}

Standalone question:"""


def rewrite_query(question: str, history: list[dict], max_turns: int = 3) -> str:
    """Rewrite a follow-up question into a standalone query for retrieval, using recent history."""
    if not history:
        return question

    recent = history[-(max_turns * 2):]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    chain = prompt | llm
    result = chain.invoke({"history_text": history_text, "question": question})
    rewritten = _extract_text(result.content).strip()
    return rewritten or question

def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2,
    )


def format_context(docs: list[Document]) -> str:
    """Format retrieved chunks into labeled context blocks for the prompt."""
    blocks = []
    for doc in docs:
        label = format_location(doc.metadata)
        blocks.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def format_history(history: list[dict], max_turns: int = 3) -> list:
    """Convert recent chat history into LangChain message tuples for the prompt."""
    recent = history[-(max_turns * 2):]  # last N user+assistant pairs
    messages = []
    for msg in recent:
        role = "human" if msg["role"] == "user" else "ai"
        messages.append((role, msg["content"]))
    return messages


def answer_question(question: str, docs: list[Document], file_tree: str = "", history: list[dict] = None) -> str:
    """Answer a question using retrieved chunks + repo structure + recent history, with inline citations."""
    llm = get_llm()
    history_messages = format_history(history or [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        *history_messages,
        ("human", "{question}"),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "file_tree": file_tree or "(not provided)",
        "context": format_context(docs),
        "question": question,
    })
    return _extract_text(result.content)


def _extract_text(content) -> str:
    """Handle both plain string content and Gemma's thinking-mode block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()
    return str(content)


if __name__ == "__main__":
    import sys
    from vectorstore import load_vectorstore, vectorstore_exists
    from retriever import retrieve

    repo_url = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "how does this app work"

    if not vectorstore_exists(repo_url):
        print(f"No cached index found for {repo_url}. Run vectorstore.py on it first.")
        sys.exit(1)

    vectorstore = load_vectorstore(repo_url)
    docs = retrieve(vectorstore, question)

    print(f"Question: {question}\n")
    answer = answer_question(question, docs)
    print(answer)
    print("\n--- Sources ---")
    for d in docs:
        print(format_location(d.metadata))