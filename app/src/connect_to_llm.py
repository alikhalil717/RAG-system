from typing import List, Dict
import threading
import re

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import *
from utils import suppress_weaviate_logs, print_object_properties

llm = ChatOpenAI(
    model=LLM_NAME,
    api_key=LLM_API_KEY,
    base_url=LLM_URL,
    temperature=LLM_TEMPERATURE,
)

store: Dict[str, ChatMessageHistory] = {}
_store_lock = threading.Lock()


def connect_to_llm() -> ChatOpenAI:
    return llm


def get_session_history(session_id: str) -> ChatMessageHistory:
    with _store_lock:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


def delete_session(session_id: str) -> None:
    with _store_lock:
        store.pop(session_id, None)


def list_sessions() -> List[str]:
    with _store_lock:
        return list(store.keys())


def format_chunks_into_context(chunks: List[Dict]) -> str:

    lines = []
    for c in chunks:
        text = c.get("chunk") or c.get("text") or ""
        src = c.get("source")
        book = c.get("book", "")
        chapter = c.get("chapter")
        page = c.get("page")
        code_lang = c.get("code_language", "")

        parts = []
        if book:
            parts.append(f"book={book}")
        if chapter:
            parts.append(f"chapter={chapter}")
        if page is not None:
            parts.append(f"page={page}")
        header = f"[{src}]" if src is not None else "[chunk]"
        if parts:
            header += f" ({', '.join(parts)})"

        if "[CODE]" in text:
            text = text.replace("[CODE]", f"\n```{code_lang}\n").replace("[/CODE]", "\n```\n")

        lines.append(f"{header}\n{text}")
    return "\n---\n".join(lines)


_REWRITE_SYSTEM_PROMPT = (
    "You extract a SHORT keyword search query from the user's input.\n\n"
    "Rules:\n"
    "- Output ONLY the rewritten query. No explanation, no quotes, no extra text.\n"
    "- Keep it between 3 and 30 words.\n"
    "- CRITICAL: ONLY use terms that explicitly appear in the user's input. "
    "Do NOT invent, guess, or add any terms that aren't there.\n"
    "- If the input contains code (File paths, function names, package names, imports), "
    "PRESERVE them verbatim — do not paraphrase or remove them.\n"
    "- If the input is a natural language question without code, extract just the 3-8 "
    "most important keywords from the user's own words.\n"
    "- If the input is already short and keyword-like, return it unchanged."
)


def rewrite_query_for_retrieval(query: str) -> str:


    _CODE_HEURISTICS = re.compile(
        r"(cmd/api/|\.go\b|package\s+\w+|import\s*\(|\bfunc\s+\w+\s*\()"
    )
    if _CODE_HEURISTICS.search(query):
        print(f"[query rewrite] SKIPPED (code query) — using raw query.")
        return query[:300]

    try:
        messages = [
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        rewriter = llm.bind(temperature=0.0, max_tokens=120)
        response = rewriter.invoke(messages)
        rewritten = response.content.strip().strip('"').strip("'")
        if rewritten:
            print(f"[query rewrite] ({len(rewritten)} chars) '{rewritten}'")
            return rewritten[:200]
    except Exception as exc:
        print(f"[query rewrite] failed ({exc}), using original query.")
    return query[:300]


_DECOMPOSE_PROMPT = (
    "You generate 3-4 specific search queries for a programming textbook knowledge base.\n\n"
    "Rules:\n"
    "- Output ONLY the queries, one per line. No numbering, no explanation, no quotes.\n"
    "- Each query must be 3-15 words.\n"
    "- CRITICAL: ALL terms in your output must appear in the user's input. "
    "Do NOT invent, guess, or add any terms that aren't there.\n"
    "- If the input contains code identifiers (file paths, function names, "
    "import paths), preserve them verbatim and spread them across different queries.\n"
    "- If the input is a natural language question, each query should use a different "
    "subset of the user's own words to target different angles.\n"
    "- Cover different conceptual angles using only the user's actual terms."
)


def decompose_query(query: str) -> List[str]:

    try:
        messages = [
            SystemMessage(content=_DECOMPOSE_PROMPT),
            HumanMessage(content=query),
        ]
        decomposer = llm.bind(temperature=0.0, max_tokens=300)
        response = decomposer.invoke(messages)
        raw = response.content.strip()

        sub_queries = [line.strip() for line in raw.split("\n") if line.strip()]
        sub_queries = [re.sub(r"^\d+[\.\)]\s*", "", q) for q in sub_queries]
        sub_queries = [q.strip('"\'') for q in sub_queries]
        sub_queries = [q for q in sub_queries if len(q) > 5]

        if sub_queries:
            print(f"[query decompose] '{query}' → {len(sub_queries)} sub-queries:")
            for sq in sub_queries:
                print(f"    • {sq}")
            return sub_queries[:5]
    except Exception as exc:
        print(f"[query decompose] failed ({exc}), using single query.")

    return [query[:300]]


def answer_with_rag(query: str, retrieved_chunks: List[Dict], session_id: str = "default") -> str:
    context = format_chunks_into_context(retrieved_chunks)
    history = get_session_history(session_id).messages

    messages = [
        SystemMessage(content=(
            "You are a programming assistant with access to a textbook knowledge base. "
            "Use the provided document chunks to answer the user's question.\n\n"
            "Rules:\n"
            "- If the user asks for code, show the COMPLETE code snippet — do not abbreviate.\n"
            "- If the relevant code is split across multiple chunks, combine them into one "
            "coherent answer.\n"
            "- Use markdown code blocks with the correct language for code.\n"
            "- Cite the source (book, chapter, page) when available.\n"
            "- If no relevant information is found in the chunks, say so honestly."
        ))
    ]
    for msg in history:
        if msg.type == "human":
            messages.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            messages.append(AIMessage(content=msg.content))
        else:
            messages.append(msg)

    user_content = f"Question: {query}\n\nRelevant data:\n{context}\n\nAnswer:"
    messages.append(HumanMessage(content=user_content))

    response = llm.invoke(messages)
    answer = response.content

    get_session_history(session_id).add_user_message(query)
    get_session_history(session_id).add_ai_message(answer)

    return answer

if __name__ == "__main__":
    print("everyrhing is woorking")