from typing import List, Dict

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


def connect_to_llm() -> ChatOpenAI:
    return llm


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def format_chunks_into_context(chunks: List[Dict]) -> str:
    lines = []
    for c in chunks:
        text = c.get("chunk") or c.get("text") or ""
        src = c.get("source")
        idx = c.get("chunk_index")
        book = c.get("book", "")
        chapter = c.get("chapter")
        page = c.get("page")
        code_lang = c.get("code_language", "")

        # Build metadata header
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

        # Render code blocks as markdown fenced code
        if "[CODE]" in text:
            text = text.replace("[CODE]", f"\n```{code_lang}\n").replace("[/CODE]", "\n```\n")

        preview = (text[:500] + "...") if len(text) > 500 else text
        lines.append(f"{header}\n{preview}")
    return "\n---\n".join(lines)


_REWRITE_SYSTEM_PROMPT = (
    "You are a search query optimizer for a technical PDF knowledge base. "
    "Rewrite the user's question into a concise, keyword-rich search query "
    "that will retrieve the most relevant chunks from a hybrid vector+BM25 search. "
    "Rules:\n"
    "- Output ONLY the rewritten query. No explanation, no quotes, no extra text.\n"
    "- Keep it under 30 words.\n"
    "- Expand vague phrasing into precise technical terms and synonyms.\n"
    "- If the question is about code, include the language name and relevant keywords "
    "(e.g. function name, algorithm, API).\n"
    "- If the query is already precise and technical, return it unchanged."
)


def rewrite_query_for_retrieval(query: str) -> str:
   
    try:
        messages = [
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        # Use a throwaway low-temperature call — deterministic query expansion
        rewriter = llm.bind(temperature=0.0, max_tokens=60)
        response = rewriter.invoke(messages)
        rewritten = response.content.strip().strip('"').strip("'")
        if rewritten:
            print(f"[query rewrite] '{query}'  →  '{rewritten}'")
            return rewritten
    except Exception as exc:
        print(f"[query rewrite] failed ({exc}), using original query.")
    return query


def answer_with_rag(query: str, retrieved_chunks: List[Dict], session_id: str = "default") -> str:
    context = format_chunks_into_context(retrieved_chunks)
    history = get_session_history(session_id).messages

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Use the conversation history and the provided document chunks to answer. "
            "If the answer is not in the history or chunks, say 'I don't know'."
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

    # Store this turn manually
    get_session_history(session_id).add_user_message(query)
    get_session_history(session_id).add_ai_message(answer)

    return answer

if __name__ == "__main__":
    print("everyrhing is woorking")