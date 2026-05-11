from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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
        preview = (text[:400] + "...") if len(text) > 400 else text
        src = c.get("source")
        idx = c.get("chunk_index")
        header = f"[{src}]" if src is not None else "[chunk]"
        lines.append(f"{header} {preview}")
    return "\n".join(lines)


def answer_with_rag(query: str, retrieved_chunks: List[Dict], session_id: str = "default") -> str:
    context = format_chunks_into_context(retrieved_chunks)
    history = get_session_history(session_id).messages

    messages = [
        ("system", (
            "You are a helpful assistant. Use the conversation history and the provided document chunks to answer. "
            "If the answer is not in the history or chunks, say 'I don't know'. Keep answers short (1-6 sentences)."
        ))
    ]
    for msg in history:
        messages.append((msg.type, msg.content))
    user_content = f"Question: {query}\n\nRelevant Chunks:\n{context}\n\nAnswer:"
    messages.append(("user", user_content))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})

    # Store this turn manually
    get_session_history(session_id).add_user_message(query)
    get_session_history(session_id).add_ai_message(response)

    return response

if __name__ == "__main__":
    print("everyrhing is woorking")