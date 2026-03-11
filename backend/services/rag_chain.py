"""
rag_chain.py — Retrieval-Augmented Generation pipeline (Mode 1).

Embeds the user's query, retrieves relevant chunks via hybrid search,
builds a grounded prompt with the retrieved context, and calls the LLM
to synthesize an answer with source citations.

Exposes a single public function: run_rag_chain(query, session_id) -> dict.
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.config import settings
from backend.services.hybrid_search import search as hybrid_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation memory — simple in-memory store keyed by session ID.
# Will be replaced by backend/services/memory.py in a later phase.
# ---------------------------------------------------------------------------
_conversation_history: dict[str, list[HumanMessage | AIMessage]] = {}

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an AI Research Assistant. Answer the user's question using ONLY "
    "the context provided below. If the context does not contain enough "
    "information to answer, say so honestly.\n\n"
    "Always cite which source(s) you used. Refer to sources by their file "
    "name and page number when available.\n\n"
    "Context:\n{context}"
)


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a single context string for the prompt.

    Args:
        chunks: List of dicts from hybrid search, each with "content" and
                "metadata" keys.

    Returns:
        A formatted string with each chunk numbered and labelled by source.
    """
    if not chunks:
        return "(No relevant documents were found.)"

    sections: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        page = chunk.get("metadata", {}).get("page", "?")
        content = chunk.get("content", "")
        sections.append(f"[{i}] (Source: {source}, Page: {page})\n{content}")

    return "\n\n".join(sections)


def _get_history(session_id: str) -> list[HumanMessage | AIMessage]:
    """Retrieve conversation history for a session, creating it if needed.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        The list of messages for this session (mutable reference).
    """
    if session_id not in _conversation_history:
        _conversation_history[session_id] = []
    return _conversation_history[session_id]


def _build_messages(
    query: str,
    context: str,
    history: list[HumanMessage | AIMessage],
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """Assemble the full message list for the LLM call.

    Args:
        query: The current user question.
        context: Formatted context string from retrieved chunks.
        history: Previous messages in this conversation session.

    Returns:
        Ordered list of messages: system → history → current user query.
    """
    system_msg = SystemMessage(content=_SYSTEM_PROMPT.format(context=context))
    user_msg = HumanMessage(content=query)
    return [system_msg, *history, user_msg]


async def run_rag_chain(query: str, session_id: str) -> dict[str, Any]:
    """Run the full RAG pipeline: retrieve → prompt → generate.

    Args:
        query: The user's natural-language question.
        session_id: Unique session identifier for conversation memory.

    Returns:
        A dict containing:
            - "answer": The LLM-generated response string.
            - "sources": List of source metadata dicts from retrieval.

    Raises:
        RuntimeError: If retrieval or LLM generation fails.
    """
    logger.info("RAG chain started — session=%s, query='%s'", session_id, query)

    # --- 1. Retrieve relevant chunks ---
    try:
        chunks = await hybrid_search(
            query=query,
            top_k=settings.TOP_K_RETRIEVAL,
        )
    except Exception as exc:
        raise RuntimeError(f"Retrieval failed: {exc}") from exc

    logger.info("Retrieved %d chunks.", len(chunks))

    # --- 2. Build prompt ---
    context = _format_context(chunks)
    history = _get_history(session_id)
    messages = _build_messages(query, context, history)

    # --- 3. Call the LLM ---
    try:
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        response = await llm.ainvoke(messages)
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc

    answer = response.content
    logger.info("LLM response received (%d chars).", len(answer))

    # --- 4. Update conversation memory ---
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=answer))

    # --- 5. Build sources list ---
    sources = [
        {
            "source": chunk.get("metadata", {}).get("source", "unknown"),
            "page": chunk.get("metadata", {}).get("page", "?"),
            "content_preview": chunk.get("content", "")[:200],
        }
        for chunk in chunks
    ]

    return {"answer": answer, "sources": sources}
