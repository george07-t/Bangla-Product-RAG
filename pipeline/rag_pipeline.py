"""
RAG Pipeline — orchestrates retrieval + LLM generation.
"""
import os
import time
import re
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retriever.faiss_retriever import FAISSRetriever
from retriever.query_rewriter import QueryRewriter
from conversation.context_manager import ConversationContextManager

# Session store: session_id → ConversationContextManager
_sessions: dict[str, ConversationContextManager] = {}

# Shared retriever (loaded once at startup)
_retriever = FAISSRetriever()

# LLM config (ChatOpenAI is instantiated with request-specific model)
_llm_api_key: Optional[str] = None
_llm_base_url: Optional[str] = None


def init_retriever():
    """Load FAISS index + model into memory. Call at app startup."""
    _retriever.load()


def init_llm(api_key: str, base_url: str = None):
    """Initialize LLM settings for ChatOpenAI."""
    global _llm_api_key, _llm_base_url
    _llm_api_key = (api_key or "").strip()
    _llm_base_url = (base_url or "").strip() or None


def get_or_create_session(session_id: str) -> ConversationContextManager:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationContextManager()
    return _sessions[session_id]


def delete_session(session_id: str):
    _sessions.pop(session_id, None)


def format_context(products: List[dict]) -> str:
    """Format retrieved products into a context string for the LLM."""
    if not products:
        return "কোনো পণ্য পাওয়া যায়নি।"

    lines = ["প্রাসঙ্গিক পণ্যের তালিকা:"]
    for i, p in enumerate(products, 1):
        lines.append(
            f"{i}. {p['name']} | ক্যাটাগরি: {p['category']} | "
            f"মূল্য: {p['price']} টাকা প্রতি {p['unit']}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """আপনি একটি বাংলাদেশি কোম্পানির সহায়ক। আপনার কাজ হলো গ্রাহকদের পণ্য সংক্রান্ত প্রশ্নের উত্তর দেওয়া।

নিয়মাবলী:
- সবসময় বাংলায় উত্তর দিন
- শুধুমাত্র প্রদত্ত পণ্যের তথ্যের উপর ভিত্তি করে উত্তর দিন
- যদি পণ্য না পাওয়া যায়, তা বিনয়ের সাথে জানান
- সংক্ষিপ্ত ও স্পষ্ট উত্তর দিন
- দাম জিজ্ঞেস করলে সরাসরি দাম বলুন"""


async def query(
    session_id: str,
    user_message: str,
    top_k: int = 5,
    llm_model: str = "gpt-4o-mini",
    response_mode: str = "fast",
) -> dict:
    """
    Main entry point for a RAG query.

    Returns dict with:
      - response: str (LLM answer)
      - original_query: str
      - rewritten_query: str
      - was_rewritten: bool
      - retrieved_products: list
      - retrieval_ms: float
      - total_ms: float
    """
    total_start = time.perf_counter()

    ctx = get_or_create_session(session_id)

    # ── Step 1: Rewrite query (coreference resolution) ─────────────────────
    rewritten_query, was_rewritten, rewrite_reason = ctx.process_user_query(user_message)

    # ── Step 2: Retrieve (FAISS) ────────────────────────────────────────────
    products, retrieval_ms = _retriever.retrieve(rewritten_query, top_k=top_k)

    # ── Step 3: Record turn + update entity tracker ─────────────────────────
    ctx.record_user_turn(
        original_query=user_message,
        rewritten_query=rewritten_query,
        retrieved_products=products,
        retrieval_ms=retrieval_ms,
    )

    # ── Step 4: Build LLM messages ──────────────────────────────────────────
    context_str = format_context(products)

    # Include conversation history for multi-turn awareness
    history = ctx.get_history_for_llm()

    # Build langchain messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add prior turns (exclude the turn we just added)
    for turn in history[:-1]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Final user message with retrieved context
    final_user_content = (
        f"পণ্যের তথ্য:\n{context_str}\n\n"
        f"ব্যবহারকারীর প্রশ্ন: {rewritten_query}"
    )
    messages.append(HumanMessage(content=final_user_content))

    # ── Step 5: LLM call ────────────────────────────────────────────────────
    llm_start = time.perf_counter()

    use_llm = response_mode == "llm" and bool(_llm_api_key)

    if not use_llm:
        # Fallback: return a rule-based answer if no LLM configured
        response_text = _rule_based_answer(products, rewritten_query)
    else:
        llm_kwargs = {
            "api_key": _llm_api_key,
            "model": llm_model,
            "temperature": 0.3,
            "max_tokens": 300,
        }
        if _llm_base_url:
            llm_kwargs["base_url"] = _llm_base_url

        llm = ChatOpenAI(**llm_kwargs)
        response = await llm.ainvoke(messages)
        response_text = response.content if isinstance(response.content, str) else str(response.content)

    llm_ms = (time.perf_counter() - llm_start) * 1000

    # ── Step 6: Record assistant turn ──────────────────────────────────────
    ctx.record_assistant_turn(response_text)

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "response": response_text,
        "original_query": user_message,
        "rewritten_query": rewritten_query,
        "was_rewritten": was_rewritten,
        "rewrite_reason": rewrite_reason,
        "retrieved_products": products[:3],  # return top 3 for display
        "retrieval_ms": round(retrieval_ms, 2),
        "llm_ms": round(llm_ms, 2),
        "total_ms": round(total_ms, 2),
        "session_turn": ctx.turn_count,
        "tracked_entity": ctx.last_entity,
    }


def _rule_based_answer(products: List[dict], query: str) -> str:
    """Fallback answer when no LLM is configured."""
    if not products:
        return "দুঃখিত, আপনার চাওয়া পণ্যটি আমাদের কাছে পাওয়া যাচ্ছে না।"

    top = _select_product_for_query(products, query)
    top_price = top.get('price')
    has_price = isinstance(top_price, (int, float)) and float(top_price) > 0

    if any(kw in query for kw in ["দাম", "মূল্য", "কত", "price"]):
        poss = _to_possessive(top['name'])
        if not has_price:
            return f"হ্যাঁ, {poss} তথ্য আমাদের কাছে আছে, তবে এই মুহূর্তে দামের তথ্য পাওয়া যাচ্ছে না।"
        return (
            f"হ্যাঁ, {poss} দাম {top['price']} টাকা প্রতি {top['unit']}।"
        )
    if any(kw in query for kw in ["আছে", "বিক্রি", "পাওয়া"]):
        if not has_price:
            return (
                f"হ্যাঁ, আমরা {top['name']} বিক্রি করি। দামের তথ্য এই মুহূর্তে পাওয়া যাচ্ছে না।"
            )
        return (
            f"হ্যাঁ, আমরা {top['name']} বিক্রি করি। "
            f"মূল্য: {top['price']} টাকা প্রতি {top['unit']}।"
        )
    if not has_price:
        return f"{top['name']} পাওয়া যাচ্ছে। দামের তথ্য এই মুহূর্তে পাওয়া যাচ্ছে না।"
    return (
        f"{top['name']} পাওয়া যাচ্ছে। মূল্য: {top['price']} টাকা প্রতি {top['unit']}।"
    )


def _select_product_for_query(products: List[dict], query: str) -> dict:
    """Pick the best product aligned with explicit entity tokens in the query."""
    if not products:
        return {}

    stop_words = {
        "আপনাদের", "আপনার", "আমাদের", "আমার", "তোমার", "তাদের",
        "কোম্পানি", "দোকান", "এখানে", "এটা", "ওটা", "সেটা",
        "কি", "কিনা", "কোনো", "কোন", "কত", "কোথায়", "কখন",
        "আছে", "নেই", "হয়", "করে", "করেন", "পাওয়া", "যায়",
        "বিক্রি", "সরবরাহ", "দেওয়া", "পাই", "পাব", "বিক্রয়",
        "হ্যাঁ", "না", "ঠিক", "আচ্ছা", "দাম", "মূল্য", "টাকা",
        "এর", "ওর", "তার", "এই", "ওই", "সেই", "একটা", "একটি",
    }

    cleaned = re.sub(r"[^\w\s\u0980-\u09FF]", " ", query.lower())
    raw_words = [w for w in cleaned.split() if len(w) > 1 and w not in stop_words]
    query_words = []
    for w in raw_words:
        query_words.append(w)
        if w.endswith("ের") and len(w) > 3:
            query_words.append(w[:-2])
        elif w.endswith("র") and len(w) > 2:
            query_words.append(w[:-1])

    for product in products:
        name = product.get("name", "").lower()
        if any(w in name for w in query_words):
            return product

    return products[0]


def _to_possessive(noun: str) -> str:
    """Return simple Bangla possessive form (e.g., নুডুলস -> নুডুলসের)."""
    noun = (noun or "").strip()
    if not noun:
        return noun
    if noun.endswith("ের") or noun.endswith("র"):
        return noun
    return f"{noun}ের"
