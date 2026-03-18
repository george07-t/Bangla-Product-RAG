"""
FastAPI RAG Server — Bangla context-aware product search.
"""
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pipeline.rag_pipeline import (
    init_retriever,
    init_llm,
    query,
    delete_session,
    get_or_create_session,
)


# Load environment variables from project root .env (if present)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index and embedding model at startup."""
    print("🚀 Starting RAG server...")

    # Initialize retriever (loads FAISS + SentenceTransformer)
    init_retriever()

    # Initialize LLM if API key is set
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if api_key:
        base_url = os.getenv("OPENAI_BASE_URL")  # Optional: for Groq/Together/etc.
        init_llm(api_key=api_key, base_url=base_url)
        print("✅ LLM client initialized")
    else:
        print("⚠️  No OPENAI_API_KEY found — using rule-based fallback answers")

    print("✅ Server ready!")
    yield
    print("👋 Shutting down...")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bangla RAG API",
    description="Context-aware Bangla product search with coreference resolution",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Request / Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: int = 5
    llm_model: str = "gpt-4o-mini"
    response_mode: str = "fast"  # fast | llm


class ChatResponse(BaseModel):
    session_id: str
    response: str
    original_query: str
    rewritten_query: str
    was_rewritten: bool
    rewrite_reason: str
    retrieved_products: list
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    session_turn: int
    tracked_entity: Optional[str]


class ResetRequest(BaseModel):
    session_id: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    html_candidates = [
        os.path.join(static_dir, "index.html"),
        os.path.join(os.path.dirname(__file__), "index.html"),
    ]
    for html_path in html_candidates:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
    return HTMLResponse("<h1>Bangla RAG API</h1><p>Visit <a href='/docs'>/docs</a> for API docs.</p>")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Automatically resolves coreferences across conversation turns.

    Example:
      Turn 1: "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
      Turn 2: "দাম কত টাকা?"  → auto-rewritten to "নুডুলসের দাম কত টাকা?"
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Auto-create session if not provided
    session_id = req.session_id or str(uuid.uuid4())

    result = await query(
        session_id=session_id,
        user_message=req.message,
        top_k=req.top_k,
        llm_model=req.llm_model,
        response_mode=req.response_mode,
    )

    return ChatResponse(session_id=session_id, **result)


@app.post("/reset")
async def reset_session(req: ResetRequest):
    """Clear conversation history for a session."""
    delete_session(req.session_id)
    return {"status": "ok", "message": f"Session {req.session_id} cleared"}


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get info about an active session."""
    ctx = get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "turn_count": ctx.turn_count,
        "tracked_entity": ctx.last_entity,
        "history": ctx.get_history_for_llm(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "message": "RAG server is running"}
