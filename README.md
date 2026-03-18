# Bangla Product RAG - Context-Aware Product Search

A context-aware Bangla RAG system for product queries, designed for the self-assessment scenario:

Q1: আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?
Q2: দাম কত টাকা?

Core requirement: Q2 should be resolved as নুডুলসের দাম কত টাকা? and total RAG response should stay under 100ms in assessment mode.

## Current Status

- Context-aware rewriting: implemented
- Retrieval latency target (<100ms): passing
- Scenario test: passing
- FastAPI API server: implemented
- OpenAI key loading from .env: fixed

## Project Structure

- `main.py` - FastAPI app and endpoints
- `pipeline/rag_pipeline.py` - end-to-end orchestration
- `conversation/context_manager.py` - session state and turn tracking
- `retriever/query_rewriter.py` - rule-based coreference-aware rewriting
- `retriever/faiss_retriever.py` - FAISS retrieval with lexical re-ranking fallback
- `indexer/build_index.py` - offline embedding + FAISS index build
- `data/generate_products.py` - 5000-product dataset generation
- `tests/test_scenario.py` - self-assessment scenario test
- `data.txt` - assessment description/reference text

## Architecture

1. User query comes in.
2. `ConversationContextManager` rewrites context-dependent queries.
3. `FAISSRetriever` retrieves top candidates.
4. Retrieved products are passed to LLM (or rule-based fallback if no API key).
5. Response is returned with diagnostics (rewritten query, latency, tracked entity).

Response modes:
- `fast` (default): deterministic rule-based answer over retrieved context (assessment mode, designed for <100ms total)
- `llm`: uses `ChatOpenAI` for richer generation (latency depends on provider/network)

## Setup

1. Create and activate virtual environment.

Windows PowerShell:
```bash
.venv\Scripts\activate
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Configure environment.
```bash
copy .env.example .env
```

Set at least:
```env
OPENAI_API_KEY=your_key_here
# Optional
OPENAI_BASE_URL=https://api.openai.com/v1
HF_TOKEN=your_hf_token_optional
```

## Build Dataset and Index

1. Generate dataset (exactly 5000 products).
```bash
python data/generate_products.py
```

2. Build FAISS index.
```bash
python -m indexer.build_index
```

Expected index size after rebuild: 5000 vectors.

## Run Tests

```bash
python -m tests.test_scenario
```

Expected:
- Q1 not rewritten
- Q2 rewritten
- Q2 rewritten query contains নুডুলস
- Retrieval < 100ms
- Total Q2 < 100ms (fast mode)

## Run Server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- `http://localhost:8000`
- `http://localhost:8000/docs`

## Streamlit UI (FastAPI on 8000)

1. Start FastAPI in one terminal:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Start Streamlit in another terminal:
```bash
python -m streamlit run streamlit_app.py --server.port 8501
```

3. Open UI:
- `http://localhost:8501`

The sidebar lets you set:
- FastAPI base URL (default: `http://127.0.0.1:8000`)
- `response_mode` (`fast` or `llm`)
- `top_k`, `llm_model`, and `session_id`

## Docker Deployment

This project includes a production-friendly container setup:
- `Dockerfile`
- `docker-compose.yml`

Services:
- `api` on port `8000`
- `streamlit` on port `8501`

Run both services:
```bash
docker compose up --build
```

Run in background:
```bash
docker compose up --build -d
```

Stop services:
```bash
docker compose down
```

Open:
- API docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

Note: In Docker, Streamlit uses `API_BASE_URL=http://api:8000` automatically via Compose.

## API

### POST `/chat`

Request:
```json
{
  "message": "দাম কত টাকা?",
  "session_id": "sess_abc123",
  "top_k": 5,
  "llm_model": "gpt-4o-mini",
  "response_mode": "fast"
}
```

Response includes:
- `original_query`
- `rewritten_query`
- `was_rewritten`
- `retrieved_products`
- `retrieval_ms`, `llm_ms`, `total_ms`
- `tracked_entity`

## Notes on Latency

- `fast` mode is optimized for <100ms total response in this assessment scenario.
- `llm` mode latency depends on provider/model and typically exceeds 100ms over network calls.
- If `OPENAI_API_KEY` is missing, system falls back to rule-based response generation.

## Why This Meets The Assessment

- Handles multi-turn coreference in Bangla using deterministic rewriting.
- Prevents generic semantic retrieval misses by adding lexical fallback re-ranking.
- Provides clear test output for the exact Q1/Q2 scenario.
- Uses clean modular Python project structure suitable for production hardening.
