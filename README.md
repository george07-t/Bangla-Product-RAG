# Bangla Product RAG - Context-Aware Product Search

A context-aware Bangla RAG system for product queries.

Example questions:

- আপনাদের কাছে নুডুলস আছে?
- দাম কত?
- ওয়ারেন্টি আছে?
- ক্যাশ অন ডেলিভারি আছে?
- কোনো অফার আছে?
- ডেলিভারি কত দ্রুত?

Core requirement: Q2 should be resolved as নুডুলসের দাম কত টাকা? and total RAG response should stay under 100ms in assessment mode.

## Business Impact and Engineering Depth

- Faster customer answer cycles: context-aware follow-up handling reduces clarification turns and improves first-response usefulness for price/product queries.
- Higher retrieval precision in Bangla: hybrid semantic + lexical strategy reduces irrelevant matches and improves trust in product answers.
- Latency-aware product design: dual response modes (`fast` for strict latency paths, `llm` for richer generation) allow balancing UX speed vs. response richness by use case.
- Operational transparency for teams: structured response diagnostics (`was_rewritten`, `tracked_entity`, `retrieval_ms`, `llm_ms`, `total_ms`) improve debugging, QA, and SLA monitoring.
- Scalable architecture boundaries: indexing, retrieval, rewriting, session context, API, and UI are cleanly separated for easier maintenance and independent optimization.
- Deployment readiness: local, API-first, Streamlit UI, and Docker Compose workflows support rapid prototyping and production-oriented rollout paths.

## Architecture

1. User query comes in.
2. `ConversationContextManager` rewrites context-dependent queries.
3. `FAISSRetriever` retrieves top candidates.
4. Retrieved products are passed to LLM (or rule-based fallback if no API key).
5. Response is returned with diagnostics (rewritten query, latency, tracked entity).

Response modes:
- `fast` (default): deterministic rule-based answer over retrieved context (assessment mode, designed for <100ms total)
- `llm`: uses `ChatOpenAI` for richer generation (latency depends on provider/network)

## Model Choice and LangChain Rationale

### Embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

- Why this model:
  - Strong multilingual coverage including Bangla.
  - Good semantic quality with compact embedding size (384 dims), which keeps indexing and retrieval fast.
  - Practical latency/memory profile for local or containerized deployment.
- Trade-off:
  - Larger multilingual models may improve recall slightly but usually add compute/memory overhead that is not ideal for strict latency targets.

### LLM model (`gpt-4o-mini` by default)

- Why this model:
  - Good balance of response quality, speed, and cost for production-style chat responses.
  - Works well for concise Bangla product-answer generation over retrieved context.
- Operational strategy:
  - `fast` mode avoids remote LLM dependency for strict latency paths.
  - `llm` mode uses `gpt-4o-mini` (or another compatible model) when richer natural language response quality is preferred.

### Why LangChain (`langchain-openai` + `ChatOpenAI`)

- Standardized model interface:
  - Keeps LLM invocation clean and consistent (`ainvoke`) without provider-specific request wiring in business logic.
- Easier provider/model swap:
  - OpenAI-compatible endpoint changes can be handled by config (`OPENAI_BASE_URL`) while preserving call flow.
- Maintainability:
  - Message abstractions (`SystemMessage`, `HumanMessage`, `AIMessage`) make multi-turn prompt assembly clearer and safer to extend.
- Production pragmatism:
  - LangChain is used selectively for LLM orchestration; retrieval, rewriting, and ranking remain custom modules for speed and control.

## Minimal Setup

1. Activate virtual environment.

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

## Dataset Preparation (Knowledge_Bank Primary)

1. Preprocess `Knowledge_Bank.txt` (first-pass dedupe + canonical merge + category policy).
```powershell
python .\data\preprocess_knowledge_bank.py
```

2. Rebuild FAISS index.
```powershell
python -m indexer.build_index
```

Optional synthetic dataset override:
```powershell
python .\data\generate_products.py
python -m indexer.build_index
```

## Run

1. Start API server.
```powershell
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Optional Streamlit UI (new terminal).
```powershell
python -m streamlit run streamlit_app.py --server.port 8501
```

## Run with Docker

1. Ensure `.env` exists (required by `docker-compose.yml` `env_file`).
```powershell
copy .env.example .env
```

2. Build and run API + Streamlit.
```powershell
docker compose up --build
```

3. Run in background (optional).
```powershell
docker compose up --build -d
```

4. Stop services.
```powershell
docker compose down
```

5. Open services.
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

Notes:
- Docker automatically runs:
  - `python data/preprocess_knowledge_bank.py`
  - `python -m indexer.build_index`
  before starting the API server.
- If startup is slow, it is expected because preprocessing + indexing run inside the container.
- `.dockerignore` excludes local env/cache files (`.venv`, `__pycache__`, logs), so image context stays clean.

## Test

Run scenario test:
```powershell
python -m tests.test_scenario
```

Manual API checks:
- Open: `http://localhost:8000/docs`
- Test queries such as:
  - `আপনাদের কাছে নুডুলস আছে?`
  - `দাম কত?`
  - `ওয়ারেন্টি আছে?`
  - `ক্যাশ অন ডেলিভারি আছে?`
  - `কোনো অফার আছে?`

