"""
Core FAISS retriever — loads index once, serves queries in <5ms.

Strategy:
  1. FAISS ANN search over top_k * 4 candidates (semantic similarity)
  2. Keyword boost: re-rank by checking if query words appear in product name
     This fixes the case where the model matches query intent ("বিক্রি করে")
     instead of the product noun ("নুডুলস").
  3. Return top_k after re-ranking.
"""
import os
import pickle
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

INDEX_PATH = os.path.join(os.path.dirname(__file__), "../data/faiss.index")
META_PATH  = os.path.join(os.path.dirname(__file__), "../data/products_meta.pkl")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Query words that are NOT product names — skip these during keyword boost
STOP_WORDS = {
    "আপনাদের", "আপনার", "আমাদের", "আমার", "তোমার", "তাদের",
    "কোম্পানি", "দোকান", "এখানে", "এটা", "ওটা", "সেটা",
    "কি", "কিনা", "কোনো", "কোন", "কত", "কোথায়", "কখন",
    "আছে", "নেই", "হয়", "করে", "করেন", "পাওয়া", "যায়",
    "বিক্রি", "সরবরাহ", "দেওয়া", "পাই", "পাব", "বিক্রয়",
    "হ্যাঁ", "না", "ঠিক", "আচ্ছা", "দাম", "মূল্য", "টাকা",
    "এর", "ওর", "তার", "এই", "ওই", "সেই", "একটা", "একটি",
}

KEYWORD_BOOST = 0.25   # added to score when product name word matches query
EXACT_BOOST   = 0.40   # added when full product base name is in query
LEXICAL_MATCH_BOOST = 1.20  # strong signal for exact word match in product name


def _normalize_tokens(text: str) -> list[str]:
    """Tokenize text and strip punctuation so Bangla words match reliably."""
    cleaned = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text.lower())
    raw = [w for w in cleaned.split() if len(w) > 1]

    # Expand common Bangla possessive endings so "নুডুলসের" can match "নুডুলস".
    tokens = []
    for w in raw:
        tokens.append(w)
        if w.endswith("ের") and len(w) > 3:
            tokens.append(w[:-2])
        elif w.endswith("র") and len(w) > 2:
            tokens.append(w[:-1])

    # Keep order while removing duplicates.
    return list(dict.fromkeys(tokens))


class FAISSRetriever:
    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._products: List[Dict] | None = None

    def load(self):
        """Load model + index into memory. Call once at startup."""
        print("🔄 Loading embedding model...")
        self._model = SentenceTransformer(MODEL_NAME)
        self._model.encode(["warmup"], show_progress_bar=False)
        print("✅ Model ready")

        print("🔄 Loading FAISS index...")
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python -m indexer.build_index"
            )
        self._index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            self._products = pickle.load(f)

        print(f"✅ Index loaded: {self._index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 5) -> tuple[List[Dict], float]:
        """
        Retrieve top_k products for a query with keyword boost re-ranking.
        Returns (results, retrieval_ms).
        """
        t0 = time.perf_counter()

        # ── Step 1: FAISS semantic search (wider pool) ──────────────────────
        query_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Fetch a wider candidate pool so keyword boost has room to re-rank
        pool = min(top_k * 6, self._index.ntotal)
        scores, indices = self._index.search(query_vec, pool)

        # ── Step 2: Keyword boost re-ranking ───────────────────────────────
        # Extract meaningful words from the query (non-stopwords)
        query_words = [w for w in _normalize_tokens(query) if w not in STOP_WORDS]

        candidates = []
        seen_product_ids = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            product = self._products[idx].copy()
            base_score = float(score)
            boost = 0.0

            product_name = product["name"].lower()

            # Exact product base name in query
            for word in query_words:
                if word in product_name:
                    boost += KEYWORD_BOOST
                    break

            # Full product name substring match in query (strongest signal)
            # e.g. query contains "নুডুলস" and product name is "নুডুলস"
            for word in query_words:
                if len(word) >= 3 and product_name.startswith(word):
                    boost += EXACT_BOOST
                    break

            product["_score"] = base_score + boost
            product["_base_score"] = base_score
            product["_boost"] = boost
            candidates.append(product)
            seen_product_ids.add(product.get("id"))

        # ── Step 3: Lexical fallback pass over full catalog ─────────────────
        # FAISS may miss explicit noun matches in first-pass candidates.
        # For small catalogs (~5k), a lightweight scan is fast and reliable.
        if query_words:
            for product in self._products:
                product_name = product["name"].lower()
                match_word = next((w for w in query_words if w in product_name), None)
                if not match_word:
                    continue

                if product.get("id") in seen_product_ids:
                    continue

                lexical = product.copy()
                lexical["_base_score"] = 0.0
                lexical["_boost"] = LEXICAL_MATCH_BOOST
                lexical["_score"] = LEXICAL_MATCH_BOOST
                candidates.append(lexical)
                seen_product_ids.add(lexical.get("id"))

        # Re-rank by boosted score
        candidates.sort(key=lambda p: p["_score"], reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return candidates[:top_k], elapsed_ms

    @property
    def is_loaded(self) -> bool:
        return self._index is not None