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
import math
from collections import Counter
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
BM25_WEIGHT = 0.35
BM25_TOP_M = 30
BM25_K1 = 1.5
BM25_B = 0.75


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
        self._bm25_doc_freq: Dict[str, int] = {}
        self._bm25_doc_tf: List[Counter] = []
        self._bm25_doc_len: List[int] = []
        self._bm25_avgdl: float = 0.0

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

        self._build_bm25_stats()

        print(f"✅ Index loaded: {self._index.ntotal} vectors")

    def _build_bm25_stats(self):
        """Build lightweight BM25 stats over product name + description."""
        self._bm25_doc_freq = {}
        self._bm25_doc_tf = []
        self._bm25_doc_len = []

        for product in self._products:
            text = f"{product.get('name', '')} {product.get('description', '')}"
            tokens = _normalize_tokens(text)
            tf = Counter(tokens)
            self._bm25_doc_tf.append(tf)
            self._bm25_doc_len.append(len(tokens))

            for token in tf:
                self._bm25_doc_freq[token] = self._bm25_doc_freq.get(token, 0) + 1

        self._bm25_avgdl = (sum(self._bm25_doc_len) / len(self._bm25_doc_len)) if self._bm25_doc_len else 0.0

    def _bm25_scores(self, query_tokens: List[str]) -> List[float]:
        """Return BM25 score per document index."""
        n_docs = len(self._bm25_doc_tf)
        if not query_tokens or n_docs == 0 or self._bm25_avgdl <= 0:
            return [0.0] * n_docs

        scores = [0.0] * n_docs
        query_terms = [q for q in query_tokens if q not in STOP_WORDS]
        if not query_terms:
            return scores

        for term in query_terms:
            df = self._bm25_doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))

            for idx, tf in enumerate(self._bm25_doc_tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue

                dl = self._bm25_doc_len[idx]
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * (dl / self._bm25_avgdl))
                scores[idx] += idf * ((f * (BM25_K1 + 1)) / denom)

        return scores

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

        # BM25 signal across full catalog
        bm25_scores = self._bm25_scores(_normalize_tokens(query))

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

            bm25_boost = BM25_WEIGHT * bm25_scores[idx]
            product["_score"] = base_score + boost + bm25_boost
            product["_base_score"] = base_score
            product["_boost"] = boost
            product["_bm25"] = bm25_boost
            candidates.append(product)
            seen_product_ids.add(product.get("id"))

        # Add BM25-only strong candidates to improve noun/entity precision
        if bm25_scores:
            top_bm25_indices = np.argsort(np.array(bm25_scores))[::-1][:BM25_TOP_M]
            for idx in top_bm25_indices:
                if bm25_scores[idx] <= 0:
                    break

                product = self._products[idx]
                if product.get("id") in seen_product_ids:
                    continue

                bm = product.copy()
                bm["_base_score"] = 0.0
                bm["_boost"] = 0.0
                bm["_bm25"] = BM25_WEIGHT * bm25_scores[idx]
                bm["_score"] = bm["_bm25"]
                candidates.append(bm)
                seen_product_ids.add(bm.get("id"))

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