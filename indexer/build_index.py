"""
Offline indexer — updated with improved authentication and logging.
"""
import json
import os
import time
import pickle
import numpy as np

import faiss
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/products.json")
INDEX_PATH = os.path.join(BASE_DIR, "../data/faiss.index")
META_PATH  = os.path.join(BASE_DIR, "../data/products_meta.pkl")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

def build_index():
    # 1. Authenticate globally to silence the "unauthenticated" warning
    if HF_TOKEN:
        print("🔑 Authenticating with Hugging Face...")
        login(token=HF_TOKEN)
    else:
        print("⚠️ No HF_TOKEN found in environment. Using public access (rate limits apply).")

    print("📦 Loading products...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find data at {DATA_PATH}")
        
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"✅ Loaded {len(products)} products")

    print(f"🤖 Loading embedding model: {MODEL_NAME}...")
    # Passing token=HF_TOKEN here as well for redundancy
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)

    # Use the description field for richer semantic search
    texts = [p.get("description", p.get("name", "")) for p in products]

    print("⚡ Generating embeddings (this takes ~30s first time)...")
    t0 = time.time()
    
    # generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=32, # Lower batch size can sometimes be more stable on consumer hardware
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    
    print(f"✅ Embeddings done in {time.time()-t0:.1f}s | shape: {embeddings.shape}")

    dim = embeddings.shape[1]

    # IndexFlatIP = exact cosine search (fast enough for 5000 vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Ensure directory exists
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    print(f"💾 FAISS index saved → {INDEX_PATH}")

    with open(META_PATH, "wb") as f:
        pickle.dump(products, f)
    print(f"💾 Metadata saved → {META_PATH}")

    print(f"\n✅ Index built: {index.ntotal} vectors @ dim={dim}")
    return index, products

if __name__ == "__main__":
    build_index()
