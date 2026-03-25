"""
Dataset-wide context-aware chat flow matrix.

Run:
  python -m tests.test_full_chatflow_matrix
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.rag_pipeline import init_retriever, query


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/products.json")


@dataclass
class Metric:
    name: str
    passed: int = 0
    failed: int = 0

    def total(self) -> int:
        return self.passed + self.failed

    def pass_rate(self) -> float:
        return (self.passed / self.total()) if self.total() else 0.0


def _load_products() -> List[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _anchor_tokens(name: str) -> List[str]:
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in (name or "").lower())
    parts = [p for p in cleaned.split() if len(p) > 1]
    return parts[:2] if parts else [name.lower()]


def _contains_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w.lower() in t for w in words if w)


async def main():
    print("=" * 72)
    print("Bangla Product RAG — Full Dataset Context-Aware Chat Flow Matrix")
    print("=" * 72)

    print("\n🔄 Loading retriever...")
    init_retriever()
    print("✅ Retriever ready")

    products = _load_products()
    print(f"\n📦 Products in matrix: {len(products)}")

    m_rewrite_price = Metric("Follow-up price rewrite")
    m_rewrite_review = Metric("Follow-up review rewrite")
    m_rewrite_warranty = Metric("Follow-up warranty rewrite")
    m_context_review_resp = Metric("Follow-up review response alignment")
    m_context_warranty_resp = Metric("Follow-up warranty response alignment")
    m_switch = Metric("Entity switch preference")

    samples = {
        "price_rewrite_fail": [],
        "review_rewrite_fail": [],
        "warranty_rewrite_fail": [],
        "review_resp_fail": [],
        "warranty_resp_fail": [],
        "switch_fail": [],
    }

    # Per-product multi-turn flow
    for idx, p in enumerate(products, start=1):
        name = p.get("name", "")
        if not name:
            continue

        tokens = _anchor_tokens(name)
        sid = f"full-matrix-{idx}"

        _ = await query(sid, f"আপনাদের কাছে {name} আছে?", response_mode="fast")
        r_price = await query(sid, "দাম কত?", response_mode="fast")
        r_review = await query(sid, "রিভিউ অনুযায়ী এটা কত স্টার?", response_mode="fast")
        r_warranty = await query(sid, "ওয়ারেন্টি কী?", response_mode="fast")

        ok_price_rewrite = bool(r_price["was_rewritten"]) and _contains_any(r_price["rewritten_query"], tokens)
        ok_review_rewrite = bool(r_review["was_rewritten"]) and _contains_any(r_review["rewritten_query"], tokens)
        ok_warranty_rewrite = bool(r_warranty["was_rewritten"]) and _contains_any(r_warranty["rewritten_query"], tokens)

        if ok_price_rewrite:
            m_rewrite_price.passed += 1
        else:
            m_rewrite_price.failed += 1
            if len(samples["price_rewrite_fail"]) < 8:
                samples["price_rewrite_fail"].append((name, r_price["rewritten_query"]))

        if ok_review_rewrite:
            m_rewrite_review.passed += 1
        else:
            m_rewrite_review.failed += 1
            if len(samples["review_rewrite_fail"]) < 8:
                samples["review_rewrite_fail"].append((name, r_review["rewritten_query"]))

        if ok_warranty_rewrite:
            m_rewrite_warranty.passed += 1
        else:
            m_rewrite_warranty.failed += 1
            if len(samples["warranty_rewrite_fail"]) < 8:
                samples["warranty_rewrite_fail"].append((name, r_warranty["rewritten_query"]))

        ok_review_resp = _contains_any(r_review["response"], ["রিভিউ", "স্টার", "তথ্য"])
        ok_warranty_resp = _contains_any(r_warranty["response"], ["ওয়ারেন্টি", "তথ্য", "পাওয়া যাচ্ছে না"])

        if ok_review_resp:
            m_context_review_resp.passed += 1
        else:
            m_context_review_resp.failed += 1
            if len(samples["review_resp_fail"]) < 8:
                samples["review_resp_fail"].append((name, r_review["response"]))

        if ok_warranty_resp:
            m_context_warranty_resp.passed += 1
        else:
            m_context_warranty_resp.failed += 1
            if len(samples["warranty_resp_fail"]) < 8:
                samples["warranty_resp_fail"].append((name, r_warranty["response"]))

    # Entity switch flow (A -> B -> follow-up should attach to B)
    for i in range(len(products) - 1):
        a = products[i].get("name", "")
        b = products[i + 1].get("name", "")
        if not a or not b:
            continue

        b_tokens = _anchor_tokens(b)
        sid = f"switch-matrix-{i}"
        _ = await query(sid, f"আপনাদের কাছে {a} আছে?", response_mode="fast")
        _ = await query(sid, f"{b} আছে?", response_mode="fast")
        r_follow = await query(sid, "দাম কত?", response_mode="fast")

        ok_switch = bool(r_follow["was_rewritten"]) and _contains_any(r_follow["rewritten_query"], b_tokens)
        if ok_switch:
            m_switch.passed += 1
        else:
            m_switch.failed += 1
            if len(samples["switch_fail"]) < 8:
                samples["switch_fail"].append((a, b, r_follow["rewritten_query"]))

    metrics = [
        m_rewrite_price,
        m_rewrite_review,
        m_rewrite_warranty,
        m_context_review_resp,
        m_context_warranty_resp,
        m_switch,
    ]

    print("\n📊 Matrix Results")
    for m in metrics:
        print(
            f"  {'✅' if m.pass_rate() >= 0.90 else '❌'} {m.name}: "
            f"{m.passed}/{m.total()} ({m.pass_rate():.1%})"
        )

    print("\n🧪 Failure Samples")
    for key, vals in samples.items():
        print(f"  - {key}: {vals}")

    hard_fail = [m for m in metrics if m.pass_rate() < 0.90]
    print("\n🧭 Assessment")
    if not hard_fail:
        print("  ✅ Dataset-wide chat flow looks robust (>=90% on all tracked context metrics).")
    else:
        print("  ❌ Some context-aware metrics are below target (90%).")
        for m in hard_fail:
            print(f"    - {m.name}: {m.pass_rate():.1%}")

    assert not hard_fail, "One or more context-aware matrix metrics are below 90%"


if __name__ == "__main__":
    asyncio.run(main())
