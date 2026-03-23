"""
Test the exact scenario from the job posting.

Run: python -m tests.test_scenario

Expected output (fast mode):
  Q1 rewritten? No  (self-contained query)
  Q2 rewritten? YES → "নুডুলসের দাম কত টাকা?"
    Total Q2 < 100ms ✅
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.rag_pipeline import init_retriever, query


def check(condition: bool, label: str):
    icon = "✅" if condition else "❌"
    print(f"  {icon} {label}")
    return condition


async def run():
    print("=" * 60)
    print("  Bangla Context-Aware RAG — Self-Assessment Test")
    print("=" * 60)

    print("\n🔄 Loading FAISS index + model...")
    init_retriever()
    print("✅ Ready\n")

    # Warm-up one query so first measured turn is not affected by cold-start.
    await query("warmup-session", "হ্যালো", response_mode="fast")

    SESSION = "test-session-001"
    all_pass = True

    # ── Turn 1 ──────────────────────────────────────────────────────────────
    q1 = "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
    print(f"[Turn 1]")
    print(f"  User: {q1}")

    r1 = await query(SESSION, q1, response_mode="fast")

    print(f"  Was rewritten : {r1['was_rewritten']}")
    print(f"  Entity tracked: {r1['tracked_entity']}")
    print(f"  Retrieval     : {r1['retrieval_ms']:.1f}ms")
    print(f"  Response      : {r1['response']}")
    print()

    p1 = check(not r1['was_rewritten'], "Q1 NOT rewritten (self-contained)")
    p2 = check(r1['tracked_entity'] is not None, f"Entity extracted: '{r1['tracked_entity']}'")
    p3 = check(r1['retrieval_ms'] < 100, f"Retrieval < 100ms ({r1['retrieval_ms']:.1f}ms)")
    q1_has_noodles = any(p['name'] and 'নুডুলস' in p['name'] for p in r1['retrieved_products'])
    p3b = check(q1_has_noodles, "Q1 retrieval contains 'নুডুলস'")
    all_pass = all_pass and p1 and p2 and p3 and p3b

    # ── Turn 2 ──────────────────────────────────────────────────────────────
    q2 = "দাম কত টাকা?"
    print(f"[Turn 2]")
    print(f"  User: {q2}")

    r2 = await query(SESSION, q2, response_mode="fast")

    print(f"  Original query  : {r2['original_query']}")
    print(f"  Rewritten query : {r2['rewritten_query']}")
    print(f"  Was rewritten   : {r2['was_rewritten']}")
    print(f"  Retrieval       : {r2['retrieval_ms']:.1f}ms")
    print(f"  LLM             : {r2['llm_ms']:.1f}ms")
    print(f"  Total           : {r2['total_ms']:.1f}ms")
    print(f"  Response        : {r2['response']}")
    print()

    p4 = check(r2['was_rewritten'], "Q2 WAS rewritten (coreference resolved) ✨")
    p5 = check("নুডুলস" in r2['rewritten_query'], f"Rewritten query contains 'নুডুলস'")
    p6 = check(r2['retrieval_ms'] < 100, f"Retrieval < 100ms ({r2['retrieval_ms']:.1f}ms)")
    p6b = check(r2['total_ms'] < 100, f"Total < 100ms ({r2['total_ms']:.1f}ms)")

    # Check if নুডুলস price appears in response
    has_price = any(p['name'] and 'নুডুলস' in p['name'] for p in r2['retrieved_products'])
    p7 = check(has_price, "নুডুলস found in top retrieved products")
    p8 = check('নুডুলস' in r2['response'], "Q2 response is about 'নুডুলস'")

    all_pass = all_pass and p4 and p5 and p6 and p6b and p7 and p8

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 60)
    if all_pass:
        print("  🎉 ALL TESTS PASSED")
    else:
        print("  ⚠️  Some tests failed — check the output above")
    print("=" * 60)

    print("\n📊 Latency Summary:")
    print(f"  Q1 retrieval : {r1['retrieval_ms']:.1f}ms")
    print(f"  Q2 retrieval : {r2['retrieval_ms']:.1f}ms")
    print(f"  Target       : < 100ms")
    print(f"  Status       : {'✅ PASS' if r2['retrieval_ms'] < 100 else '❌ FAIL'}")


if __name__ == "__main__":
    asyncio.run(run())
