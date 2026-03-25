"""
Quality and hallucination-risk checks for Bangla Product RAG.

Run:
  python -m tests.test_quality_cases
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.rag_pipeline import init_retriever, query


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/products.json")


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    severity: str = "medium"  # low | medium | high


def _icon(ok: bool) -> str:
    return "✅" if ok else "❌"


def _load_products():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _category_expected(name: str):
    n = (name or "").lower()
    rules = [
        ("ডিজিটাল সেবা", ["সাবস্ক্রিপশন", "ভিপিএন", "ক্লাউড", "সফটওয়্যার", "ইন্টারনেট", "গিফট কার্ড"]),
        ("ইলেকট্রনিক্স", ["চার্জার", "পাওয়ার ব্যাংক", "ওয়েবক্যাম", "হেডফোন", "ইয়ারফোন", "ট্যাবলেট", "ল্যাপটপ", "স্মার্ট", "মোবাইল"]),
        ("স্বাস্থ্য ও ঔষধ", ["স্যানিটাইজার", "থার্মোমিটার", "ব্লাড", "অক্সিমিটার", "মাস্ক", "ব্যান্ডেজ"]),
        ("ফ্যাশন ও পোশাক", ["জিন্স", "শাড়ি", "জুতা", "স্কার্ফ", "পাঞ্জাবি", "পোশাক"]),
        ("আসবাবপত্র", ["টেবিল", "চেয়ার", "খাট", "আলমারি", "ড্রেসিং", "সোফা", "শেলফ"]),
        ("খাদ্য ও পানীয়", ["নুডুলস", "চা", "মশলা", "সরিষার তেল", "বেবি ফুড", "আটা"]),
        ("কৃষি ও বাগান", ["সার", "বীজ", "কীটনাশক", "পাম্প", "বাগান"]),
    ]
    for category, kws in rules:
        if any(k in n for k in kws):
            return category
    return None


def run_dataset_checks(products):
    results = []

    total = len(products)
    missing_price = sum(1 for p in products if p.get("price", -1) in (-1, None, 0))
    missing_ratio = (missing_price / total) if total else 0.0
    results.append(
        CheckResult(
            name="Price coverage",
            passed=missing_ratio < 0.8,
            details=f"missing/unknown price: {missing_price}/{total} ({missing_ratio:.1%})",
            severity="high",
        )
    )

    payment_contaminated = [
        p["name"]
        for p in products
        if any("ব্র্যান্ডের গুণগত মান" in x for x in p.get("payment_methods", []))
    ]
    contam_ratio = (len(payment_contaminated) / total) if total else 0.0
    results.append(
        CheckResult(
            name="Payment-field contamination",
            passed=contam_ratio < 0.05,
            details=f"products with brand text in payment_methods: {len(payment_contaminated)}/{total} ({contam_ratio:.1%})",
            severity="high",
        )
    )

    mismatches = []
    for p in products:
        expected = _category_expected(p.get("name", ""))
        actual = p.get("category")
        if expected and actual and expected != actual:
            mismatches.append((p.get("name", ""), actual, expected))

    mismatch_ratio = (len(mismatches) / total) if total else 0.0
    results.append(
        CheckResult(
            name="Category coherence",
            passed=mismatch_ratio < 0.15,
            details=f"keyword-based category mismatch: {len(mismatches)}/{total} ({mismatch_ratio:.1%})",
            severity="high",
        )
    )

    return results, {
        "payment_examples": payment_contaminated[:5],
        "category_examples": mismatches[:5],
    }


async def run_behavior_checks():
    results = []

    # 1) Ambiguous standalone price query should ask clarification
    r1 = await query("qc-standalone-1", "প্রাইস কত টাকা", response_mode="fast")
    ok1 = "কোন পণ্যের দাম" in r1["response"]
    results.append(
        CheckResult(
            name="Ambiguous price clarification",
            passed=ok1,
            details=f"response={r1['response']}",
            severity="high",
        )
    )

    # 2) Follow-up price query should keep entity context (no hallucinated switch)
    _ = await query("qc-followup-1", "কোম্পানি কি নুডুলস বিক্রি করে", response_mode="fast")
    r2 = await query("qc-followup-1", "প্রাইস কত টাকা", response_mode="fast")
    ok2 = ("নুডুলস" in r2["rewritten_query"]) and ("নুডুলস" in r2["response"])
    results.append(
        CheckResult(
            name="Follow-up entity retention",
            passed=ok2,
            details=f"rewritten={r2['rewritten_query']} | response={r2['response']}",
            severity="high",
        )
    )

    # 3) New explicit entity should override previous context
    _ = await query("qc-switch-1", "কোম্পানি কি নুডুলস বিক্রি করে", response_mode="fast")
    r3 = await query("qc-switch-1", "চেয়ার আছে?", response_mode="fast")
    ok3 = ("চেয়ার" in r3["response"]) and ("নুডুলস" not in r3["response"])
    results.append(
        CheckResult(
            name="Entity switch handling",
            passed=ok3,
            details=f"response={r3['response']}",
            severity="medium",
        )
    )

    # 4) Unknown product should not hallucinate availability
    r4 = await query("qc-unknown-1", "আপনাদের কাছে স্পেসশিপ ব্যাটারি আছে?", response_mode="fast")
    ok4 = (
        any(x in r4["response"] for x in ["পাওয়া যাচ্ছে না", "দুঃখিত"])
        and ("হ্যাঁ, আমরা" not in r4["response"])
    )
    results.append(
        CheckResult(
            name="Unknown product refusal",
            passed=ok4,
            details=f"response={r4['response']}",
            severity="high",
        )
    )

    return results


async def main():
    print("=" * 68)
    print("Bangla Product RAG — Quality, Misleading & Hallucination Checks")
    print("=" * 68)

    print("\n🔄 Loading retriever...")
    init_retriever()
    print("✅ Retriever ready")

    products = _load_products()
    dataset_results, samples = run_dataset_checks(products)
    behavior_results = await run_behavior_checks()

    all_results = dataset_results + behavior_results

    print("\n📊 Check Results")
    for r in all_results:
        print(f"  {_icon(r.passed)} {r.name} [{r.severity}] -> {r.details}")

    high_fail = [r for r in all_results if (not r.passed and r.severity == "high")]
    medium_fail = [r for r in all_results if (not r.passed and r.severity == "medium")]

    print("\n🧪 Sample Problem Examples")
    print(f"  - payment contamination examples: {samples['payment_examples']}")
    print(f"  - category mismatch examples: {samples['category_examples']}")

    print("\n🧭 Assessment")
    if not high_fail and not medium_fail:
        print("  ✅ System behavior looks robust and low-risk.")
    elif high_fail:
        print("  ❌ High-risk misleading/hallucination vectors still exist.")
        names = {r.name for r in high_fail}
        if "Price coverage" in names:
            print("  - Primary risk source is missing structured price values in products.json")
        elif "Payment-field contamination" in names:
            print("  - Primary risk source is field contamination in products.json")
        else:
            print("  - Primary risk source is mixed data + retrieval behavior")
    else:
        print("  ⚠️ No high-risk failures, but medium-risk issues remain.")

    print("\nSummary:")
    print(f"  total checks: {len(all_results)}")
    print(f"  passed: {sum(1 for r in all_results if r.passed)}")
    print(f"  failed: {sum(1 for r in all_results if not r.passed)}")
    print(f"  high-risk failed: {len(high_fail)}")


if __name__ == "__main__":
    asyncio.run(main())
