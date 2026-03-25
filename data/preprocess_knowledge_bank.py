"""
First-pass preprocessing for noisy Knowledge_Bank.txt data.

Goals:
1) Remove repeated boilerplate promo sentences.
2) Extract structured attributes (brand, warranty, offers, payment, delivery, rating).
3) Canonical merge duplicated products with conflicting attributes.
4) Apply category policy for physical vs digital/service products.
5) Export schema-compatible JSON for current RAG pipeline.

Usage:
  python data/preprocess_knowledge_bank.py \
    --input data/Knowledge_Bank.txt \
        --output data/products.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

BOILERPLATE_PATTERNS = [
    r"১০০% অরিজিনাল পণ্য",
    r"নকলের বিরুদ্ধে গ্যারান্টি",
    r"দ্রুত ডেলিভারি",
    r"সারা বাংলাদেশে হোম ডেলিভারি",
    r"পরিবেশবান্ধব প্যাকেজিং",
    r"নিরাপদ ও সুরক্ষিত",
    r"প্রতিটি পণ্য কঠোর মান নিয়ন্ত্রণ",
    r"সর্বোচ্চ গ্রাহক সন্তুষ্টি নিশ্চিত",
    r"রিটার্ন পলিসি প্রযোজ্য",
    r"গ্রাহকদের রিভিউ অনুযায়ী এটি ৪\.৫ স্টার রেটিং",
]

DIGITAL_SERVICE_KEYWORDS = {
    "সাবস্ক্রিপশন",
    "ভিপিএন",
    "vpn",
    "ক্লাউড স্টোরেজ",
    "সফটওয়্যার লাইসেন্স",
    "ইন্টারনেট প্যাকেজ",
    "ডিজিটাল গিফট কার্ড",
    "লাইসেন্সড সফটওয়্যার",
}

CATEGORY_KEYWORDS = {
    "ডিজিটাল সেবা": ["সাবস্ক্রিপশন", "ভিপিএন", "ক্লাউড", "সফটওয়্যার", "ইন্টারনেট প্যাকেজ", "গিফট কার্ড", "রিচার্জ সেবা", "মোবাইল রিচার্জ"],
    "ইলেকট্রনিক্স": ["চার্জার", "পাওয়ার ব্যাংক", "ওয়েবক্যাম", "ফোন", "হেডফোন", "ইয়ারফোন", "ফ্রিজ", "ট্যাবলেট", "ল্যাপটপ", "স্মার্ট ওয়াচ"],
    "স্বাস্থ্য ও ঔষধ": ["স্যানিটাইজার", "ব্যান্ডেজ", "থার্মোমিটার", "ব্লাড প্রেশার", "পালস অক্সিমিটার", "মাস্ক"],
    "ফ্যাশন ও পোশাক": ["জিন্স", "শাড়ি", "জুতা", "স্কার্ফ", "পাঞ্জাবি", "পোশাক"],
    "আসবাবপত্র": ["টেবিল", "চেয়ার", "খাট", "আলমারি", "ড্রেসিং", "সোফা", "বুক শেলফ"],
    "খাদ্য ও পানীয়": ["নুডুলস", "চা", "মশলা", "সরিষার তেল", "আচার", "বেবি ফুড"],
    "কৃষি ও বাগান": ["সার", "বীজ", "কীটনাশক", "পাম্প", "বাগানের"],
}

BRAND_RE = re.compile(r"([\u0980-\u09FFA-Za-z]+) ব্র্যান্ডের গুণগত মানের পণ্য")
WARRANTY_RE = re.compile(r"এই পণ্যের ([\u0980-\u09FFA-Za-z]+) ওয়ারেন্টি রয়েছে")
RATING_RE = re.compile(r"([0-9০-৯]+(?:[\.,٫][0-9০-৯]+)?)\s*স্টার")

BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _split_blocks(raw_text: str) -> List[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw_text) if b.strip()]
    return blocks


def _split_sentences(block: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[।!?]\s*", block) if p.strip()]
    return parts


def _is_boilerplate(sentence: str) -> bool:
    return any(re.search(pat, sentence) for pat in BOILERPLATE_PATTERNS)


def _normalize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r"\([^)]*\)", "", name).strip()
    return name.lower()


def _is_digital_service(name: str, text: str) -> bool:
    t = f"{name} {text}".lower()
    return any(k.lower() in t for k in DIGITAL_SERVICE_KEYWORDS)


def _guess_category(name: str, text: str) -> str:
    t = f"{name} {text}".lower()
    for category, keys in CATEGORY_KEYWORDS.items():
        if any(k.lower() in t for k in keys):
            return category
    return "অন্যান্য"


def _extract_attributes(sentences: List[str]) -> Dict[str, object]:
    offers: List[str] = []
    payments: List[str] = []
    delivery: List[str] = []
    brands: List[str] = []
    warranties: List[str] = []
    rating = None

    payment_markers = [
        "ক্যাশ অন ডেলিভারি",
        "পেমেন্ট গ্রহণযোগ্য",
        "কার্ডে পেমেন্ট",
        "নগদ",
        "বিকাশ",
    ]

    delivery_markers = [
        "ডেলিভারি",
        "হোম ডেলিভারি",
        "সারা বাংলাদেশে",
    ]

    for s in sentences:
        if "অফার" in s or "বিশেষ মূল্য ছাড়" in s:
            offers.append(s)

        # Keep payment extraction strict to avoid contaminating with brand sentences
        # e.g. "বিকাশ ব্র্যান্ডের গুণগত মানের পণ্য" must NOT be treated as payment.
        if any(m in s for m in payment_markers):
            if "ব্র্যান্ডের গুণগত মানের পণ্য" not in s:
                payments.append(s)

        if any(m in s for m in delivery_markers):
            delivery.append(s)

        brand_match = BRAND_RE.search(s)
        if brand_match:
            brands.append(brand_match.group(1).strip())

        warranty_match = WARRANTY_RE.search(s)
        if warranty_match:
            warranties.append(warranty_match.group(1).strip())

        rating_match = RATING_RE.search(s)
        if rating_match:
            try:
                raw = rating_match.group(1).translate(BENGALI_DIGIT_MAP).replace("٫", ".").replace(",", ".")
                rating = float(raw)
            except ValueError:
                pass

    return {
        "offers": sorted(set(offers)),
        "payments": sorted(set(payments)),
        "delivery": sorted(set(delivery)),
        "brands": sorted(set(brands)),
        "warranties": sorted(set(warranties)),
        "rating": rating,
    }


def _parse_block(block: str) -> Dict[str, object]:
    sentences = _split_sentences(block)
    if not sentences:
        return {}

    name = sentences[0]
    if len(name) > 120:
        # defensive fallback in case malformed line becomes too long
        name = name[:120].strip()

    filtered_sentences = [s for s in sentences[1:] if not _is_boilerplate(s)]
    description = "। ".join(filtered_sentences).strip()
    if description:
        description = f"{description}।"

    attrs = _extract_attributes(sentences)
    is_digital = _is_digital_service(name, block)

    category = "ডিজিটাল সেবা" if is_digital else _guess_category(name, block)

    return {
        "name": name,
        "name_key": _normalize_name(name),
        "category": category,
        "is_digital_service": is_digital,
        "description": description or f"{name} - পণ্যের বিবরণ",
        "offers": attrs["offers"],
        "payment_methods": attrs["payments"],
        "delivery": attrs["delivery"],
        "brands": attrs["brands"],
        "warranties": attrs["warranties"],
        "rating": attrs["rating"],
    }


def _canonical_merge(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        grouped[item["name_key"]].append(item)

    merged = []
    for name_key, records in grouped.items():
        if not records:
            continue

        canonical_name = Counter(r["name"] for r in records).most_common(1)[0][0]
        canonical_category = Counter(r["category"] for r in records).most_common(1)[0][0]
        is_digital = any(bool(r.get("is_digital_service")) for r in records)

        desc_parts = []
        seen_desc = set()
        for rec in records:
            d = (rec.get("description") or "").strip()
            if d and d not in seen_desc:
                seen_desc.add(d)
                desc_parts.append(d)

        description = " ".join(desc_parts[:3]) if desc_parts else f"{canonical_name} - পণ্যের বিবরণ"

        offers = sorted({x for rec in records for x in rec.get("offers", [])})
        payment_methods = sorted({x for rec in records for x in rec.get("payment_methods", [])})
        delivery = sorted({x for rec in records for x in rec.get("delivery", [])})
        brands = sorted({x for rec in records for x in rec.get("brands", [])})
        warranties = sorted({x for rec in records for x in rec.get("warranties", [])})

        rating = next(
            (r.get("rating") for r in records if isinstance(r.get("rating"), (int, float))),
            None,
        )

        source_count = len(records)

        merged.append(
            {
                "name": canonical_name,
                "category": canonical_category,
                "price": None,
                "is_price_estimated": False,
                "unit": "ইউনিট",
                "description": description,
                "offers": offers,
                "payment_methods": payment_methods,
                "delivery": delivery,
                "brands": brands,
                "warranties": warranties,
                "rating": rating,
                "is_digital_service": is_digital,
                "source_count": source_count,
            }
        )

    merged.sort(key=lambda x: x["name"])
    for idx, item in enumerate(merged, start=1):
        item["id"] = idx
    return merged


def preprocess(input_path: Path, output_path: Path) -> Dict[str, int]:
    raw = input_path.read_text(encoding="utf-8")
    blocks = _split_blocks(raw)

    parsed: List[Dict[str, object]] = []
    for block in blocks:
        parsed_item = _parse_block(block)
        if parsed_item:
            parsed.append(parsed_item)

    merged = _canonical_merge(parsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    digital_count = sum(1 for x in merged if x.get("is_digital_service"))

    return {
        "raw_blocks": len(blocks),
        "parsed_rows": len(parsed),
        "merged_products": len(merged),
        "digital_services": digital_count,
    }


def main():
    parser = argparse.ArgumentParser(description="First-pass preprocess for Knowledge_Bank.txt")
    parser.add_argument("--input", default="data/Knowledge_Bank.txt", help="Path to Knowledge_Bank.txt")
    parser.add_argument("--output", default="data/products.json", help="Output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stats = preprocess(input_path, output_path)

    print("✅ Preprocess completed")
    print(f"- Raw blocks        : {stats['raw_blocks']}")
    print(f"- Parsed rows       : {stats['parsed_rows']}")
    print(f"- Canonical products: {stats['merged_products']}")
    print(f"- Digital services  : {stats['digital_services']}")
    print(f"- Output            : {output_path}")


if __name__ == "__main__":
    main()
