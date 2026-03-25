"""
Context-aware query rewriter.
Resolves coreferences using conversation history + entity tracking.

Core logic:
  Q1: "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"  → entity extracted: "নুডুলস"
  Q2: "দাম কত টাকা?"  → detected as context-dependent → rewritten: "নুডুলসের দাম কত টাকা?"
"""
import re
from typing import Optional

# Bangla pronouns / anaphoric expressions that indicate the query is context-dependent
CONTEXT_DEPENDENT_PATTERNS = [
    r"^এটার?\s",
    r"^ওটার?\s",
    r"^সেটার?\s",
    r"^এর\s",
    r"^ওর\s",
    r"^তার\s",
    r"^এই পণ্যের",
    r"^ওই পণ্যের",
    # queries that START with price/quantity without a subject
    r"^দাম\s",
    r"^দাম\?",
    r"^দাম কত",
    r"^প্রাইস\s",
    r"^প্রাইস\?",
    r"^প্রাইস কত",
    r"^কত টাকা",
    r"^কত দাম",
    r"^মূল্য কত",
    r"^কত করে",
    r"^আর কত",
    r"^এর দাম",
    r"^ওর দাম",
    r"^কি দামে",
    r"^কোথায় পাব",
    r"^কোথায় পাওয়া যায়",
    r"^কখন পাব",
    r"^আর কি কি",
    r"^কি কি আছে",
]

# Patterns to extract product/entity names from a query
# Order matters — most specific first
ENTITY_EXTRACTION_PATTERNS = [
    # "কোম্পানি কি নুডুলস বিক্রি করে" → নুডুলস  (THE KEY PATTERN)
    r"(?:কোম্পানি|দোকান|আপনারা|আপনাদের\s+\S+)\s+কি\s+(.+?)\s+(?:বিক্রি|বিক্রয়|সরবরাহ|পাওয়া)",
    # "কি নুডুলস বিক্রি" → নুডুলস
    r"কি\s+(.+?)\s+(?:বিক্রি|পাওয়া|পাই|আছে|বিক্রয়)",
    # "নুডুলস আছে কি" → নুডুলস
    r"(.+?)\s+(?:আছে|পাওয়া যায়|বিক্রি হয়)\s*(?:কি|কিনা)?",
    # "নুডুলস বিক্রি করেন" → নুডুলস
    r"(.+?)\s+(?:বিক্রি|সরবরাহ)\s+(?:করেন|করো|করে|হয়)",
]

# Words that are NOT product names (stop-words for entity extraction)
NON_ENTITY_WORDS = {
    "আপনাদের", "আপনার", "আমাদের", "আমার", "তোমার", "তাদের",
    "কোম্পানি", "দোকান", "এখানে", "এটা", "ওটা", "সেটা",
    "কি", "কিনা", "কোনো", "কোন", "কত", "কোথায়",
    "আছে", "নেই", "হয়", "করে", "করেন", "পাওয়া", "যায়",
    "বিক্রি", "সরবরাহ", "দেওয়া", "পাই", "পাব",
    "হ্যাঁ", "না", "ঠিক", "আচ্ছা",
}


class QueryRewriter:
    """
    Rule-based, zero-latency query rewriter.
    Maintains entity context from conversation history.
    """

    def __init__(self):
        self._last_entity: Optional[str] = None  # e.g. "নুডুলস"
        self._last_category: Optional[str] = None

    def rewrite(self, query: str) -> tuple[str, bool, str]:
        """
        Returns (rewritten_query, was_rewritten, reason).
        """
        query = query.strip()

        if self._is_context_dependent(query) and self._last_entity:
            rewritten = self._inject_entity(query, self._last_entity)
            return rewritten, True, f"Resolved '{query}' → entity='{self._last_entity}'"

        return query, False, "Query is self-contained"

    def extract_and_store_entity(self, query: str) -> Optional[str]:
        """
        Extract product entity from a query and store it for future coreference.
        Called AFTER retrieving results so we know what was discussed.
        """
        entity = self._extract_entity(query)
        if entity:
            self._last_entity = entity
        return entity

    def update_entity_from_result(self, product_name: str):
        """
        After retrieval, update entity with the actual matched product name.
        This is more reliable than extracting from the query.
        """
        if product_name:
            self._last_entity = product_name

    def _is_context_dependent(self, query: str) -> bool:
        for pattern in CONTEXT_DEPENDENT_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _extract_entity(self, query: str) -> Optional[str]:
        for pattern in ENTITY_EXTRACTION_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                words = candidate.split()
                # Filter out non-entity words
                filtered = [w for w in words if w not in NON_ENTITY_WORDS and len(w) > 1]
                if filtered:
                    return " ".join(filtered)
        return None

    def _inject_entity(self, query: str, entity: str) -> str:
        """
        Prepend entity to the query.
        "দাম কত টাকা?" → "নুডুলসের দাম কত টাকা?"
        """
        # If query starts with দাম/মূল্য/কত, prepend "X এর"
        price_starters = ["দাম", "মূল্য", "কত", "আর", "কোথায়", "কখন", "এর", "ওর", "price", "প্রাইস"]
        first_word = query.split()[0] if query.split() else ""

        if first_word in price_starters or query.startswith("এর") or query.startswith("ওর"):
            return f"{self._make_possessive(entity)} {query}"

        # Generic: prepend entity
        return f"{entity} - {query}"

    def _make_possessive(self, entity: str) -> str:
        """Return Bangla possessive form, e.g. নুডুলস -> নুডুলসের."""
        if not entity:
            return entity

        if entity.endswith("ের") or entity.endswith("র"):
            return entity

        return f"{entity}ের"

    def reset(self):
        self._last_entity = None
        self._last_category = None

    @property
    def last_entity(self) -> Optional[str]:
        return self._last_entity