"""
Conversation context manager.
Tracks full turn history + integrates with QueryRewriter for entity resolution.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from retriever.query_rewriter import QueryRewriter


@dataclass
class Turn:
    role: str           # "user" or "assistant"
    content: str
    original_query: Optional[str] = None    # user's raw query
    rewritten_query: Optional[str] = None  # after coreference resolution
    retrieved_products: List[dict] = field(default_factory=list)
    retrieval_ms: float = 0.0


class ConversationContextManager:
    """
    Manages multi-turn conversation state.
    Each session (session_id) gets its own context + query rewriter.
    """

    def __init__(self):
        self.turns: List[Turn] = []
        self.rewriter = QueryRewriter()

    def process_user_query(self, query: str) -> tuple[str, bool, str]:
        """
        Takes raw user query, returns (rewritten_query, was_rewritten, reason).
        """
        rewritten, was_rewritten, reason = self.rewriter.rewrite(query)
        return rewritten, was_rewritten, reason

    def record_user_turn(
        self,
        original_query: str,
        rewritten_query: str,
        retrieved_products: List[dict],
        retrieval_ms: float,
        was_rewritten: bool = False,
    ):
        """Record a user turn after retrieval is done."""
        # Extract entity only from self-contained turns.
        # For rewritten follow-ups (e.g., "দাম কত?"), keep existing entity to avoid drift.
        query_entity = None
        if not was_rewritten:
            query_entity = self.rewriter.extract_and_store_entity(original_query)

        # Best-effort bootstrap for very first vague self-contained turn.
        if not query_entity and not was_rewritten and not self.rewriter.last_entity and retrieved_products:
            top_product = retrieved_products[0]["name"]
            self.rewriter.update_entity_from_result(top_product)

        turn = Turn(
            role="user",
            content=rewritten_query,
            original_query=original_query,
            rewritten_query=rewritten_query,
            retrieved_products=retrieved_products,
            retrieval_ms=retrieval_ms,
        )
        self.turns.append(turn)

    def record_assistant_turn(self, content: str):
        """Record assistant response."""
        self.turns.append(Turn(role="assistant", content=content))

    def get_history_for_llm(self) -> List[dict]:
        """
        Returns conversation history formatted for LLM API calls.
        """
        history = []
        for turn in self.turns:
            history.append({"role": turn.role, "content": turn.content})
        return history

    def get_last_retrieved_context(self) -> List[dict]:
        """Get the products retrieved in the most recent user turn."""
        for turn in reversed(self.turns):
            if turn.role == "user" and turn.retrieved_products:
                return turn.retrieved_products
        return []

    def reset(self):
        self.turns = []
        self.rewriter.reset()

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    @property
    def last_entity(self):
        return self.rewriter.last_entity