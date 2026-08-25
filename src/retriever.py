"""
Retrieval over the embedded knowledge-base chunks.

Design choices (see README for the full rationale):
  - Status precedence is applied as a *score multiplier*, not a hard filter,
    so a superseded or internal/draft document can still surface when a user
    explicitly references it (e.g. the migration-note prompt-injection case) --
    but it will be clearly tagged as non-authoritative, and the agent's system
    prompt is responsible for never treating it as instruction or sole authority.
  - Conflict detection is a heuristic: among the top active, official results,
    if two chunks from *different* documents are both strongly relevant to the
    query and contain opposing keyword signals (e.g. "dishwasher safe" vs.
    "hand-wash"), we flag a genuine conflict instead of silently picking one.
    This is intentionally simple and documented as a limitation for production
    (a real system would want an NLI-style contradiction model).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ingest import Chunk, EmbeddingBackend

STATUS_MULTIPLIER = {
    "active": 1.0,
    "superseded": 0.55,
    "draft": 0.35,
}
DEFAULT_STATUS_MULTIPLIER = 0.35

# (keyword_a, keyword_b) pairs that indicate two passages are making opposite
# claims about the same topic. Case-insensitive substring match.
CONTRADICTION_SIGNALS = [
    ("dishwasher safe", "hand-wash"),
    ("dishwasher safe", "hand washed"),
    ("45 calendar day", "30 calendar day"),
    ("60 day", "30 calendar day"),
    ("60 day", "45 calendar day"),
    ("lifetime warranty", "not offer a lifetime warranty"),
]

RELEVANCE_FLOOR = 0.15  # below this raw cosine score, treat as "not found"


@dataclass
class ScoredChunk:
    chunk: Chunk
    raw_score: float
    boosted_score: float

    @property
    def is_authoritative(self) -> bool:
        return self.chunk.status == "active" and self.chunk.policy_authority == "official"


@dataclass
class ConflictGroup:
    topic_hint: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    results: list[ScoredChunk]
    conflicts: list[ConflictGroup]
    insufficient: bool  # true if nothing cleared the relevance floor


class Retriever:
    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.backend = EmbeddingBackend(model_name)
        self._matrix = np.array([c.embedding for c in chunks], dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        q_vec = self.backend.encode([query])[0]
        # embeddings are normalized at encode time, so dot product == cosine similarity
        raw_scores = self._matrix @ q_vec

        scored = []
        for chunk, raw in zip(self.chunks, raw_scores):
            mult = STATUS_MULTIPLIER.get(chunk.status, DEFAULT_STATUS_MULTIPLIER)
            scored.append(ScoredChunk(chunk=chunk, raw_score=float(raw), boosted_score=float(raw) * mult))

        scored.sort(key=lambda s: s.boosted_score, reverse=True)
        top = scored[:top_k]

        insufficient = all(s.raw_score < RELEVANCE_FLOOR for s in top)
        conflicts = self._detect_conflicts(top)

        return RetrievalResult(query=query, results=top, conflicts=conflicts, insufficient=insufficient)

    def _detect_conflicts(self, top: list[ScoredChunk]) -> list[ConflictGroup]:
        authoritative = [s for s in top if s.is_authoritative and s.raw_score >= RELEVANCE_FLOOR]
        conflicts: list[ConflictGroup] = []
        seen_pairs = set()

        for i in range(len(authoritative)):
            for j in range(i + 1, len(authoritative)):
                a, b = authoritative[i].chunk, authoritative[j].chunk
                if a.doc_id == b.doc_id:
                    continue
                pair_key = tuple(sorted([a.chunk_id, b.chunk_id]))
                if pair_key in seen_pairs:
                    continue
                text_a, text_b = a.text.lower(), b.text.lower()
                for kw1, kw2 in CONTRADICTION_SIGNALS:
                    if (kw1 in text_a and kw2 in text_b) or (kw2 in text_a and kw1 in text_b):
                        conflicts.append(ConflictGroup(topic_hint=f"{kw1} vs {kw2}", chunks=[a, b]))
                        seen_pairs.add(pair_key)
                        break
        return conflicts


def format_sources(result: RetrievalResult) -> list[dict]:
    """Sanitized, citation-ready source list for logging/agent context."""
    out = []
    for s in result.results:
        if s.raw_score < RELEVANCE_FLOOR:
            continue
        out.append(
            {
                "doc_id": s.chunk.doc_id,
                "heading": s.chunk.heading_path,
                "status": s.chunk.status,
                "policy_authority": s.chunk.policy_authority,
                "score": round(s.raw_score, 4),
            }
        )
    return out


if __name__ == "__main__":
    import os
    from .ingest import build_or_load_index

    kb = os.path.join(os.path.dirname(__file__), "..", "knowledge-base")
    cache = os.path.join(os.path.dirname(__file__), "..", ".cache", "embeddings.json")
    chunks = build_or_load_index(kb, cache)
    r = Retriever(chunks)

    for q in [
        "How long can I return an unused backpack?",
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        "Are your bags vegan?",
    ]:
        res = r.retrieve(q)
        print(f"\nQ: {q}")
        for s in res.results[:3]:
            print(f"  {s.boosted_score:.3f} (raw {s.raw_score:.3f}) [{s.chunk.status}] {s.chunk.source_label}")
        if res.conflicts:
            print(f"  CONFLICT: {[c.topic_hint for c in res.conflicts]}")
        if res.insufficient:
            print("  INSUFFICIENT")
