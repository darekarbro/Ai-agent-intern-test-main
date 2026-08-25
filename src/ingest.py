"""
Ingestion pipeline for the Aster & Row knowledge base.

Responsibilities:
  1. Parse YAML front matter from each knowledge-base markdown file.
  2. Chunk the document body by heading section (not fixed character count),
     so every chunk maps back to a real, citable heading.
  3. Embed each chunk and cache the result to disk so re-runs are fast and
     don't require a model / network call every time.

This module has no knowledge of retrieval ranking or generation -- it just
turns markdown files into a list of `Chunk` objects with embeddings attached.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    doc_id: str            # e.g. "01-returns-policy-current.md"
    document_id: str       # front-matter document_id, e.g. "RET-2026-01"
    title: str
    status: str            # active | superseded | draft | ...
    audience: str
    policy_authority: str
    effective_date: Optional[str]
    supersedes: Optional[str]
    superseded_by: Optional[str]
    heading_path: str      # e.g. "Returns Policy > Standard return window"
    chunk_id: str          # stable id: f"{doc_id}::{heading_path}"
    text: str
    embedding: Optional[list] = field(default=None, repr=False)

    def to_cache_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def source_label(self) -> str:
        return f"{self.doc_id} — {self.heading_path}"


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_text, body = m.group(1), m.group(2)
    try:
        meta = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, body


def _chunk_by_heading(body: str, doc_title: str) -> list[tuple[str, str]]:
    """Split the document body into (heading_path, text) chunks.

    A new chunk starts at every heading line. Text before the first heading
    (if any) is attached to the document title itself so nothing is dropped.
    The `#` (h1) title line is treated as the doc title, not its own chunk,
    since it duplicates front-matter `title`.
    """
    lines = body.split("\n")
    chunks: list[tuple[str, str]] = []
    current_path: list[str] = []
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            heading_path = " > ".join(current_path) if current_path else doc_title
            chunks.append((heading_path, text))

    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            flush()
            current_lines = []
            level, heading_text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                # Document title heading -- reset path, don't nest under it twice.
                current_path = []
                continue
            # level 2+ heading: replace deepest path segment at this depth
            depth = level - 2  # h2 -> depth 0
            current_path = current_path[:depth]
            current_path.append(heading_text)
        else:
            current_lines.append(line)
    flush()
    return chunks


def load_documents(kb_dir: str) -> list[Chunk]:
    """Parse every .md file in kb_dir into heading-based Chunks (no embeddings yet)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        title = meta.get("title", path.stem)
        status = meta.get("status", "unknown")
        audience = meta.get("audience", "unknown")
        policy_authority = meta.get("policy_authority", "unknown")
        effective_date = meta.get("effective_date")
        supersedes = meta.get("supersedes")
        superseded_by = meta.get("superseded_by")
        document_id = meta.get("document_id", path.stem)

        for heading_path, text in _chunk_by_heading(body, title):
            chunk_id = f"{path.name}::{heading_path}"
            chunks.append(
                Chunk(
                    doc_id=path.name,
                    document_id=document_id,
                    title=title,
                    status=status,
                    audience=audience,
                    policy_authority=policy_authority,
                    effective_date=str(effective_date) if effective_date else None,
                    supersedes=supersedes,
                    superseded_by=superseded_by,
                    heading_path=heading_path,
                    chunk_id=chunk_id,
                    text=text,
                )
            )
    return chunks


class EmbeddingBackend:
    """Wraps sentence-transformers when available.

    Falls back to a deterministic, dependency-free hashed bag-of-words
    embedding when sentence-transformers / its model weights can't be
    loaded (e.g. no network access to download the model). The fallback
    is purely local and reproducible so the system still runs end-to-end
    without an internet connection -- retrieval quality is lower, and this
    tradeoff is called out in the README.
    """

    FALLBACK_DIM = 512

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._mode = "fallback"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self._mode = "sentence-transformers"
        except Exception:
            self._model = None
            self._mode = "fallback"

    @property
    def mode(self) -> str:
        return self._mode

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.FALLBACK_DIM, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.FALLBACK_DIM
            sign = 1.0 if (h // self.FALLBACK_DIM) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def _corpus_fingerprint(kb_dir: str) -> str:
    """Hash of file contents + mtimes so the cache invalidates on any change."""
    h = hashlib.sha256()
    for path in sorted(Path(kb_dir).glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build_or_load_index(
    kb_dir: str,
    cache_path: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[Chunk]:
    """Return embedded chunks, using a JSON cache keyed by corpus fingerprint."""
    fingerprint = _corpus_fingerprint(kb_dir)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint and cached.get("model") == model_name:
                chunks = []
                for d in cached["chunks"]:
                    chunks.append(Chunk(**d))
                return chunks
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # fall through and rebuild

    chunks = load_documents(kb_dir)
    backend = EmbeddingBackend(model_name)
    embeddings = backend.encode([c.text for c in chunks])
    for c, emb in zip(chunks, embeddings):
        c.embedding = emb.tolist()

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": fingerprint,
                "model": model_name,
                "embedding_mode": backend.mode,
                "chunks": [c.to_cache_dict() for c in chunks],
            },
            f,
        )
    return chunks


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.ingest`
    kb = os.path.join(os.path.dirname(__file__), "..", "knowledge-base")
    cache = os.path.join(os.path.dirname(__file__), "..", ".cache", "embeddings.json")
    result = build_or_load_index(kb, cache)
    print(f"Loaded {len(result)} chunks from {kb}")
    for c in result[:5]:
        print(f"  [{c.status}] {c.doc_id} :: {c.heading_path} ({len(c.text)} chars)")
