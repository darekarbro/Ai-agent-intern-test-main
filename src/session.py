"""
In-memory, per-session conversation state.

Kept deliberately simple: a dict of session_id -> list of turns. No cross-session
sharing is possible because each session's history is only ever read/written
under its own key. `max_turns` bounds how much history is fed back into the
model so unrelated details eventually age out rather than persisting forever.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    last_order_id: str | None = None  # last order ID mentioned, for pronoun follow-ups
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, max_turns: int = 12, ttl_seconds: int | None = 3600):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        self._evict_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        session = self.get_or_create(session_id)
        session.turns.append(Turn(role=role, content=content))
        session.turns = session.turns[-self.max_turns :]
        session.updated_at = time.time()

    def set_last_order_id(self, session_id: str, order_id: str) -> None:
        session = self.get_or_create(session_id)
        session.last_order_id = order_id
        session.updated_at = time.time()

    def history_as_text(self, session_id: str, exclude_last: bool = False) -> list[dict]:
        session = self.get_or_create(session_id)
        turns = session.turns[:-1] if exclude_last and session.turns else session.turns
        return [{"role": t.role, "content": t.content} for t in turns]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        if not self.ttl_seconds:
            return
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.updated_at > self.ttl_seconds]
        for sid in expired:
            self._sessions.pop(sid, None)
