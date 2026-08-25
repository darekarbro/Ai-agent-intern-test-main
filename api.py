"""
Optional minimal FastAPI interface (not the primary demo surface -- see cli.py).

Run with: uvicorn api:app --reload
Endpoints: POST /chat, POST /reset, GET /health
"""
from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from cli import build_agent, ROOT
from src.session import SessionStore

load_dotenv(os.path.join(ROOT, ".env"))
app = FastAPI(title="Aster & Row Support Agent")
agent = build_agent()
sessions = SessionStore()


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"ok": True, "agent_configured": agent.available}


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())[:8]
    history = sessions.history_as_text(session_id)
    resp = agent.handle_message(session_id, req.message, history)
    sessions.add_turn(session_id, "user", req.message)
    sessions.add_turn(session_id, "assistant", resp.answer)
    return {
        "session_id": session_id,
        "answer": resp.answer,
        "sources": resp.sources,
        "handoff": resp.handoff,
        "handoff_reason": resp.handoff_reason,
    }


@app.post("/reset")
def reset(req: ResetRequest):
    sessions.reset(req.session_id)
    return {"ok": True}
