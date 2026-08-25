#!/usr/bin/env python3
"""
Aster & Row support agent -- CLI.

Usage:
    python cli.py             # interactive chat
    python cli.py --debug     # interactive chat, print the full trace each turn

Type 'exit' or 'quit' to leave, 'reset' to clear the current session's history.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import uuid

from dotenv import load_dotenv

from src.agent import Agent
from src.ingest import build_or_load_index
from src.llm_backend import build_backend
from src.logging_utils import TurnTrace, configure_logging, emit_trace
from src.session import SessionStore

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_agent() -> Agent:
    load_dotenv(os.path.join(ROOT, ".env"))
    kb_dir = os.path.join(ROOT, "knowledge-base")
    orders_path = os.path.join(ROOT, "data", "orders.json")
    cache_path = os.path.join(ROOT, ".cache", "embeddings.json")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    chunks = build_or_load_index(kb_dir, cache_path, embedding_model)
    backend = build_backend()
    return Agent(chunks, orders_path, backend=backend, embedding_model_name=embedding_model)


def print_response(resp, debug: bool) -> None:
    print(f"\nAgent: {resp.answer}")
    if resp.sources:
        labels = [f"{s.get('doc_id')}" + (f" ({s.get('heading')})" if s.get("heading") else "") for s in resp.sources]
        print(f"Sources: {', '.join(labels)}")
    if resp.handoff:
        reason = f" ({resp.handoff_reason})" if resp.handoff_reason else ""
        print(f"[Recommending human support{reason}]")
    if debug:
        print("\n--- debug trace ---")
        for tc in resp.tool_calls:
            print(f"  tool_call: {tc.name} args={tc.args} -> {tc.result_summary}")
        for chunk in resp.retrieved_chunks:
            print(f"  retrieved: [{chunk['status']}] {chunk['doc_id']} :: {chunk['heading']} (score={chunk['score']})")
        if resp.errors:
            print(f"  errors: {resp.errors}")
        print("--- end trace ---")


def main():
    parser = argparse.ArgumentParser(description="Aster & Row support agent CLI")
    parser.add_argument("--debug", action="store_true", help="Print the full trace for every turn")
    parser.add_argument("--session-id", default=None, help="Reuse a specific session ID")
    args = parser.parse_args()

    trace_path = configure_logging(os.path.join(ROOT, "logs"))
    print(f"(logging turn traces to {trace_path})")

    agent = build_agent()
    if not agent.available:
        print("WARNING: no LLM provider is configured (see .env.example for free options: "
              "Groq, Gemini, Ollama, OpenRouter, DeepSeek, Qwen, NVIDIA NIM, or Anthropic). "
              "The agent will report that it isn't configured until you set one.\n")

    sessions = SessionStore()
    session_id = args.session_id or str(uuid.uuid4())[:8]
    print(f"Aster & Row support agent. Session: {session_id}. Type 'exit' to quit, 'reset' to clear history.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message.lower() in ("exit", "quit"):
            break
        if user_message.lower() == "reset":
            sessions.reset(session_id)
            print("(session cleared)")
            continue

        history = sessions.history_as_text(session_id)
        resp = agent.handle_message(session_id, user_message, history)
        sessions.add_turn(session_id, "user", user_message)
        sessions.add_turn(session_id, "assistant", resp.answer)

        print_response(resp, args.debug)

        trace = TurnTrace(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            session_id=session_id,
            user_message=user_message,
            history_snapshot=history,
            retrieved_chunks=resp.retrieved_chunks,
            tool_calls=[{"name": tc.name, "args": tc.args, "result": tc.result_summary} for tc in resp.tool_calls],
            final_response=resp.answer,
            handoff_flag=resp.handoff,
            errors=resp.errors,
        )
        emit_trace(trace)


if __name__ == "__main__":
    sys.exit(main())
