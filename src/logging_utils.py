"""
Structured per-turn trace logging.

Writes one JSON object per turn to a rotating debug.jsonl file, and mirrors a
concise line to stdout via the standard `logging` module. Never receives or
logs the API key, and tool results passed in here are expected to already be
sanitized (order_tool.py enforces this before the record ever reaches here).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("aster_row_agent")


def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> str:
    os.makedirs(log_dir, exist_ok=True)
    trace_path = os.path.join(log_dir, "debug.jsonl")

    logger.setLevel(level)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(trace_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return trace_path


@dataclass
class TurnTrace:
    timestamp: str
    session_id: str
    user_message: str
    history_snapshot: list = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    final_response: str = ""
    handoff_flag: bool = False
    errors: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "session_id": self.session_id,
                "user_message": self.user_message,
                "history_snapshot": self.history_snapshot,
                "retrieved_chunks": self.retrieved_chunks,
                "tool_calls": self.tool_calls,
                "final_response": self.final_response,
                "handoff_flag": self.handoff_flag,
                "errors": self.errors,
            },
            ensure_ascii=False,
        )


def emit_trace(trace: TurnTrace) -> None:
    """Write the full trace to debug.jsonl (DEBUG file handler) and a short
    summary line to stdout (INFO stream handler)."""
    logger.debug(trace.to_json())
    n_chunks = len(trace.retrieved_chunks)
    n_tools = len(trace.tool_calls)
    logger.info(
        f"session={trace.session_id} chunks={n_chunks} tool_calls={n_tools} "
        f"handoff={trace.handoff_flag} errors={len(trace.errors)}"
    )
