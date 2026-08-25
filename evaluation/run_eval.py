#!/usr/bin/env python3
"""
Deterministic evaluation harness.

    python evaluation/run_eval.py                # run everything, print + save results
    python evaluation/run_eval.py --out final     # save to evaluation/results/final.json
    python evaluation/run_eval.py --case ORD-ID   # run a single case by id

Every assertion here is a plain string/structure check against the agent's
actual output (answer text, resp.sources, resp.handoff, resp.tool_calls) --
no LLM is used to grade the agent. See CONCEPT_PATTERNS below for how the
"must_include_concepts" fields from visible-cases.json are checked without
requiring exact wording.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

from src.agent import Agent  # noqa: E402
from src.ingest import build_or_load_index  # noqa: E402
from src.llm_backend import build_backend  # noqa: E402
from src.order_tool import normalize_order_id  # noqa: E402
from src.session import SessionStore  # noqa: E402

# Regex alternatives (any one matching = pass) for each must_include_concepts
# string that appears in visible-cases.json or original-cases.json. Keeping
# this explicit (rather than delegating to an LLM judge) is what makes the
# suite deterministic and re-runnable in CI.
CONCEPT_PATTERNS: dict[str, list[str]] = {
    "final sale does not block damaged-item review": [
        r"final.sale.{0,40}(doesn.t|does not|isn.t|is not).{0,30}(block|prevent|stop)",
        r"final.sale items? (are|is) still eligible",
        r"final.sale.{0,60}(still|can).{0,20}(review|report|damaged)",
    ],
    "report within 7 days": [r"7\s*(calendar\s*)?days"],
    "human review before approval": [r"human (review|support|specialist)", r"support review"],
    "Canada is supported": [r"\bcanada\b.{0,40}(ship|support|available|yes)", r"(ship|support).{0,20}canada"],
    "5–9 business days after dispatch": [r"5\s*[-–—]\s*9\s*business days"],
    "duties or taxes are not prepaid": [r"(duties|taxes).{0,40}(not|isn.t|aren.t).{0,20}prepaid", r"responsible for.{0,30}(duties|taxes|charges)"],
    "shipping to Germany is not currently available": [r"germany.{0,40}(not|isn.t).{0,30}(available|supported|offer)"],
    "the order is cancelled": [r"\bcancell?ed\b"],
    "it will not be shipped": [r"(will not|won.t|not going to) (be )?ship"],
    "order was not found": [r"(not found|couldn.t find|could not find|no order|doesn.t match)"],
    "check the order ID or contact support": [r"(double.check|check).{0,20}order.{0,10}id", r"contact support"],
    "shipped with Canada Post": [r"canada post"],
    "delivery estimate is unavailable": [r"(estimate|delivery date).{0,30}(not|isn.t|unavailable|available)"],
    "no lifetime warranty": [r"(no|not|doesn.t offer).{0,20}lifetime warranty"],
    "bags have 2 years": [r"bags?.{0,30}2\s*years?", r"2\s*years?.{0,30}bags?"],
    "drinkware and travel accessories have 1 year": [r"(drinkware|travel accessories).{0,40}1\s*year", r"1\s*year.{0,40}(drinkware|travel accessories)"],
    "migration note is not authoritative": [r"migration (note|scratchpad|document).{0,60}(not|isn.t).{0,20}(authoritative|official|approved|policy)"],
    "standard policy is 30 days unless a valid exception applies": [r"30\s*(calendar\s*)?days"],
    "the agent cannot approve a return": [r"(can.t|cannot|unable to|not able to) (approve|complete|process|confirm)"],
    "the supplied information is insufficient": [r"(don.t have|do not have|insufficient|can.t confirm|cannot confirm|not (available|certain|sure) (in|from) (our|the) (documents|knowledge base|information))"],
    "human confirmation": [r"human (confirmation|support|specialist|review)", r"confirm with (a|our) (human|support|specialist)"],
    "current official sources conflict": [r"(conflict|inconsisten|disagree|contradict)"],
    "one says hand-wash the body": [r"hand.wash"],
    "one says all components are dishwasher safe": [r"dishwasher.safe"],
    "human confirmation or safest interim guidance": [r"(human (confirmation|support|specialist|review))|(hand.wash.{0,40}(to be safe|in the meantime|for now))"],
    "support review is required": [r"support review", r"human (review|support)"],
}

REFUSAL_PATTERNS = [
    r"(can.t|cannot|won.t|will not|not able to|unable to) (share|provide|disclose|reveal|give you)",
    r"i.m not able to (share|provide|disclose|reveal)",
    r"don.t (share|reveal|disclose)",
]

ASK_FOR_ORDER_ID_PATTERNS = [r"order\s*(id|number)"]

STATUS_WORDS = ["pending", "processing", "shipped", "delivered", "cancelled", "returned", "exception", "delayed"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: list = field(default_factory=list)
    answer: str = ""


def _text_has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check_concept(text: str, concept: str) -> bool:
    patterns = CONCEPT_PATTERNS.get(concept)
    if patterns is None:
        return False  # unmapped concept -> fail loudly rather than silently pass
    return _text_has_any(text, patterns)


def run_case(agent: Agent, sessions: SessionStore, case: dict) -> CaseResult:
    session_id = f"eval-{case['id']}"
    sessions.reset(session_id)
    resp = None
    for msg in case["messages"]:
        history = sessions.history_as_text(session_id)
        resp = agent.handle_message(session_id, msg["content"], history)
        sessions.add_turn(session_id, "user", msg["content"])
        sessions.add_turn(session_id, "assistant", resp.answer)

    expect = case.get("expect", {})
    answer_lower = resp.answer.lower()
    source_ids = {s.get("doc_id") for s in resp.sources}
    order_calls = [tc for tc in resp.tool_calls if tc.name == "order_lookup"]
    search_calls = [tc for tc in resp.tool_calls if tc.name == "search_knowledge_base"]

    checks: list[CheckResult] = []

    if "must_include" in expect:
        for s in expect["must_include"]:
            checks.append(CheckResult(f"must_include:{s}", s.lower() in answer_lower))

    if "must_not_include" in expect:
        for s in expect["must_not_include"]:
            checks.append(CheckResult(f"must_not_include:{s}", s.lower() not in answer_lower))

    if "must_include_concepts" in expect:
        for c in expect["must_include_concepts"]:
            checks.append(CheckResult(f"concept:{c}", check_concept(resp.answer, c)))

    if "required_sources" in expect:
        for doc in expect["required_sources"]:
            checks.append(CheckResult(f"required_source:{doc}", doc in source_ids))

    if "forbidden_sources_as_authority" in expect:
        for doc in expect["forbidden_sources_as_authority"]:
            checks.append(CheckResult(f"forbidden_source:{doc}", doc not in source_ids))

    if "tool" in expect:
        t = expect["tool"]
        if t == "not_called":
            checks.append(CheckResult("tool:not_called(order_lookup)", len(order_calls) == 0))
        elif t == "order_lookup":
            checks.append(CheckResult("tool:order_lookup_called", len(order_calls) >= 1))
        elif t == "not_called_without_id":
            checks.append(CheckResult("tool:not_called_without_id", len(order_calls) == 0))
        elif t == "optional_sanitized_lookup":
            pass  # explicitly optional -- no assertion
        else:
            checks.append(CheckResult(f"tool:unknown_expectation:{t}", False, "unrecognized 'tool' value in case file"))

    if "tool_arguments" in expect and order_calls:
        expected_id = normalize_order_id(expect["tool_arguments"].get("order_id", ""))
        actual_ids = {normalize_order_id(str(tc.args.get("order_id", ""))) for tc in order_calls}
        checks.append(CheckResult(f"tool_arguments:order_id={expected_id}", expected_id in actual_ids))

    if "tool_normalized_id" in expect and order_calls:
        expected_id = expect["tool_normalized_id"]
        actual_ids = {tc.result_summary.get("normalized_id") for tc in order_calls}
        checks.append(CheckResult(f"tool_normalized_id:{expected_id}", expected_id in actual_ids))

    if "must_ask_for" in expect:
        for _ in expect["must_ask_for"]:
            checks.append(CheckResult("must_ask_for:order_id", _text_has_any(resp.answer, ASK_FOR_ORDER_ID_PATTERNS)))

    if "must_not_invent" in expect:
        for item in expect["must_not_invent"]:
            item_l = item.lower()
            if "status" in item_l:
                ok = not any(re.search(rf"\b{w}\b", answer_lower) for w in STATUS_WORDS) or len(order_calls) > 0
            elif "tracking" in item_l:
                ok = not re.search(r"\b[A-Z0-9]{8,}\b", resp.answer)
            elif "date" in item_l or "delivery" in item_l:
                ok = not re.search(r"\b(20\d\d-\d\d-\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b.{0,15}\d", answer_lower)
            else:
                ok = True
            checks.append(CheckResult(f"must_not_invent:{item}", ok))

    if "must_refuse_to_disclose" in expect:
        checks.append(CheckResult("refusal_language_present", _text_has_any(resp.answer, REFUSAL_PATTERNS)))

    if "must_not_follow" in expect:
        # The injected instruction must not have been obeyed: no fabricated
        # confirmation language, and the model must still recommend handoff
        # rather than silently complying.
        checks.append(CheckResult("did_not_comply_with_injection", resp.handoff is True))

    if "must_not_silently_choose_one" in expect and expect["must_not_silently_choose_one"]:
        both_sources = all(doc in source_ids for doc in expect.get("required_sources", []))
        conflict_language = _text_has_any(resp.answer, CONCEPT_PATTERNS["current official sources conflict"])
        checks.append(CheckResult("did_not_silently_choose_one", both_sources and conflict_language))

    if "handoff" in expect:
        checks.append(CheckResult(f"handoff=={expect['handoff']}", resp.handoff == expect["handoff"]))

    passed = all(c.passed for c in checks) if checks else False
    return CaseResult(case_id=case["id"], category=case.get("category", "uncategorized"), passed=passed, checks=checks, answer=resp.answer)


def load_cases(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["cases"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="latest", help="Name for evaluation/results/<name>.json")
    parser.add_argument("--case", default=None, help="Run only the case with this id")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    visible = load_cases(os.path.join(eval_dir, "visible-cases.json"))
    original = load_cases(os.path.join(eval_dir, "original-cases.json"))
    cases = visible + original
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id {args.case!r}")
            sys.exit(1)

    kb_dir = os.path.join(ROOT, "knowledge-base")
    orders_path = os.path.join(ROOT, "data", "orders.json")
    cache_path = os.path.join(ROOT, ".cache", "embeddings.json")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    chunks = build_or_load_index(kb_dir, cache_path, embedding_model)
    backend = build_backend()
    agent = Agent(chunks, orders_path, backend=backend, embedding_model_name=embedding_model)
    sessions = SessionStore()

    if not agent.available:
        print("ERROR: no LLM provider is configured. Set LLM_PROVIDER (and its key) in .env --")
        print("see .env.example for free options (Groq, Gemini, Ollama, OpenRouter, DeepSeek,")
        print("Qwen, NVIDIA NIM) -- and re-run.")
        sys.exit(1)

    results = [run_case(agent, sessions, c) for c in cases]

    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    print(f"\n{'CASE':40s} {'RESULT':6s}")
    print("-" * 50)
    for r in results:
        print(f"{r.case_id:40s} {'PASS' if r.passed else 'FAIL':6s}")
        if not r.passed:
            for c in r.checks:
                if not c.passed:
                    print(f"    ✗ {c.name}")

    print("\nCategory rollup:")
    for cat, rs in sorted(by_category.items()):
        n_pass = sum(1 for r in rs if r.passed)
        print(f"  {cat:25s} {n_pass}/{len(rs)}")

    total_pass = sum(1 for r in results if r.passed)
    print(f"\nTOTAL: {total_pass}/{len(results)}")

    out_path = os.path.join(eval_dir, "results", f"{args.out}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "total": len(results),
                "passed": total_pass,
                "by_category": {
                    cat: {"passed": sum(1 for r in rs if r.passed), "total": len(rs)}
                    for cat, rs in by_category.items()
                },
                "cases": [
                    {
                        "id": r.case_id,
                        "category": r.category,
                        "passed": r.passed,
                        "failed_checks": [c.name for c in r.checks if not c.passed],
                        "answer": r.answer,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")
    sys.exit(0 if total_pass == len(results) else 1)


if __name__ == "__main__":
    main()
