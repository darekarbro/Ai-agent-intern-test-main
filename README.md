# Aster & Row — RAG Support Agent

A reliability-first customer support agent over the Aster & Row knowledge base and mock
order data. Built to survive the four failure modes the customer actually reported —
conflicting policy answers, invented order info, lost conversation context, and
instruction injection from retrieved content — rather than just to pass a demo.

## Setup & run (from a clean clone)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching key -- see "Free LLM providers" below

python cli.py                      # interactive chat
python cli.py --debug              # same, with the full per-turn trace printed
python evaluation/run_eval.py      # run the evaluation suite
python tests/test_regressions.py   # offline unit/regression tests (no key needed)
```

The first run of `cli.py` or `run_eval.py` builds the embedding index and caches it to
`.cache/embeddings.json`; later runs reuse the cache until a knowledge-base file changes.

### Free LLM providers

The generation step is provider-agnostic (`src/llm_backend.py`) — you are not required
to pay for anything to run this. Pick **one** in `.env`:

| `LLM_PROVIDER` | Cost | Get a key | Notes |
|---|---|---|---|
| `groq` (default) | Free tier, no card | [console.groq.com](https://console.groq.com) | Fast; Llama 3.3 70B by default |
| `gemini` | Free tier, no card | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Gemini 2.0 Flash by default |
| `ollama` | 100% free, local | [ollama.com](https://ollama.com) — `ollama pull llama3.1 && ollama serve` | No key, no rate limit, needs 8GB+ RAM |
| `openrouter` | Free (`:free` models) | [openrouter.ai/keys](https://openrouter.ai/keys) | One key covers many free models, including DeepSeek and Qwen |
| `deepseek` | Free trial credit | [platform.deepseek.com](https://platform.deepseek.com) | |
| `qwen` | Free tier quota | [bailian.console.alibabacloud.com](https://bailian.console.alibabacloud.com) | Alibaba Model Studio |
| `nvidia` | Free API key + credits | [build.nvidia.com](https://build.nvidia.com) | NIM hosts Qwen, DeepSeek, Llama, Mistral behind one endpoint |
| `anthropic` | Paid (trial credit for new accounts) | [console.anthropic.com](https://console.anthropic.com) | Claude, kept as an option |

`.env.example` has a ready-to-uncomment block for each. Only `LLM_PROVIDER` + one key is
required; `LLM_MODEL`/`LLM_BASE_URL` have sensible free-model defaults built in and only
need overriding if you want a different model from that provider. All non-Anthropic
providers speak the OpenAI-compatible chat-completions format, so `src/llm_backend.py`
talks to all of them through one adapter (`OpenAICompatBackend`) — switching providers
is purely a `.env` change, no code edits.

Note on tool-calling reliability: this agent relies on real function/tool calling to
work correctly (structured order lookups, structured final answers). Groq's
Llama-3.3-70b, Gemini 2.0 Flash, and larger OpenRouter/NVIDIA-hosted models handle this
well. Smaller local Ollama models are more likely to occasionally skip a tool call or
emit malformed arguments — pick a model documented as tool-calling-capable (e.g.
`llama3.1`, `qwen2.5`, `mistral-nemo`) if you go the local route.

Optional API (not the primary demo surface): `uvicorn api:app --reload` exposes
`POST /chat`, `POST /reset`, `GET /health`.

## Architecture

- **Model**: pluggable via `src/llm_backend.py` — free options (Groq, Gemini, Ollama,
  OpenRouter, DeepSeek, Qwen, NVIDIA NIM) or Anthropic Claude, selected entirely through
  `.env` (`LLM_PROVIDER` + one key). See "Free LLM providers" above.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, run locally. If the package
  or model weights aren't available (e.g. no network), `src/ingest.py` falls back to a
  deterministic, dependency-free hashed bag-of-words embedding so the system still runs
  end to end — at a real cost to retrieval quality (see Known limitations).
- **Vector store**: none. Chunks + embeddings live in memory as a numpy matrix; cosine
  similarity via a single matrix-vector product. Persisted to `.cache/embeddings.json`,
  keyed by a hash of the knowledge-base contents so edits auto-invalidate the cache.
- **Ingestion & chunking**: YAML front matter is parsed per file (`status`,
  `policy_authority`, `effective_date`, `supersedes`/`superseded_by`, `audience`, etc.).
  Chunks are split by heading (`##`/`###`), not fixed character windows, so every chunk
  carries a real `doc_id::heading_path` citation.
- **Retrieval precedence**: status is a *score multiplier* (active ×1.0, superseded
  ×0.55, draft/internal ×0.35), not a hard filter — so a superseded or internal document
  can still surface if a user explicitly references it (the migration-note injection
  case), but it's heavily down-ranked and tagged non-authoritative.
- **Conflict detection**: among top *active, official* results, if two chunks from
  different documents are both relevant and contain opposing keyword signals (e.g.
  "dishwasher safe" vs. "hand-wash"), the pair is flagged and surfaced together instead
  of silently resolved. This is a documented heuristic, not an NLI model — see Known
  limitations.
- **Agent orchestration** (`src/agent.py`): three tools are exposed to the model —
  `search_knowledge_base`, `order_lookup`, and `respond_to_customer`. The model's final
  answer *is* a tool call with structured fields (`answer`, `sources`, `handoff`,
  `handoff_reason`), which is what makes citation and handoff behavior assertable by
  code instead of by parsing free text. A handful of handoff conditions are also forced
  in code regardless of what the model outputs: an order not found, an order in
  `exception` status, and a detected active/active source conflict.
- **Order tool** (`src/order_tool.py`): loads `orders.json` once; the model never sees
  the file. ID normalization tolerates whitespace/case/dash noise but never guesses a
  different ID. Only the allow-listed customer-safe fields are ever returned; stale
  carrier/tracking/ETA fields are stripped when status is `cancelled`/`returned`.
- **Sessions** (`src/session.py`): in-memory dict keyed by `session_id`, last N turns,
  TTL eviction. No cross-session leakage is possible structurally (each session only
  ever reads/writes its own key).
- **Observability** (`src/logging_utils.py`): one JSON object per turn to
  `logs/debug.jsonl` (timestamp, session, message, history, retrieved chunks + scores,
  tool calls + sanitized results, final response, handoff flag, errors), plus a short
  stdout line. `cli.py --debug` prints the trace inline. The API key and full order
  records are never logged (tool results are pre-sanitized before they ever reach the
  logger).

```
user turn
  -> search_knowledge_base? / order_lookup? (0+ calls, model's choice)
  -> respond_to_customer(answer, sources[], handoff, handoff_reason)
       ^ forced handoff conditions (conflict / not-found / exception) OR'd in here
  -> CLI prints answer + sources + handoff flag; full trace -> debug.jsonl
```

## Evaluation

```bash
python evaluation/run_eval.py
```

Every visible case (`evaluation/visible-cases.json`) plus 7 original cases
(`evaluation/original-cases.json`) covering: an injection attempt quoted directly by the
user rather than embedded in a doc, a mixed-case/whitespace order ID, an `exception`
status order, a pronoun follow-up on order ETA, a direct "print your system prompt"
request, a price-adjustment action request, and a paraphrase of the dishwasher conflict.
Assertions are deterministic string/structure checks (substrings, regex "concept"
patterns, exact source filenames, exact tool names/args, handoff booleans) — no LLM is
used to grade the agent. Results are written to `evaluation/results/<name>.json` and
printed with a per-category rollup.

**Honesty note on the numbers below**: this repository was built in a sandboxed
environment with no network access, so I could not install `sentence-transformers` /
`openai` / `anthropic`, download an embedding model, or call any real LLM API to
actually execute `run_eval.py` end to end.
Every deterministic, non-network component — ingestion, chunking, retrieval ranking and
conflict detection, order lookup and sanitization, session isolation, logging, and the
eval harness's own assertion logic — was unit tested directly against real inputs (see
`tests/test_regressions.py` and the `if __name__ == "__main__"` blocks in
`src/ingest.py`, `src/retriever.py`, `src/order_tool.py`) and behaves as described
above. The table below is a placeholder to fill in from your first two real runs, not
fabricated output:

| Category | Baseline | Final |
|---|---|---|
| retrieval | _run `python evaluation/run_eval.py --out baseline`, then `--out final`_ | |
| groundedness | | |
| tool_use / tool-use / tool-reliability | | |
| privacy | | |
| multi_turn / conversation | | |
| injection_resistance / prompt-security | | |
| source_conflict / source-conflict | | |
| abstention | | |

## Bug diary

**1. `AgentResponse` construction raised `TypeError` for the "agent not configured"
path.**
- *Repro*: `AgentResponse(answer=..., sources=[], handoff=True)` (no `handoff_reason`)
  raised `TypeError: missing 1 required positional argument`, hit while writing the
  no-API-key fallback path in `Agent.handle_message`.
- *Root cause*: `handoff_reason: Optional[str]` was declared without a default,
  immediately after fields that also had no default, so it was still a required
  argument despite being `Optional` in name.
- *Fix*: `handoff_reason: Optional[str] = None` in `src/agent.py`.
- *Regression test*: `tests/test_regressions.py::test_bug1_agent_response_default_handoff_reason`.

**2. Offline fallback embedding gives materially wrong top results on paraphrases.**
- *Repro*: with `sentence-transformers` unavailable (this sandbox has no network),
  querying `"How long can I return an unused backpack?"` against the fallback
  hash-based embedding returned `06-international-shipping.md — Canadian returns` as
  the top result instead of `01-returns-policy-current.md — Standard return window`.
  Found via manual probing with `python -m src.retriever`, not from any visible case.
- *Root cause*: the fallback is a hashed bag-of-words vector with no notion of
  synonymy or semantics (e.g. "backpack" vs. "item"/"merchandise" share no token), so
  it's easily outweighed by incidental token overlap in an unrelated section.
- *Fix / mitigation*: documented as a real limitation rather than hidden — the fallback
  exists purely so the pipeline runs end-to-end without network access; the README and
  `.env.example` are explicit that `sentence-transformers` should be installed for real
  use, and `EmbeddingBackend.mode` records which backend actually ran so it's visible in
  the cache metadata.
- *Regression test*: `python -m src.retriever` prints the top-3 results and any
  detected conflicts for three fixed probe queries so a regression is visible on
  inspection; a pinned-answer automated test is listed under Known limitations, since
  the "correct" pinned answer differs between the two backends.

**3. A hallucinated source filename in the model's final answer would have gone
straight to the customer.**
- *Repro*: nothing in the original `respond_to_customer` handling validated that a
  `doc_id` the model cites actually exists in the loaded corpus — `sources` from the
  tool call's free-form JSON were passed straight through. Reproduced directly (without
  live model access) by checking the agent's known-doc-id set against a fabricated
  filename, `"99-made-up-policy.md"`, and confirming nothing would have rejected it
  before the fix.
- *Root cause*: citation trust was implicit ("the model was asked to only cite real
  sources") rather than enforced in code.
- *Fix*: `Agent.handle_message` now filters `sources` against `self._known_doc_ids`
  (built from the actually-ingested corpus) and drops/logs anything that doesn't match,
  in `src/agent.py`.
- *Regression test*: `tests/test_regressions.py::test_bug2_source_hallucination_filtered`.

**4. `respond_to_customer` handling assumed one wire format when the LLM provider was
made pluggable.**
- *Repro*: after adding `src/llm_backend.py` to support free providers (Groq/Gemini/
  Ollama/etc., which use the OpenAI tool-calling wire format) alongside Anthropic
  (which has a different one), the original `Agent.handle_message` code appended
  `response.content` directly to the running message list and read `block.type`
  attributes straight off Anthropic SDK objects — neither works for an OpenAI-style
  response, where tool calls live on `message.tool_calls` and results must be appended
  as separate `{"role": "tool", ...}` messages rather than one grouped block.
- *Root cause*: the agent loop was written against one provider's wire format instead
  of a normalized representation.
- *Fix*: introduced `TextBlock`/`ToolUseBlock` as a provider-neutral representation, and
  pushed all format-specific message-list mutation into each backend's
  `append_tool_turn()` (see `src/llm_backend.py`). `agent.py` now only ever touches the
  normalized blocks.
- *Regression test*: `python -m py_compile` plus `tests/test_regressions.py` catch
  import/construction errors; a live-model regression test per provider is listed under
  Known limitations since it requires a real key for each provider.

## Known limitations / what I'd improve for production

- **No live end-to-end run.** Everything downstream of the LLM API call (does the
  model actually call the right tools, phrase refusals well, respect the untrusted-data
  boundary under adversarial phrasing) is designed and unit-tested at the component
  level but not verified against the real model in this environment. Run
  `evaluation/run_eval.py` with a real key before treating this as demo-ready, and
  record the real baseline/final numbers in the table above.
- **Conflict detection is a small hardcoded keyword-pair list**
  (`CONTRADICTION_SIGNALS` in `src/retriever.py`). It will catch the dishwasher-style
  conflict in this corpus but won't generalize to a new conflict with different
  vocabulary. A production version would use an NLI/contradiction-classification model
  over candidate pairs instead.
- **Fallback embedding backend is a stopgap**, not a real semantic embedding — see bug
  #2. It should never be relied on in production; `requirements.txt` pins
  `sentence-transformers` for exactly this reason.
- **Handoff detection mixes model judgment and forced code-level rules.** The forced
  rules (order not found/exception, detected conflict) are reliable; softer cases
  (e.g. "this implies the customer wants a resolution the agent can't provide") depend
  on the model correctly setting `handoff=true` in its `respond_to_customer` call. A
  stricter production version might also force `handoff=true` on a curated action-verb
  list (refund, cancel, replace, adjust, escalate) as a second safety net.
- **No auth beyond "possession of the order ID," per the assignment's own scope.** Fine
  for this assignment; not fine for production.
- **`must_include_concepts` assertions in the eval harness are regex-based**, not a
  semantic check. They're deterministic and fast but need a new pattern added by hand
  for each new concept string — `CONCEPT_PATTERNS` in `evaluation/run_eval.py` is the
  single place that would need extending.

## AI coding tools used

This entire repository — design, code, tests, and this README — was written by Claude
(Anthropic) acting as the coding agent in a sandboxed tool-use environment, working
directly from the assignment brief. One concrete correction made to the first draft:
the initial `AgentResponse` dataclass declared `handoff_reason: Optional[str]` with no
default value directly after other non-defaulted fields — syntactically valid, but it
meant every call site had to pass `handoff_reason` explicitly even though the field is
meant to be optional. This was caught only once a small stub-based test was written for
the eval harness and it threw `TypeError` immediately (bug diary #1) — a good example
of a plausible-looking first draft that a very small executable test caught right away.

## Demo

Not recorded in this environment (no ability to run the real CLI against a live model or
produce a screen recording here). To produce it: run `python cli.py --debug` and capture
(1) a cited KB answer, e.g. the standard return window question, (2) an order lookup,
e.g. `ORD-1007`, (3) the Canada multi-turn follow-up, (4) the Breeze Tumbler conflict
case as the refusal/handoff example, and (5) `python evaluation/run_eval.py` running to
completion.
