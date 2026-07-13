## Project Overview

opencode-agent-pod is a standalone, reusable service that runs autonomous LLM agents (via [opencode](https://opencode.ai)) behind an HTTP + SSE API. Any project can POST a prompt and get back free text or structured output, streamed live, without embedding the opencode runtime, provider keys, or an agent loop of its own.

## Status

The proven agent engine has been copied in (`agent/`), made import-clean with local shims, and renamed to fit the pod. Reworks B (permissions) and C (tool/MCP registry) are done, and a first HTTP/SSE pod service (`pod/`, FastAPI + SSE) is built and verified against real opencode (a disconnect-reap leak was found and fixed). The full suite is green (143 passed, 0 reds).

Key custody (DESIGN §7) is closed: a key-injecting proxy (`pod/key_proxy.py`) keeps every provider key out of the daemon and its Bash children, built and live-verified end to end (`scripts/key_proxy_probe.py`). The remaining Bash egress control (§11) is a deploy-time network policy ([DEPLOY.md](DEPLOY.md)), not pod code, so a Bash-enabled consumer is now safe to build. The pod is a deployable container: the `Dockerfile` bundles the pinned `opencode` 1.17.18 binary plus the service (bumped from 1.17.11 on 2026-07-13 and re-verified live via `scripts/smoke_run.py` — tool-less and workspace+`Read` runs both PASS; the 1.17.11 image was verified to boot and answer `/health`, a rebuild on 1.17.18 is pending — see PLAN.md 2026-07-13). The first consumer (BGP-LLaMA, `../BGP-LLaMA-webservice`) is now built against this pod and **verified end to end** (2026-07-12): a real question streamed from that backend through `POST /agent/run` and back, including a `Bash`-tool run that staged a `file://` workspace, ran 3 turns, and was reaped. Live token/tool SSE events are now built and live-verified (2026-07-12): the pod streams `token` (assistant-text deltas) and `tool` (name/status/input, plus output on completion) events as the agent works, ahead of the terminal `cost`/`done`, via an additive default-off event sink threaded `run_session` → the session watcher → the SSE generator (verified end to end over the socket with `scripts/pod_http_events_probe.py`). Still pending, both consumer-side in `../BGP-LLaMA-webservice`: rendering those new frames as the live step-trace, and real BGP-data staging. A second consumer (ai-customs, `../ai-customs`) ran **eval-first and is complete including review** (2026-07-12): the scripted pipeline was baselined on 30 real customs documents, one `POST /agent/run` per escalation document (same model, same output schema) handled all 8 of the valuation criticals the pipeline could flag but not resolve, and the verdicts were independently verified (values re-derived from source, arithmetic re-computed in code): 5 resolved as extraction artifacts, 3 confirmed as real under-declarations (SGS uplifts up to ×4). One prompt iteration was needed — the two-tier conclusion (pipeline default, pod escalation) is confirmed (see PLAN.md Phase "consumer 2" and `../ai-customs/eval/README.md`).

New session: read [DESIGN.md](DESIGN.md) (architecture and rationale), then [PLAN.md](PLAN.md) (work log and what is next), then this file, before writing code.

## The core idea in one paragraph

The pod is a stateless compute primitive, not a stateful service. One request is one fully autonomous agent run: `(prompt, workspace, tools, model, schema, budget) → streamed events + a final result`. The agent loops internally with tools (read files, run code, call MCP tools) as many turns as it needs, but the pod holds nothing between requests: no cache, no conversation history, no sessions, no domain knowledge. Everything stateful and domain-specific lives in the calling services. Provider keys live inside the pod and never cross the boundary; callers get results, never credentials.

## Origin & current shape

The engine is not new; it was copied from a production-proven opencode wrapper and made standalone. That wrapper carried a few couplings (a project config object, a logger, a small model-router, a trace hook, an MCP tool server), each now satisfied by a local shim at the same import path, so the copied `agent/` code runs unmodified. The shims are bridges, to be collapsed or replaced as the pod's own equivalents land (tracked in [PLAN.md](PLAN.md)). Nothing here depends on the original project.

## Repo layout

```
opencode-agent-pod/
├── agent/                  # the engine (copied, then renamed) — unmodified logic
│   ├── runner.py           #   OpencodeRunner: one autonomous run; caps, retries, concurrency
│   ├── models.py           #   OpencodeResult (the response contract) + DEFAULT_TOOLS
│   ├── permissions.py      #   PermissionSpec
│   └── opencode/           #   server (daemon lifecycle), daemon_pool, client, driver,
│                           #   events (budget/turn caps), providers (model/key map), permission_config
├── config/                 # shim → config.runner.{GLOBAL_CONCURRENCY, SERVE_STARTUP_TIMEOUT}
├── core/
│   ├── utils/logger.py     # shim → the process logger
│   ├── integrations/…      # shim → no-op trace hook
│   └── tools/mcp.py        # config-driven MCP registry (AGENT_MCP_SERVERS) — Rework C
├── llm/                    # shim → types (Provider, ReasoningEffort), model resolution, RetryConfig
├── pod/                    # the service — FastAPI + SSE over the engine
│   ├── app.py              #   create_app: bearer auth, POST /agent/run, /health, lifespan drain
│   ├── service.py          #   budget resolution, runner assembly, the SSE generator, per-run reap
│   ├── workspace.py        #   stage workspace-by-reference → ephemeral cwd → reap
│   ├── key_proxy.py        #   key-injecting reverse proxy: real keys never enter the daemon (§7)
│   ├── schema.py           #   RunRequest + JSON-Schema response pass-through
│   ├── settings.py         #   AGENT_POD_* env config    · __main__.py: `python -m pod` (auto-loads .env)
├── tests/agent/ · tests/pod/   # unit tests (mirror agent/ and pod/)
├── Dockerfile · .dockerignore   # the deployable image (pinned opencode + the service)
├── DESIGN.md · PLAN.md · AGENTS.md · DEPLOY.md   # internal docs (git-ignored — never committed)
└── pyproject.toml · requirements*.txt · .pre-commit-config.yaml · .env.example
```

The pod service (`pod/`) is a thin FastAPI + SSE layer over `OpencodeRunner`; it holds no state and bakes in no domain logic (see DESIGN §9). It streams live `token`/`tool` events as a run works, keep-alive comments while it is idle, then `cost` and a terminal `done`.

## Running the tests

The project has its own `.venv`; run tooling through it. A fresh clone has no committed venv, so any Python 3.12 env with `pytest pytest-asyncio pydantic httpx fastapi uvicorn` also works.

```bash
cd opencode-agent-pod
.venv/bin/python -m pytest tests/ -q          # expect: 143 passed, 0 reds
```

The suite is fully green. The two MCP tests in `tests/agent/opencode/test_client.py` that were red before Rework C now assert the config-driven registry (`AGENT_MCP_SERVERS`-declared servers) rather than a baked-in `mcp_server.py`. See PLAN.md (Rework C, Tests) for why they were reshaped instead of satisfied literally.

## Decisions locked

Came out of the design discussion; do not relitigate without a reason:

1. Stateless pure-function pod — no server-held sessions, no cache, no history.
2. Single-shot autonomous runs — the internal tool loop is multi-turn; the caller interaction is one streamed request/response. Follow-ups are new single-shots with distilled prior findings threaded in by the caller.
3. Workspace passed by reference to external storage (a pointer, read-only), not uploaded bytes; the pod reads it, runs, reaps. The pod owns no cache.
4. Callers own caching, conversation history, domain logic, workspace provisioning, and end-user auth.
5. Provider keys are pod-internal config (the custody boundary), never returned.
6. Pluggable tool/MCP registry replaces the placeholder MCP stub.
7. Internal, trusted-network service first — not a public endpoint. Bearer token, per-request budget cap, guaranteed workspace reap, and Bash egress lockdown.

## Open gates (see [PLAN.md](PLAN.md#phase-0--gates))

- G1 — can opencode drive a custom OpenAI-compatible provider (local vLLM Gemma / fine-tuned LLaMA)? Resolved: Rework A added custom providers via AGENT_CUSTOM_PROVIDERS, verified live. This gated the two known consumers' local models.
- G2 — are the local / fine-tuned models good enough at tool-calling and structured-output discipline to drive an agent loop at all? Deferred; no hardware to serve a local model.
- G3 — is read-workspace-from-storage-per-run cheap enough, or is the data large enough to force pod-local caching plus sticky routing (bending purity)?

## Conventions

- Python 3.12, Ruff for lint and format (line length 100, rules `E,W,F,I,UP,B,C4`), mypy pragmatic — see `pyproject.toml`.
- Naming fits the pod. The engine uses `agent.*`, `OpencodeRunner`, `OpencodeResult`, `config.runner.*`, and `AGENT_*` env vars. No `scan`, `audit`, `sdk`, or original-project vocabulary anywhere; keep it that way.
- Keep the pod domain-free. No BGP, customs, or audit concepts in pod code. If a change teaches the pod what a "timerange" or a "declaration" is, it belongs in a caller instead.
- Docstrings and comments are self-contained and describe behaviour, not the repo. A docstring says, clearly and concisely, what the thing does; a function's docstring describes what that function does. Do not reference the design or plan docs ("DESIGN §7", "see PLAN"), and do not name other functions, variables, or modules; if the reader has to go elsewhere to understand the line, rewrite it to stand alone. Terse and plain: one line where it fits, no `Args:`/`Returns:` restating the signature. Env var names and external tools like `opencode` are fine; they are the interface, not cross-references.
- Keep it simple; this is a prototype. Do not build the shared "workspace service" or a client SDK speculatively; add them when a second consumer needs them.
- Read existing code first, and reuse before adding. Check for a helper that already does the job before writing a new one, and match the patterns already in the file.
- Favor readability over brevity. Descriptive names, no cryptic abbreviations; code should read as its own explanation. Flatten deep nesting, preferring early returns to pyramids of `if`/`try`. Keep private helpers at the bottom of the file, split them into a dedicated module once there are many, and let the larger function coordinate them.
- Error-handling posture: critical paths fail loudly, non-critical fail soft. Auth, the budget cap, workspace isolation, and daemon/workspace reaping must surface failures, not swallow them. Best-effort work such as teardown or tracing may suppress and log at warning level with a fallback, never crashing a run over it.
- Config is env-driven; never hardcode ports, model names, keys, or timeouts.
- Run Python tooling through the project venv when it exists (`.venv/bin/python -m ruff`, `... -m mypy`, `... -m pytest`), never a global `python`/`pytest`.
