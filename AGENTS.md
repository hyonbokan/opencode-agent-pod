## Project Overview

opencode-agent-pod is a standalone, reusable service that runs autonomous LLM agents (via [opencode](https://opencode.ai)) behind an HTTP + SSE API. Any project can POST a prompt and get back free text or structured output, streamed live, without embedding the opencode runtime, provider keys, or an agent loop of its own.

## Status

Built and covered by the unit suite: the engine (`agent/`), the HTTP/SSE service (`pod/`), the
key-injecting proxy that keeps provider keys out of the daemon and its `Bash` children, and the
Docker image. Two consumers have been built against the pod:
[BGP-LLaMA](https://github.com/hyonbokan/BGP-LLaMA-webservice) (code-executing routing analyst) and
[ai-customs](https://github.com/hyonbokan/ai-customs) (escalation of customs-declaration checks).
Egress enforcement is a deploy-time network policy, documented in [DEPLOY.md](DEPLOY.md) and not
applied by the pod itself.

## Prototype scope (ideal vs. kept-simple)

This is a portfolio prototype, not a deployed system. The design targets a fully decoupled,
isolated deployment. **Workspace transport** now supports a network `https://` pull (a pre-signed
object-store URL is fetched over the network into the ephemeral cwd, sharing no filesystem with the
caller — the mode to use across a trust boundary); local `file://` copy remains for single-host
dev, where it assumes caller and pod share a filesystem. What is still deliberately deferred is
**egress enforcement** (documented in DEPLOY.md, not applied anywhere) and the backend↔pod network
namespace — both deployment-shaped, not design-shaped. "Complete isolation" = the `https://`
transport **plus** applying the egress policy in a real deployment. An LLM-with-`Bash` sandbox
must never share a filesystem or network with the backend.

## The core idea in one paragraph

The pod is a stateless compute primitive, not a stateful service. One request is one fully autonomous agent run: `(prompt, workspace, tools, model, schema, budget) → streamed events + a final result`. The agent loops internally with tools (read files, run code, call MCP tools) as many turns as it needs, but the pod holds nothing between requests: no cache, no conversation history, no sessions, no domain knowledge. Everything stateful and domain-specific lives in the calling services. Provider keys live inside the pod and never cross the boundary; callers get results, never credentials.

## Origin & current shape

The engine is not new; it was copied from a production-proven opencode wrapper and made standalone. Its couplings to that project (a config object, a logger, a model router, a trace hook, an MCP tool server) have been replaced by the pod's own modules, so `agent/` no longer mirrors the original import paths and is maintained as a fork. Nothing here depends on the original project.

## Repo layout

```
opencode-agent-pod/
├── agent/                  # the engine
│   ├── runner.py           #   OpencodeRunner: one autonomous run; caps, retries, concurrency
│   ├── models.py           #   OpencodeResult + RunSubtype (the response contract), ReasoningEffort, DEFAULT_TOOLS
│   ├── permissions.py      #   PermissionSpec
│   ├── mcp.py              #   config-driven MCP registry (AGENT_MCP_SERVERS)
│   ├── logger.py           #   the process logger
│   └── opencode/           #   client (one session: run_session), watcher (spend/turn caps, wrap-up,
│                           #   live sink), events (shared event consumer), results, server + daemon_pool
│                           #   (daemon lifecycle), process (tree reaping), providers (model/key map),
│                           #   catalog (bare model id → provider), permission_config
├── pod/                    # the service — FastAPI + SSE over the engine
│   ├── app.py              #   create_app: bearer auth, POST /agent/run, /health, lifespan drain
│   ├── service.py          #   budget resolution, runner assembly, the SSE generator, per-run reap
│   ├── workspace.py        #   stage workspace-by-reference → ephemeral cwd → reap
│   ├── key_proxy.py        #   key-injecting reverse proxy: real keys never enter the daemon (§7)
│   ├── schema.py           #   RunRequest + JSON-Schema response pass-through
│   ├── settings.py         #   AGENT_POD_* env config    · __main__.py: `python -m pod` (auto-loads .env)
├── tests/agent/ · tests/pod/   # unit tests (mirror agent/ and pod/)
├── Dockerfile · .dockerignore   # the deployable image (pinned opencode + the service)
├── README.md · AGENTS.md · DEPLOY.md   # usage, contributor guide, deployment + egress hardening
└── pyproject.toml · requirements*.txt · .pre-commit-config.yaml · env.example
```

The pod service (`pod/`) is a thin FastAPI + SSE layer over `OpencodeRunner`; it holds no state and bakes in no domain logic. It streams live `token`/`tool` events as a run works, keep-alive comments while it is idle, then `cost` and a terminal `done`.

## Running the tests

The project has its own `.venv`; run tooling through it. A fresh clone has no committed venv, so any Python 3.12 env with `pytest pytest-asyncio pydantic httpx fastapi uvicorn` also works.

```bash
cd opencode-agent-pod
.venv/bin/python -m pytest tests/ -q          # expect: all passed, 0 reds
```

## Decisions locked

Came out of the design discussion; do not relitigate without a reason:

1. Stateless pure-function pod — no server-held sessions, no cache, no history.
2. Single-shot autonomous runs — the internal tool loop is multi-turn; the caller interaction is one streamed request/response. Follow-ups are new single-shots with distilled prior findings threaded in by the caller.
3. Workspace passed by reference to external storage (a pointer, read-only), not uploaded bytes; the pod reads it, runs, reaps. The pod owns no cache.
4. Callers own caching, conversation history, domain logic, workspace provisioning, and end-user auth.
5. Provider keys are pod-internal config (the custody boundary), never returned.
6. Pluggable tool/MCP registry replaces the placeholder MCP stub.
7. Internal, trusted-network service first — not a public endpoint. Bearer token, per-request budget cap, guaranteed workspace reap, and Bash egress lockdown.

## Open gates

- G1 — can opencode drive a custom OpenAI-compatible provider (local vLLM Gemma / fine-tuned LLaMA)? Resolved: custom providers are declared via AGENT_CUSTOM_PROVIDERS, verified live. This gated the two known consumers' local models.
- G2 — are the local / fine-tuned models good enough at tool-calling and structured-output discipline to drive an agent loop at all? Deferred; no hardware to serve a local model.
- G3 — is read-workspace-from-storage-per-run cheap enough, or is the data large enough to force pod-local caching plus sticky routing (bending purity)?

## Conventions

- Python 3.12, Ruff for lint and format (line length 100, rules `E,W,F,I,UP,B,C4`), mypy pragmatic — see `pyproject.toml`.
- Naming fits the pod. The engine uses `agent.*`, `OpencodeRunner`, `OpencodeResult`, and `AGENT_*` env vars. No `scan`, `audit`, `sdk`, or original-project vocabulary anywhere; keep it that way.
- Keep the pod domain-free. No BGP, customs, or audit concepts in pod code. If a change teaches the pod what a "timerange" or a "declaration" is, it belongs in a caller instead.
- Docstrings and comments are self-contained and describe behaviour, not the repo. A docstring says, clearly and concisely, what the thing does; a function's docstring describes what that function does. Do not reference the design or plan docs ("DESIGN §7", "see PLAN"), and do not name other functions, variables, or modules; if the reader has to go elsewhere to understand the line, rewrite it to stand alone. Terse and plain: one line where it fits, no `Args:`/`Returns:` restating the signature. Env var names and external tools like `opencode` are fine; they are the interface, not cross-references.
- Keep it simple; this is a prototype. Do not build the shared "workspace service" or a client SDK speculatively; add them when a second consumer needs them.
- Read existing code first, and reuse before adding. Check for a helper that already does the job before writing a new one, and match the patterns already in the file.
- Favor readability over brevity. Descriptive names, no cryptic abbreviations; code should read as its own explanation. Flatten deep nesting, preferring early returns to pyramids of `if`/`try`. Keep private helpers at the bottom of the file, split them into a dedicated module once there are many, and let the larger function coordinate them.
- Error-handling posture: critical paths fail loudly, non-critical fail soft. Auth, the budget cap, workspace isolation, and daemon/workspace reaping must surface failures, not swallow them. Best-effort work such as teardown or tracing may suppress and log at warning level with a fallback, never crashing a run over it.
- Config is env-driven; never hardcode ports, model names, keys, or timeouts.
- Run Python tooling through the project venv when it exists (`.venv/bin/python -m ruff`, `... -m mypy`, `... -m pytest`), never a global `python`/`pytest`.
