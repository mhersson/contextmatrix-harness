# contextmatrix-harness

The FSM-free engine that drives one LLM session: an agentic tool-use loop plus
the model client, tool registry, secret redaction, and event stream it needs —
and nothing about *what* the session is for.

This module is consumed by both backends in the ContextMatrix ecosystem:

- **contextmatrix-agent** wraps it with a task FSM (`orchestrator`/`worker`) to
  execute board cards.
- **contextmatrix-chat** (planned) wraps it with an interactive driver to power
  chat sessions.

## Packages

| Package   | Responsibility                                                              |
| --------- | --------------------------------------------------------------------------- |
| `events`  | typed event stream → stdout + JSON transcript                               |
| `llm`     | raw-HTTP OpenAI-compatible client (default OpenRouter): `Send`/`SendStream`, SSE, `models[]` failover |
| `tools`   | jailed filesystem/shell `Tool` registry, read-only subset, Skill tool        |
| `harness` | the loop: `Run`, `Verify`, `SpawnSubagents`, the `Inbox` seam, `Config`       |
| `redact`  | literal-secret masking, injected into the loop via `Config.RedactToolOutput`  |

## Boundary invariant

`harness` imports **only** `events`, `llm`, `tools` (+ stdlib). The module as a
whole has **zero external module dependencies**. Both rules are enforced in CI by
`scripts/deps-gate.sh` (`make deps-gate`). Keep the loop free of orchestration,
protocol, transport, and policy concerns — those belong in the consuming backend.

## Entry points

`harness.Run`, `harness.Config`, `harness.Inbox`, `harness.Result`,
`events.Emitter`, `tools.Registry`, `tools.Tool`, `llm.LLM`, `llm.Catalog`.
