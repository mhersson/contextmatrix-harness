# contextmatrix-harness

The FSM-free engine that drives one LLM session: an agentic tool-use loop plus
the model client, tool registry, secret redaction, and event stream it needs —
and nothing about *what* the session is for.

This module is consumed by both backends in the ContextMatrix ecosystem:

- **contextmatrix-agent** wraps it with a task FSM (`orchestrator`/`worker`) to
  execute board cards.
- **contextmatrix-chat** wraps it with an interactive driver using the Inbox,
  compaction, and seeded-history surfaces to power chat sessions.

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
`events.Emitter`, `tools.Registry`, `tools.Tool`, `tools.Terminal`, `llm.LLM`,
`llm.Catalog`.

## Terminating tool

By default `Run` ends when the model emits a turn with no tool calls. A consumer
can instead let the model end the run with an explicit action: register a tool
that implements `tools.Terminal`.

```go
type Terminal interface {
	Terminal() bool
}
```

A **successful** call to such a tool ends `Run` that turn with
`Result.Completed = true`, `Result.Reason = "done"`, and the call's arguments on
`Result.CompletionArgs` (a `json.RawMessage` the consumer unmarshals into its
own type; empty arguments normalize to `{}`). Other tool calls in the same turn
execute if they precede the terminating call and are skipped if they follow it.

Termination gates on execution: a terminating tool that returns an error, or
whose arguments fail to parse, does **not** end the run — the model receives the
error and retries. If the model ends by omission instead, `CompletionArgs` is
`nil`. A registry with no `Terminal` tool behaves exactly as before.
