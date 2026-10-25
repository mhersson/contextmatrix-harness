# contextmatrix-harness

The FSM-free engine that drives one LLM session: an agentic tool-use loop plus
the model client, tool registry, secret redaction, and event stream it needs -
and nothing about *what* the session is for.

Three backends in the ContextMatrix ecosystem drive `Run`:

- **[contextmatrix-agent](https://github.com/mhersson/contextmatrix-agent)**
  wraps it with a task FSM (`orchestrator`/`worker`) to execute board cards.
- **[contextmatrix-chat](https://github.com/mhersson/contextmatrix-chat)**
  wraps it with an interactive driver using the Inbox, compaction, and
  seeded-history surfaces to power chat sessions.
- **[contextmatrix-architect](https://github.com/mhersson/contextmatrix-architect)**
  wraps it to drive the stage runs that turn a request into board cards.

**[contextmatrix-backendkit](https://github.com/mhersson/contextmatrix-backendkit)**,
the backends' shared serve plumbing, imports `redact`.

## Packages

| Package   | Responsibility                                                              |
| --------- | --------------------------------------------------------------------------- |
| `events`  | typed event stream → stdout + JSON transcript                               |
| `llm`     | raw-HTTP OpenAI-compatible client (default OpenRouter): `Send`/`SendStream`, SSE, `models[]` failover |
| `tools`   | jailed filesystem/shell `Tool` registry, read-only subset, Skill tool        |
| `harness` | the loop: `Run`, `Verify`, `SpawnSubagents`, the `Inbox` seam, `Config`       |
| `redact`  | literal-secret masking, injected into the loop via `Config.RedactToolOutput`  |
| `tlsca`   | HTTP clients that trust an extra operator-supplied CA, stdlib only           |

## Boundary invariant

`harness` imports **only** `events`, `llm`, `tools` (+ stdlib). The module has
**zero external module dependencies** in non-test code (`testify` is the only
test dependency). CI enforces the `harness` import allowlist and rejects every
external module dependency in non-test code via `scripts/deps-gate.sh`
(`make deps-gate`).
Keep the loop free of orchestration,
protocol, transport, and policy concerns - those belong in the consuming backend.
The harness is likewise language-neutral toward the target workspace: it bakes in
no target-language tools, prompts, or defaults - toolchain specifics enter only
from the caller through its seams (system prompt, verify check, bash env, skills
mount, tool set).

## Entry points

`harness.Run`, `harness.Config`, `harness.Inbox`, `harness.Result`,
`events.Emitter`, `tools.Registry`, `tools.Tool`, `tools.Terminal`, `llm.LLM`,
`llm.Catalog`.

## Terminating tool

By default `Run` ends when the model emits a non-empty turn with no tool calls.
An empty reply (no content, reasoning or tool calls) is retried up to twice,
then fails the run with `harness.ErrEmptyResponse` (`Reason "error"`). A consumer
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
execute if they precede the terminating call. Calls that follow it are never
executed - the run has already ended - but each still receives a synthetic tool
result saying so, so the message history the run carries out stays well-formed.

Termination gates on execution: a terminating tool that returns an error, or
whose arguments fail to parse, does **not** end the run - the model receives the
error and retries. If the model ends by omission instead, `CompletionArgs` is
`nil`. Without a `Terminal` tool in the registry, only the default ending by
omission applies.

## Reasoning replay

`Config.ReplayReasoning` (off by default) sends each assistant reply's
`reasoning_details` back verbatim on later requests, on both dialects, and
`SubagentOpts.ReplayReasoning` passes the choice to children. Off, seeded
`History` is stripped of them and the wire is unchanged. The compaction
summarizer never sees them; the kept tail keeps them. They are opaque and
signed: consumers never modify or redact them, and drop them from `History` when
the model changes.

Signed reasoning validates only on the provider that produced it, so with
replay on each reply records that provider on its message
(`Message.ReasoningProvider`: kept by `json.Marshal`, never sent). Every turn
request then puts the newest details' provider first in `Provider`'s `order`,
keeps the consumer's other routing keys, and drops details from any other
provider; the consumer's `only` and `ignore` win over the pin, and a fallback
reply moves the pin. Their entries are compared verbatim (case included) with
the reported provider name, so prefer `order` over `only` and `ignore`. The compaction summarizer is never pinned, and the
endpoint dialect sends no routing object. A 400 on a request that carried a pin
or details is retried once without either and recorded as a `state_change`
`replay_dropped` event; a model whose upstream rejects the field then costs one
rejected request per turn, so leave replay off for it.

## Developing

Contributor conventions, verification commands (`make test`, `make test-race`,
`make lint`, `make deps-gate`, `make build`), and commit discipline live in
`AGENTS.md`.
