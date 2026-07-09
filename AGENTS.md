# AGENTS.md — contextmatrix-harness

The FSM-free engine that drives one LLM session: an agentic tool-use loop plus
the model client, tool registry, secret redaction, and event stream it needs.
This file is the contributor contract; `README.md` holds the design overview and
entry points.

## Boundary discipline

This module owns the loop and nothing about *what* the session is for.
Orchestration, protocol, transport, and policy belong in the consuming backends
(`contextmatrix-agent`, `contextmatrix-chat`), never here.

- `harness` imports **only** `events`, `llm`, `tools` (+ stdlib + itself).
- The module has **zero external module dependencies** in non-test code
  (`testify` is the only test dependency).
- Both rules are enforced in CI by `make deps-gate` (`scripts/deps-gate.sh`). A
  new import that breaks either one fails the gate — fix the design, not the gate.

## Coding conventions

- **Go 1.26+.** Format with `make fmt` (gofumpt); imports via goimports.
- Wrap errors with `fmt.Errorf("operation: %w", err)`. Never swallow an error.
- `context.Context` is the first parameter of every function that does I/O.
- No global state, no `init()`. Dependencies are struct fields wired by the caller.
- Interfaces live in the package that *uses* them; constructors return concrete
  types.
- Tools are jailed to a workspace root — resolve every path through
  `resolveInRoot`; never touch the filesystem outside it.
- Tests sit next to code (`bash.go` → `bash_test.go`), table-driven, `t.TempDir()`,
  `t.Helper()` in helpers. `testify/require` for fatal checks, `testify/assert`
  otherwise.
- Lint is strict (`wsl_v5`, `nlreturn`, `gosec`, `revive`, …) and `make lint` is
  the authority: blank line before `return`, no naked returns past 5 lines.

## Key domain rules

- **The loop is FSM-free.** `harness.Run` drives turns and knows nothing about
  tasks, cards, or chat. Consumers own the state machine.
- **Terminating tool.** `Run` ends on a turn with no tool calls, *or* when a tool
  implementing `tools.Terminal` succeeds — then `Result.Completed`, `Reason
  "done"`, and the call args on `Result.CompletionArgs`. Termination gates on
  execution: a terminating tool that errors does not end the run. See `README.md`
  § Terminating tool.
- **Redaction is injected, not baked in.** The loop masks output only through
  `Config.RedactToolOutput`; the `redact` package supplies the masker.
- **Single-tenant, non-adversarial trust model.** The agent is the sole actor on
  its workspace (see the TOCTOU note in `tools/jail.go`). Do not add hardening
  that assumes a hostile workspace occupant.
- **The client is OpenAI-compatible over raw HTTP** (default OpenRouter), with
  `models[]` failover in `llm.Catalog`. No SDK.

## Verification & commit discipline

Run before every commit, in order:

```bash
go fix ./...     # modern stdlib idioms
make fmt         # gofumpt
make test        # unit tests — must be clean
make test-race   # race detector — must be clean
make lint        # golangci-lint — must be clean
make deps-gate   # boundary invariant — must pass
make build       # must build
```

Fix any failure before proceeding. Never commit red.

**NEVER** commit without explicit approval from the user. No exceptions.

**NEVER** reference a plan phase or task number in a commit message.

Write **conventional commits** — `type(scope): concise description`, lowercase,
scoped to a package (`harness`, `tools`, `llm`, `events`, `redact`). Keep the
subject short; use body bullet points for the *what* and *why*, not long
paragraphs.

```
feat(harness): add tools.Terminal for explicit Run completion
fix(tools): reject empty old_string in edit before it corrupts the file
docs(harness): document the terminating-tool contract
```
