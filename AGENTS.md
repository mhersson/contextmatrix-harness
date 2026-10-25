# AGENTS.md - contextmatrix-harness

The FSM-free engine that drives one LLM session: an agentic tool-use loop plus
the model client, tool registry, secret redaction, and event stream it needs.
This file is the contributor contract; `README.md` holds the design overview and
entry points.

## Boundary discipline

This module owns the loop and nothing about *what* the session is for.
Orchestration, protocol, transport, and policy belong in the consuming backends
(`contextmatrix-agent`, `contextmatrix-chat`, `contextmatrix-architect`), never
here. `contextmatrix-backendkit`, their shared serve plumbing, imports `redact`.

- `harness` imports **only** `events`, `llm`, `tools` (+ stdlib + itself).
- The module has **zero external module dependencies** in non-test code
  (`testify` is the only test dependency).
- CI (`make deps-gate`, `scripts/deps-gate.sh`) enforces the `harness` import
  allowlist and rejects every external module dependency in non-test code;
  `testify` stays allowed because test imports are outside the graph. An import
  that trips the gate means fix the design, not the gate.
- tlsca is the one non-loop package: a pure-stdlib leaf utility (extra-CA HTTP
  trust) shared by the backends. It imports stdlib only.

## Coding conventions

- **Go 1.27+.** Format with `make fmt` (gofumpt); imports via goimports.
- Wrap errors with `fmt.Errorf("operation: %w", err)`. Never swallow an error.
- `context.Context` is the first parameter of every function that does I/O.
- No global state, no `init()`. Dependencies are struct fields wired by the caller.
- Interfaces live in the package that *uses* them; constructors return concrete
  types. The exceptions are the shared contracts the loop and its callers both
  implement against: `llm.LLM`, `tools.Terminal`, and `tools.ReadOnly`.
- Tools are jailed to a workspace root - resolve every path through
  `resolveInRoot`; never touch the filesystem outside it.
- Tests sit next to code (`bash.go` → `bash_test.go`), table-driven, `t.TempDir()`,
  `t.Helper()` in helpers. `testify/require` for fatal checks, `testify/assert`
  otherwise.
- Tests must exercise behavior. A test that only fails when someone rewords a
  string, renames a heading, or edits prose is not a test - do not write it, and
  delete it when found. Asserting exact text is legitimate only where the text is
  a wire contract a consumer parses; if no code reads it, no test should pin it.
- Lint is strict (`wsl_v5`, `nlreturn`, `gosec`, `revive`, …) and `make lint` is
  the authority: blank line before `return`, no naked returns past 5 lines.

## Documentation

- Documentation must always be true: every statement matches the code at
  HEAD, and a change that makes a doc line false updates it in the same
  commit.
- Document the current state - what exists now and why, not how it got here.
- Comments explain only non-obvious decisions, constraints, safety invariants,
  or workarounds. Do not narrate what code does or record change history; git
  holds history. Keep comments to one or two tight lines unless a longer
  explanation is genuinely necessary. Rewrite or delete if you find comments
  that don't follow this rule.
- Never use em-dashes; use hyphens (-).
- Never reference plan phases, task numbers, or private card IDs in doc comments.

## Key domain rules

- **The loop is FSM-free.** `harness.Run` drives turns and knows nothing about
  tasks, cards, or chat. Consumers own the state machine.
- **Language-neutral toward the target workspace.** The harness bakes in no
  target-language behavior - no language-specific tools, prompts, defaults, or
  file-type special cases. Toolchain and language specifics enter only from the
  caller through the existing seams: `Config.SystemPrompt` (`harness/harness.go`),
  the verify `Check` callback (`harness/verify.go`), `BashTool.WithExtraEnv`
  (`tools/bash.go`), the caller-mounted skills directory (`NewSkillTool`,
  `tools/skill.go`), and tool-set composition (`tools.ReadOnlyTools`,
  `tools/readonly.go`).
- **Terminating tool.** `Run` ends on a non-empty turn with no tool calls, *or*
  when a tool implementing `tools.Terminal` succeeds - then `Result.Completed`,
  `Reason "done"`, and the call args on `Result.CompletionArgs`. Termination gates
  on execution: a terminating tool that errors does not end the run. An empty
  reply (no content, reasoning or tool calls) is retried up to twice, then fails
  the run with `harness.ErrEmptyResponse` (`Reason "error"`). See `README.md`
  § Terminating tool.
- **Redaction is injected, not baked in.** The loop masks output only through
  `Config.RedactToolOutput`; the `redact` package supplies the masker.
- **Single-tenant, non-adversarial trust model.** The agent is the sole actor on
  its workspace (see the TOCTOU note in `tools/jail.go`). Do not add hardening
  that assumes a hostile workspace occupant.
- **The client is OpenAI-compatible over raw HTTP** (default OpenRouter), with
  `models[]` failover via `Request.Models` (an OpenRouter request extra). No SDK.

## Verification & commit discipline

Run before every commit, in order:

```bash
make fmt         # gofumpt
make test        # unit tests - must be clean
make test-race   # race detector - must be clean
make lint        # golangci-lint - must be clean
make deps-gate   # boundary invariant - must pass
make build       # must build
```

Fix any failure before proceeding. Never commit red.

- **Never commit to `main`.** Work on a feature branch; tasks of a plan or
  scope the user approved commit there without asking again. Never push
  without explicit user approval.
- No session links in commits or PRs.

**NEVER** reference a plan phase or task number in a commit message.

Write **conventional commits** - `type(scope): concise description`, lowercase,
scoped to a package (`harness`, `tools`, `llm`, `events`, `redact`, `tlsca`).
Keep the subject short; use body bullet points for the *what* and *why*, not
long paragraphs.

```
feat(harness): add tools.Terminal for explicit Run completion
fix(tools): reject empty old_string in edit before it corrupts the file
docs(harness): document the terminating-tool contract
```
