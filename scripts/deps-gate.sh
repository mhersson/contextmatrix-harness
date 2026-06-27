#!/usr/bin/env bash
# Enforces the harness boundary invariant:
#   1. the `harness` package imports only events/llm/tools (+ stdlib + itself)
#   2. the module as a whole depends on no contextmatrix-* module
set -euo pipefail

mod="github.com/mhersson/contextmatrix-harness"

bad_harness=$(go list -deps ./harness \
  | grep "^${mod}/" \
  | grep -vE "/(events|llm|tools|harness)$" || true)
if [ -n "${bad_harness}" ]; then
  echo "FAIL: harness imports outside {events,llm,tools}:" >&2
  echo "${bad_harness}" >&2
  exit 1
fi

bad_ext=$(go list -deps ./... \
  | grep -E 'mhersson/contextmatrix-(agent|runner|protocol|githubauth|chat)' || true)
if [ -n "${bad_ext}" ]; then
  echo "FAIL: forbidden contextmatrix-* dependency:" >&2
  echo "${bad_ext}" >&2
  exit 1
fi

echo "deps-gate: ok"
