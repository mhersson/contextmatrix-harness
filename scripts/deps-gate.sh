#!/usr/bin/env bash
# Enforces the harness boundary invariant:
#   1. the `harness` package imports only events/llm/tools (+ stdlib + itself)
#   2. non-test code depends on no module outside this one and the stdlib
# A failing `go list` fails the gate; it never passes vacuously.
set -euo pipefail

mod="github.com/mhersson/contextmatrix-harness"

# Standalone assignments so set -e aborts when go list itself fails.
harness_deps=$(go list -deps ./harness)
bad_harness=$(printf '%s\n' "${harness_deps}" \
  | grep "^${mod}/" \
  | grep -vE "/(events|llm|tools|harness)$" || true)
if [ -n "${bad_harness}" ]; then
  echo "FAIL: harness imports outside {events,llm,tools}:" >&2
  echo "${bad_harness}" >&2
  exit 1
fi

all_deps=$(go list -deps -f '{{if not .Standard}}{{.ImportPath}}{{end}}' ./...)
bad_ext=$(printf '%s\n' "${all_deps}" \
  | grep -v '^$' \
  | grep -vE "^${mod}(/|$)" || true)
if [ -n "${bad_ext}" ]; then
  echo "FAIL: external module dependency in non-test code:" >&2
  echo "${bad_ext}" >&2
  exit 1
fi

echo "deps-gate: ok"
