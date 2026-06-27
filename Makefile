.PHONY: build test test-race fmt lint deps-gate

# Kept in sync with contextmatrix-agent so CI installs the same linter release.
GOLANGCI_LINT_VERSION ?= v2.12.2

build:
	go build ./...
test:
	go test ./...
test-race:
	CGO_ENABLED=1 go test -race ./...
fmt:
	gofumpt -w .
lint:
	golangci-lint run
deps-gate:
	bash scripts/deps-gate.sh
