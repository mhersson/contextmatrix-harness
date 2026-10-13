package tools

import (
	"os"
	"os/exec"
	"testing"
)

// requireBinary skips when the binary is absent locally, but fails under CI
// so a missing dependency can never silently disable the test.
func requireBinary(t *testing.T, ok bool, what string) {
	t.Helper()

	if ok {
		return
	}

	if os.Getenv("CI") != "" {
		t.Fatalf("%s not installed (required in CI)", what)
	}

	t.Skipf("%s not installed", what)
}

func requireRg(t *testing.T) {
	t.Helper()

	_, err := exec.LookPath("rg")
	requireBinary(t, err == nil, "ripgrep (rg)")
}

func requireFd(t *testing.T) {
	t.Helper()
	requireBinary(t, fdBinary() != "", "fd")
}

func requireFdOrRg(t *testing.T) {
	t.Helper()

	_, err := exec.LookPath("rg")
	requireBinary(t, fdBinary() != "" || err == nil, "fd or rg")
}
