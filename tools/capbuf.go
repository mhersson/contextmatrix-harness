package tools

import (
	"bytes"
	"os/exec"
)

// subprocessOutputCap is the in-memory ceiling for subprocess tool output. It is
// a memory backstop only; the harness's post-return HeadTail applies the semantic
// (ToolOutputMaxBytes) cap. Comfortably above any ToolOutputMaxBytes.
const subprocessOutputCap = 10 << 20

// capWriter accumulates up to limit bytes and drops the rest, recording that
// truncation occurred. Write always reports full consumption so exec's pipe
// copier keeps draining (never blocks/errors the child). limit<=0 disables it.
type capWriter struct {
	limit     int
	buf       bytes.Buffer
	truncated bool
}

func (w *capWriter) Write(p []byte) (int, error) {
	if w.limit <= 0 {
		w.buf.Write(p)

		return len(p), nil
	}

	room := max(w.limit-w.buf.Len(), 0)

	if room >= len(p) {
		w.buf.Write(p)
	} else {
		w.buf.Write(p[:room])
		w.truncated = true
	}

	return len(p), nil
}

func (w *capWriter) String() string {
	if w.truncated {
		return w.buf.String() + "\n[output truncated]"
	}

	return w.buf.String()
}

// runCombinedCapped runs cmd with stdout+stderr merged into a capped buffer,
// returning the (possibly truncated) combined output and the run error.
func runCombinedCapped(cmd *exec.Cmd) (string, error) {
	cw := &capWriter{limit: subprocessOutputCap}
	cmd.Stdout = cw
	cmd.Stderr = cw
	err := cmd.Run()

	return cw.String(), err
}
