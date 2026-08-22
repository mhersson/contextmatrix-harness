package harness

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsConcatenatedJSON(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want bool
	}{
		{"two objects no whitespace", `{"a":1}{"a":1}`, true},
		{"pretty printed then compact", "{\n  \"a\": 1\n}{\"a\":1}", true},
		{"objects separated by whitespace", `{"a":1} {"a":1}`, true},
		{"three objects back to back", `{"a":1}{"a":1}{"a":1}`, true},
		{"single valid object", `{"a":1}`, false},
		{"single valid array", `[1,2,3]`, false},
		{"truncated object", `{"path":`, false},
		{"not json at all", `{ this is not json`, false},
		{"empty string", "", false},
		{"whitespace only", "   ", false},
		{"valid object plus trailing garbage", `{"a":1}xyz`, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isConcatenatedJSON(tt.raw))
		})
	}
}

func TestClassifyIncapable(t *testing.T) {
	const concatenated = `{"a":1}{"a":1}`

	tests := []struct {
		name       string
		window     [][]string
		wantPat    string
		wantDetail string
	}{
		{
			name:       "three turns of identical concatenated payload",
			window:     [][]string{{concatenated}, {concatenated}, {concatenated}},
			wantPat:    patternConcatenatedObjects,
			wantDetail: detailConcatenatedObjects,
		},
		{
			name: "three turns of varying malformed payloads",
			window: [][]string{
				{"{ this is not json"},
				{`{"path":`},
				{"[1,2,3]"},
			},
			wantPat:    "",
			wantDetail: "",
		},
		{
			name:       "three turns of identical non-concatenated payload",
			window:     [][]string{{"{bad"}, {"{bad"}, {"{bad"}},
			wantPat:    patternIdenticalPayload,
			wantDetail: detailIdenticalPayload,
		},
		{
			name:       "two identical turns",
			window:     [][]string{{"{bad"}, {"{bad"}},
			wantPat:    patternIdenticalPayload,
			wantDetail: detailIdenticalPayload,
		},
		{
			name:       "one turn with one payload",
			window:     [][]string{{"{bad"}},
			wantPat:    "",
			wantDetail: "",
		},
		{
			name:       "empty window",
			window:     nil,
			wantPat:    "",
			wantDetail: "",
		},
		{
			name:       "one turn's slice is nil",
			window:     [][]string{{"{bad"}, nil},
			wantPat:    "",
			wantDetail: "",
		},
		{
			name: "mixed window: one concatenated turn, one plain-malformed turn",
			window: [][]string{
				{concatenated},
				{"{bad"},
			},
			wantPat:    "",
			wantDetail: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pat, detail := classifyIncapable(tt.window)
			assert.Equal(t, tt.wantPat, pat)
			assert.Equal(t, tt.wantDetail, detail)
		})
	}
}
