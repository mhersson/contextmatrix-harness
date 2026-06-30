package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWithDialectSetsDialect(t *testing.T) {
	c := NewClient("k", WithDialect(DialectOpenAI))
	assert.Equal(t, DialectOpenAI, c.dialect)
}

func TestDefaultDialectIsOpenRouter(t *testing.T) {
	c := NewClient("k")
	assert.Equal(t, DialectOpenRouter, c.dialect)
}
