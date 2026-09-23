package utils

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadExampleDataValidation(t *testing.T) {
	for _, tc := range []struct{ name, input, wantError string }{
		{"whitespace", " A  c ,1\nc\tA,0\n", ""},
		{"empty", "", "empty"},
		{"blank", "a c\n\n", "line 2"},
		{"unknown", "a x\n", "position 2"},
		{"ragged", "a c\na\n", "sequence length"},
		{"scanner limit", strings.Repeat("a", 1024*1024+1), "read sequences"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "input.list")
			if err := os.WriteFile(path, []byte(tc.input), 0600); err != nil {
				t.Fatal(err)
			}
			got, err := ReadExampleData(path, map[string]int{"a": 1, "c": 2})
			if tc.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("got %v; want %s", err, tc.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 2 || got[0][0][1] != 1 || got[0][1][2] != 1 || got[1][0][2] != 1 {
				t.Fatalf("wrong one-hot tensor: %v", got)
			}
		})
	}
}

func TestReadersReturnErrors(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing")
	if _, err := ReadExampleData(missing, map[string]int{"a": 1}); err == nil {
		t.Fatal("missing file accepted")
	}
	for _, vocab := range []map[string]int{nil, {"a": -1}, {"a": 2}, {"a": 1, "c": 1}} {
		if _, err := ReadExampleData(missing, vocab); err == nil {
			t.Fatal("invalid vocabulary accepted")
		}
	}
	if _, err := ReadModelParameterFile(missing); err == nil {
		t.Fatal("missing model accepted")
	}
}

func TestMalformedModelDoesNotPanic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "truncated.txt")
	if err := os.WriteFile(path, []byte(strings.Repeat("0\n", 129)), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := ReadFeedFowardFile(path); err == nil {
		t.Fatal("truncated feed-forward weights accepted")
	}
	for _, lines := range [][]string{nil, {""}, {"1 2", "3"}, {"NaN"}, {"Inf"}} {
		if _, err := parseMatrix(lines); err == nil {
			t.Fatalf("invalid matrix accepted: %v", lines)
		}
	}
}
