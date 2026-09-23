package utils

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteResultToFileCreatesDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "output")
	values := [][][]float64{{{1, 2, 3}}}
	if err := WriteResultToFile(dir, values); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "output.txt")); err != nil {
		t.Fatalf("output file was not created: %v", err)
	}
}
