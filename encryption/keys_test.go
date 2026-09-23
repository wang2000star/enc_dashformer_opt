package encryption

import (
	"reflect"
	"testing"
)

func TestRequiredRotationSteps(t *testing.T) {
	want := []int{
		-50, -49, -48, -47, -46, -45, -44, -43, -42, -36, -35,
		-29, -28, -22, -21, -15, -14, -8, -7, -1,
		1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 21, 22, 28, 29,
		35, 36, 42, 43, 49,
	}
	got := RequiredRotationSteps(50, 7, 8)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected rotations\n got: %v\nwant: %v", got, want)
	}
}

func TestDeduplicateGaloisElements(t *testing.T) {
	got := deduplicateGaloisElements([]uint64{7, 3, 7, 9, 3})
	want := []uint64{7, 3, 9}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
