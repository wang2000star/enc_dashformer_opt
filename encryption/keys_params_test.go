package encryption

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func TestTightSpecialPrimeProfileKeepsDepth(t *testing.T) {
	legacy, err := NewHERealParameters([]int{35, 34})
	if err != nil {
		t.Fatal(err)
	}
	tight, err := NewHERealParameters([]int{31, 31})
	if err != nil {
		t.Fatal(err)
	}
	if legacy.MaxLevel() != tight.MaxLevel() {
		t.Fatalf("depth changed: legacy=%d tight=%d", legacy.MaxLevel(), tight.MaxLevel())
	}
	// Generated NTT primes are close to, but not exactly, powers of two.
	if math.Abs(tight.LogQP()-430) > 1e-3 {
		t.Fatalf("tight LogQP=%f, want 430", tight.LogQP())
	}
	if legacy.Q()[0] != tight.Q()[0] || len(legacy.Q()) != len(tight.Q()) {
		t.Fatal("tight profile unexpectedly changed the computation modulus chain")
	}
}

func TestRejectInvalidSpecialPrimeProfile(t *testing.T) {
	if _, err := NewHERealParameters(nil); err == nil {
		t.Fatal("empty LogP should fail")
	}
	if _, err := NewHERealParameters([]int{61}); err == nil {
		t.Fatal("61-bit special prime should fail")
	}
}

func TestTightSpecialPrimeKeySwitchPrecision(t *testing.T) {
	for _, profile := range []struct {
		name string
		logP []int
	}{
		{name: "tight-2P", logP: []int{31, 31}},
	} {
		t.Run(profile.name, func(t *testing.T) {
			params, err := NewHERealParameters(profile.logP)
			if err != nil {
				t.Fatal(err)
			}
			kgen := rlwe.NewKeyGenerator(params)
			sk := kgen.GenSecretKeyNew()
			pk := kgen.GenPublicKeyNew(sk)
			rlk := kgen.GenRelinearizationKeyNew(sk)
			gk := kgen.GenGaloisKeyNew(params.GaloisElementForRotation(1), sk)
			evaluator := hefloat.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(rlk, gk))
			encoder := hefloat.NewEncoder(params)

			values := make([]float64, params.MaxSlots())
			for i := range values {
				values[i] = 0.25*math.Sin(0.013*float64(i)) + 0.5
			}
			pt := hefloat.NewPlaintext(params, params.MaxLevel())
			if err := encoder.Encode(values, pt); err != nil {
				t.Fatal(err)
			}
			ct, err := rlwe.NewEncryptor(params, pk).EncryptNew(pt)
			if err != nil {
				t.Fatal(err)
			}
			// Production constructs rotations from source ciphertexts; it does not apply
			// all 44 distinct Galois keys serially to one value. Two dependent rotations
			// followed by relinearization is a representative key-switch chain.
			const rotations = 2
			for i := 0; i < rotations; i++ {
				ct, err = evaluator.RotateNew(ct, 1)
				if err != nil {
					t.Fatal(err)
				}
			}
			ct, err = evaluator.MulRelinNew(ct, ct)
			if err != nil {
				t.Fatal(err)
			}
			if err := evaluator.Rescale(ct, ct); err != nil {
				t.Fatal(err)
			}

			have := make([]float64, params.MaxSlots())
			if err := encoder.Decode(rlwe.NewDecryptor(params, sk).DecryptNew(ct), have); err != nil {
				t.Fatal(err)
			}
			maxError := 0.0
			for i, got := range have {
				want := values[(i+rotations)%len(values)]
				want *= want
				maxError = math.Max(maxError, math.Abs(got-want))
			}
			if maxError > 1e-2 {
				t.Fatalf("LogP=%v key-switch chain max error=%g exceeds 1e-2", profile.logP, maxError)
			}
			t.Logf("LogP=%v, 2 rotations + MulRelin + Rescale: max error=%g, final level=%d", profile.logP, maxError, ct.Level())
		})
	}
}
