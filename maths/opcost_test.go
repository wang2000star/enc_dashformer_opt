package maths

import (
	"dashformer/encryption"
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// TestOpCost measures the wall-clock cost of each CKKS operation pattern
// used in the Dashformer pipeline, under the exact production parameters
// (logN=14, LogQ=38+33*10, single goroutine).
func TestOpCost(t *testing.T) {
	publicKeys, _, err := encryption.SetHERealParams()
	if err != nil {
		panic(err)
	}
	params := *publicKeys.Params
	fmt.Printf("logN=%d, MaxLevel=%d, MaxSlots=%d, LogDefaultScale=%d\n",
		params.LogN(), params.MaxLevel(), params.MaxSlots(), params.LogDefaultScale())

	// Encrypt a dummy [100, 50, 25] tensor (same shape as X0)
	plain := make([][]float64, 100)
	for i := range plain {
		plain[i] = make([]float64, 50)
		for j := range plain[i] {
			plain[i][j] = 0.001 * float64(i+j)
		}
	}
	tensor3d := make([][][]float64, 1)
	tensor3d[0] = plain
	ctTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, tensor3d)
	if err != nil {
		panic(err)
	}
	ct := ctTensor.Ciphertexts[0]
	fmt.Printf("ct level=%d, scale=%v, degree=%d\n", ct.Level(), ct.LogScale(), ct.Degree())

	evaluator := publicKeys.Evaluator.ShallowCopy()

	vec := make([]float64, params.MaxSlots())
	for i := range vec {
		vec[i] = 0.0001 * float64(i%7)
	}

	bench := func(name string, iters int, f func()) {
		// warmup
		for i := 0; i < 2; i++ {
			f()
		}
		start := time.Now()
		for i := 0; i < iters; i++ {
			f()
		}
		elapsed := time.Since(start)
		fmt.Printf("%-40s %8.3f ms/op  (%d iters, total %s)\n",
			name, float64(elapsed.Milliseconds())/float64(iters), iters, elapsed)
	}

	// fresh ciphertexts at max level for repeated use
	ctA := ct.CopyNew()
	ctB := ct.CopyNew()

	bench("ct-pt MulRelinNew (vec)", 20, func() {
		_, err := evaluator.MulRelinNew(ctA, vec)
		if err != nil {
			panic(err)
		}
	})

	bench("ct-pt MulRelinNew+Rescale", 20, func() {
		r, err := evaluator.MulRelinNew(ctA, vec)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(r, r); err != nil {
			panic(err)
		}
	})

	bench("ct-pt MulThenAdd (accumulate)", 20, func() {
		acc := hefloat.NewCiphertext(params, ctA.Degree(), ctA.Level())
		if err := evaluator.MulThenAdd(ctA, vec, acc); err != nil {
			panic(err)
		}
	})

	// Dashformer currently expands a scalar matrix coefficient to a constant
	// SIMD vector before every MulThenAdd. Compare that path with Lattigo's
	// native scalar operand and with an already encoded plaintext.
	constVec := make([]float64, params.MaxSlots())
	for i := range constVec {
		constVec[i] = 0.123456789
	}
	bench("MulThenAdd+Rescale constant vector", 30, func() {
		acc := hefloat.NewCiphertext(params, ctA.Degree(), ctA.Level())
		if err := evaluator.MulThenAdd(ctA, constVec, acc); err != nil {
			panic(err)
		}
		if err := evaluator.Rescale(acc, acc); err != nil {
			panic(err)
		}
	})
	bench("MulThenAdd+Rescale native scalar", 30, func() {
		acc := hefloat.NewCiphertext(params, ctA.Degree(), ctA.Level())
		if err := evaluator.MulThenAdd(ctA, 0.123456789, acc); err != nil {
			panic(err)
		}
		if err := evaluator.Rescale(acc, acc); err != nil {
			panic(err)
		}
	})
	bench("MulThenAdd native int64 fixed-point", 60, func() {
		acc := hefloat.NewCiphertext(params, ctA.Degree(), ctA.Level())
		acc.Scale = ctA.Scale
		if err := evaluator.MulThenAdd(ctA, int64(506), acc); err != nil {
			panic(err)
		}
	})

	level := ctA.Level()
	q := params.Q()[level]
	plainScale := rlwe.NewScale(new(big.Int).SetUint64(q))
	pt := hefloat.NewPlaintext(params, level)
	pt.MetaData = ctA.MetaData.CopyNew()
	pt.Scale = plainScale
	if err := publicKeys.Encoder.Encode(constVec, pt); err != nil {
		panic(err)
	}
	bench("MulThenAdd+Rescale encoded plaintext", 30, func() {
		acc := hefloat.NewCiphertext(params, ctA.Degree(), ctA.Level())
		acc.Scale = ctA.Scale.Mul(plainScale)
		if err := evaluator.MulThenAdd(ctA, pt, acc); err != nil {
			panic(err)
		}
		if err := evaluator.Rescale(acc, acc); err != nil {
			panic(err)
		}
	})

	bench("ct-ct MulRelinNew", 20, func() {
		_, err := evaluator.MulRelinNew(ctA, ctB)
		if err != nil {
			panic(err)
		}
	})

	bench("ct-ct MulRelinNew+Rescale", 20, func() {
		r, err := evaluator.MulRelinNew(ctA, ctB)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(r, r); err != nil {
			panic(err)
		}
	})

	bench("RotateNew", 20, func() {
		_, err := evaluator.RotateNew(ctA, 7)
		if err != nil {
			panic(err)
		}
	})

	bench("Rescale", 40, func() {
		r := ctA.CopyNew()
		if err := evaluator.Rescale(r, r); err != nil {
			panic(err)
		}
	})

	bench("AddNew (ct+ct)", 100, func() {
		_, err := evaluator.AddNew(ctA, ctB)
		if err != nil {
			panic(err)
		}
	})

	bench("AddNew (ct+vec)", 40, func() {
		_, err := evaluator.AddNew(ctA, vec)
		if err != nil {
			panic(err)
		}
	})

	bench("InnerSum (1,50)", 10, func() {
		r := ctA.CopyNew()
		evaluator.InnerSum(r, 1, 50, r)
	})

	// Composite patterns actually used in the pipeline ---------------------

	bench("WQWKT pair: 2xMulRelin+Add+Rescale", 15, func() {
		l, err := evaluator.MulRelinNew(ctA, vec)
		if err != nil {
			panic(err)
		}
		r, err := evaluator.MulRelinNew(ctB, vec)
		if err != nil {
			panic(err)
		}
		l.Scale = r.Scale
		res, err := evaluator.AddNew(l, r)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(res, res); err != nil {
			panic(err)
		}
	})

	bench("RotationByCols per-depth: 2Mul+2Rot+Add+Rescale", 10, func() {
		l, err := evaluator.MulRelinNew(ctA, vec)
		if err != nil {
			panic(err)
		}
		r, err := evaluator.MulRelinNew(ctA, vec)
		if err != nil {
			panic(err)
		}
		evaluator.Rotate(l, 3, l)
		evaluator.Rotate(r, 47, r)
		res, err := evaluator.AddNew(l, r)
		if err != nil {
			panic(err)
		}
		if err = evaluator.Rescale(res, res); err != nil {
			panic(err)
		}
	})
}
