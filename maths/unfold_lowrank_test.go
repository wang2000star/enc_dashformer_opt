package maths

import (
	"dashformer/encryption"
	"math"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func TestHoistedCipherTensorRotationsMatchSequential(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParamsWithLogP([]int{31, 31})
	if err != nil {
		t.Fatal(err)
	}
	plain := make([][][]float64, 2)
	for row := range plain {
		plain[row] = make([][]float64, 50)
		for col := range plain[row] {
			plain[row][col] = []float64{
				math.Sin(float64(row*50+col) * 0.17),
				math.Cos(float64(row*50+col) * 0.11),
			}
		}
	}
	cipherTensor, err := encryption.EncryptTensorValue(publicKeys, plain)
	if err != nil {
		t.Fatal(err)
	}
	rotations := []int{-14, 0, 1, 7}
	sequentialLeft := make(map[int]*encryption.CiphertextTensor, len(rotations))
	sequentialRight := make(map[int]*encryption.CiphertextTensor, len(rotations))
	start := time.Now()
	for _, rotation := range rotations {
		sequentialLeft[rotation], sequentialRight[rotation], err = CipherTensorRotationByColsNotAddMultiThread(publicKeys.Evaluator.ShallowCopy(), cipherTensor, rotation, 1, publicKeys.Params.MaxSlots())
		if err != nil {
			t.Fatal(err)
		}
	}
	sequentialTime := time.Since(start)
	start = time.Now()
	hoistedLeft, hoistedRight, err := HoistedCipherTensorRotationsByColsNotAdd(publicKeys, cipherTensor, rotations)
	if err != nil {
		t.Fatal(err)
	}
	hoistedTime := time.Since(start)

	maxError := 0.0
	for _, rotation := range rotations {
		for _, pair := range [][2]*encryption.CiphertextTensor{{sequentialLeft[rotation], hoistedLeft[rotation]}, {sequentialRight[rotation], hoistedRight[rotation]}} {
			want, decryptErr := encryption.DecryptTensorValue(secretKeys, pair[0])
			if decryptErr != nil {
				t.Fatal(decryptErr)
			}
			got, decryptErr := encryption.DecryptTensorValue(secretKeys, pair[1])
			if decryptErr != nil {
				t.Fatal(decryptErr)
			}
			for row := range want {
				for col := range want[row] {
					for depth := range want[row][col] {
						if diff := math.Abs(got[row][col][depth] - want[row][col][depth]); diff > maxError {
							maxError = diff
						}
					}
				}
			}
		}
	}
	t.Logf("four logical rotations over two ciphertexts: sequential=%s hoisted=%s max-error=%g", sequentialTime, hoistedTime, maxError)
	if maxError > 1e-3 {
		t.Fatalf("hoisted rotations differ from sequential path: max abs error %.3g", maxError)
	}
}

func TestRotateAndReduceGiantStepsRescaleHoisted(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParamsWithLogPAndBSGS([]int{31, 31}, 50, 7, 8)
	if err != nil {
		t.Fatal(err)
	}
	const giantStep = 8
	const babyStep = 7
	giants := make([]*encryption.CiphertextTensor, giantStep)
	for giant := 0; giant < giantStep; giant++ {
		plain := make([][][]float64, 1)
		plain[0] = make([][]float64, 50)
		for col := range plain[0] {
			plain[0][col] = []float64{
				0.002 * float64((giant+1)*(col+1)),
				0.01 * math.Sin(float64((giant+1)*(col+1))),
			}
		}
		giants[giant], err = encryption.EncryptTensorValue(publicKeys, plain)
		if err != nil {
			t.Fatal(err)
		}
	}

	start := time.Now()
	referenceParts := make([]*encryption.CiphertextTensor, giantStep)
	for giant := 0; giant < giantStep; giant++ {
		referenceParts[giant], err = CiphertextTensorRotationByColsNewMultiThread(publicKeys.Evaluator.ShallowCopy(), giants[giant], babyStep*giant, 1, publicKeys.Params.MaxSlots())
		if err != nil {
			t.Fatal(err)
		}
	}
	referenceCiphertexts := make([]*rlwe.Ciphertext, giants[0].NumDepth)
	for channel := range referenceCiphertexts {
		referenceCiphertexts[channel] = referenceParts[0].Ciphertexts[channel].CopyNew()
		for giant := 1; giant < giantStep; giant++ {
			if err = publicKeys.Evaluator.Add(referenceCiphertexts[channel], referenceParts[giant].Ciphertexts[channel], referenceCiphertexts[channel]); err != nil {
				t.Fatal(err)
			}
		}
	}
	referenceTime := time.Since(start)
	reference := &encryption.CiphertextTensor{Ciphertexts: referenceCiphertexts, NumRows: 1, NumCols: 50, NumDepth: giants[0].NumDepth}

	start = time.Now()
	hoisted, err := RotateAndReduceGiantStepsRescaleHoisted(publicKeys, giants, babyStep)
	if err != nil {
		t.Fatal(err)
	}
	hoistedTime := time.Since(start)
	start = time.Now()
	rawParts := make([]*encryption.CiphertextTensor, giantStep)
	for giant := 0; giant < giantStep; giant++ {
		rawParts[giant], err = RotateGiantStepRaw(publicKeys.Evaluator.ShallowCopy(), giants[giant], babyStep*giant, publicKeys.Params.MaxSlots())
		if err != nil {
			t.Fatal(err)
		}
	}
	pipelined, err := ReduceRawGiantStepsAndRescale(publicKeys, rawParts)
	if err != nil {
		t.Fatal(err)
	}
	pipelinedTime := time.Since(start)
	if hoisted.Ciphertexts[0].Level() != reference.Ciphertexts[0].Level() {
		t.Fatalf("level mismatch: hoisted=%d reference=%d", hoisted.Ciphertexts[0].Level(), reference.Ciphertexts[0].Level())
	}
	if pipelined.Ciphertexts[0].Level() != reference.Ciphertexts[0].Level() {
		t.Fatalf("pipelined level mismatch: got=%d reference=%d", pipelined.Ciphertexts[0].Level(), reference.Ciphertexts[0].Level())
	}
	want, err := encryption.DecryptTensorValue(secretKeys, reference)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encryption.DecryptTensorValue(secretKeys, hoisted)
	if err != nil {
		t.Fatal(err)
	}
	pipelinedValues, err := encryption.DecryptTensorValue(secretKeys, pipelined)
	if err != nil {
		t.Fatal(err)
	}
	maxError := 0.0
	maxPipelinedError := 0.0
	for row := range want {
		for col := range want[row] {
			for channel := range want[row][col] {
				if diff := math.Abs(got[row][col][channel] - want[row][col][channel]); diff > maxError {
					maxError = diff
				}
				if diff := math.Abs(pipelinedValues[row][col][channel] - want[row][col][channel]); diff > maxPipelinedError {
					maxPipelinedError = diff
				}
			}
		}
	}
	t.Logf("giant reduction: legacy=%s hoisted=%s pipelined=%s max-error=%g pipelined-max-error=%g level=%d", referenceTime, hoistedTime, pipelinedTime, maxError, maxPipelinedError, hoisted.Ciphertexts[0].Level())
	if maxError > 1e-3 {
		t.Fatalf("hoisted giant reduction differs from legacy path: max abs error %.3g", maxError)
	}
	if maxPipelinedError > 1e-3 {
		t.Fatalf("pipelined giant reduction differs from legacy path: max abs error %.3g", maxPipelinedError)
	}
}

// cipherTensorMulPlainMatRectReference retains the original per-term rescale
// schedule so the optimized accumulate-then-rescale implementation can be
// checked independently.
func cipherTensorMulPlainMatRectReference(param *hefloat.Parameters, evaluator *hefloat.Evaluator, left, right *encryption.CiphertextTensor, matrix [][]float64, rotation int) *encryption.CiphertextTensor {
	outDepth := len(matrix[0])
	result := make([]*rlwe.Ciphertext, outDepth)
	rotation = (rotation + left.NumCols) % left.NumCols
	for output := 0; output < outDepth; output++ {
		acc := hefloat.NewCiphertext(*param, left.Ciphertexts[0].Degree(), left.Ciphertexts[0].Level())
		for input := 0; input < left.NumDepth; input++ {
			maskLeft, maskRight := GeneratePlainVecLeftAndRight(left.NumRows, left.NumCols, rotation, matrix[input][output])
			termLeft, err := evaluator.MulRelinNew(left.Ciphertexts[input], maskLeft)
			if err != nil {
				panic(err)
			}
			termRight, err := evaluator.MulRelinNew(right.Ciphertexts[input], maskRight)
			if err != nil {
				panic(err)
			}
			termLeft.Scale = termRight.Scale
			term, err := evaluator.AddNew(termLeft, termRight)
			if err != nil {
				panic(err)
			}
			if err = evaluator.Rescale(term, term); err != nil {
				panic(err)
			}
			term.Scale = acc.Scale
			if err = evaluator.Add(acc, term, acc); err != nil {
				panic(err)
			}
		}
		result[output] = acc
	}
	return &encryption.CiphertextTensor{Ciphertexts: result, NumRows: left.NumRows, NumCols: left.NumCols, NumDepth: outDepth}
}

func TestCipherTensorMulPlainMatRectAccumulateThenRescale(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		t.Fatal(err)
	}

	plain := make([][][]float64, 1)
	plain[0] = make([][]float64, 50)
	for col := range plain[0] {
		plain[0][col] = []float64{
			0.01 * float64(col+1),
			math.Sin(float64(col)) * 0.1,
			math.Cos(float64(col)) * 0.2,
		}
	}
	matrix := [][]float64{{0.3, -0.7}, {1.1, 0.2}, {-0.4, 0.9}}

	cipherTensor, err := encryption.EncryptTensorValue(publicKeys, plain)
	if err != nil {
		t.Fatal(err)
	}
	left, right, err := CipherTensorRotationByColsNotAddMultiThread(publicKeys.Evaluator.ShallowCopy(), cipherTensor, 1, 1, publicKeys.Params.MaxSlots())
	if err != nil {
		t.Fatal(err)
	}
	wantCipher := cipherTensorMulPlainMatRectReference(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1)
	gotCipher, err := CipherTensorMulPlainMatRectWithLeftAndRightTensorMultiThread(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1, publicKeys.Params.MaxSlots())
	if err != nil {
		t.Fatal(err)
	}
	if gotCipher.Ciphertexts[0].Level() != wantCipher.Ciphertexts[0].Level() {
		t.Fatalf("level mismatch: got %d want %d", gotCipher.Ciphertexts[0].Level(), wantCipher.Ciphertexts[0].Level())
	}

	want, err := encryption.DecryptTensorValue(secretKeys, wantCipher)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encryption.DecryptTensorValue(secretKeys, gotCipher)
	if err != nil {
		t.Fatal(err)
	}
	maxError := 0.0
	for row := range want {
		for col := range want[row] {
			for depth := range want[row][col] {
				error := math.Abs(got[row][col][depth] - want[row][col][depth])
				if error > maxError {
					maxError = error
				}
			}
		}
	}
	if maxError > 1e-5 {
		t.Fatalf("accumulate-then-rescale differs from reference: max abs error %.3g", maxError)
	}
}

func TestCipherTensorMulPlainMatRectFixedPoint(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		t.Fatal(err)
	}
	plain := make([][][]float64, 1)
	plain[0] = make([][]float64, 50)
	for col := range plain[0] {
		plain[0][col] = []float64{0.01 * float64(col+1), math.Sin(float64(col)) * 0.1, math.Cos(float64(col)) * 0.2}
	}
	matrix := [][]float64{{0.31731, -0.71219}, {1.10123, 0.21991}, {-0.42317, 0.93137}}
	cipherTensor, err := encryption.EncryptTensorValue(publicKeys, plain)
	if err != nil {
		t.Fatal(err)
	}
	left, right, err := CipherTensorRotationByColsNotAddMultiThread(publicKeys.Evaluator.ShallowCopy(), cipherTensor, 1, 1, publicKeys.Params.MaxSlots())
	if err != nil {
		t.Fatal(err)
	}
	wantCipher := cipherTensorMulPlainMatRectReference(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1)
	want, err := encryption.DecryptTensorValue(secretKeys, wantCipher)
	if err != nil {
		t.Fatal(err)
	}
	for _, bits := range []int{8, 12, 16} {
		gotCipher, err := CipherTensorMulPlainMatRectFixedPoint(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1, bits)
		if err != nil {
			t.Fatal(err)
		}
		got, err := encryption.DecryptTensorValue(secretKeys, gotCipher)
		if err != nil {
			t.Fatal(err)
		}
		maxError := 0.0
		for row := range want {
			for col := range want[row] {
				for depth := range want[row][col] {
					if error := math.Abs(got[row][col][depth] - want[row][col][depth]); error > maxError {
						maxError = error
					}
				}
			}
		}
		t.Logf("fixed-point bits=%d max abs error=%.6g", bits, maxError)
		if gotCipher.Ciphertexts[0].Level() != wantCipher.Ciphertexts[0].Level() {
			t.Fatalf("bits=%d level mismatch: got %d want %d", bits, gotCipher.Ciphertexts[0].Level(), wantCipher.Ciphertexts[0].Level())
		}
		if maxError > 0.01 {
			t.Fatalf("bits=%d fixed-point error too large: %.6g", bits, maxError)
		}
	}
}

func TestPlainVecMulCipherTensorMulPlainMatFixedPoint(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		t.Fatal(err)
	}
	plain := make([][][]float64, 1)
	plain[0] = make([][]float64, 50)
	before := make([]float64, 50)
	for col := range plain[0] {
		plain[0][col] = []float64{0.01 * float64(col+1), math.Sin(float64(col)) * 0.1, math.Cos(float64(col)) * 0.2}
		before[col] = 0.5 + 0.25*math.Sin(float64(col))
	}
	matrix := [][]float64{{0.31731, -0.71219}, {1.10123, 0.21991}, {-0.42317, 0.93137}}
	cipherTensor, err := encryption.EncryptTensorValue(publicKeys, plain)
	if err != nil {
		t.Fatal(err)
	}
	wantCipher, err := PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, cipherTensor, before, matrix)
	if err != nil {
		t.Fatal(err)
	}
	gotCipher, err := PlainVecMulCipherTensorMulPlainMatFixedPoint(publicKeys, cipherTensor, before, matrix, 12)
	if err != nil {
		t.Fatal(err)
	}
	if gotCipher.Ciphertexts[0].Level() != wantCipher.Ciphertexts[0].Level() {
		t.Fatalf("fixed-point diag-mat level mismatch: got %d want %d", gotCipher.Ciphertexts[0].Level(), wantCipher.Ciphertexts[0].Level())
	}
	want, err := encryption.DecryptTensorValue(secretKeys, wantCipher)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encryption.DecryptTensorValue(secretKeys, gotCipher)
	if err != nil {
		t.Fatal(err)
	}
	maxError := 0.0
	for row := range want {
		for col := range want[row] {
			for depth := range want[row][col] {
				if value := math.Abs(got[row][col][depth] - want[row][col][depth]); value > maxError {
					maxError = value
				}
			}
		}
	}
	t.Logf("fixed-point diag-mat bits=12 max abs error=%.6g", maxError)
	if maxError > 0.01 {
		t.Fatalf("fixed-point diag-mat error too large: %.6g", maxError)
	}
}

func TestPrecisionAwareFixedPointBits(t *testing.T) {
	if got := precisionAwareFixedPointBits(33, []float64{0, 2e-6, 3e-6}, 12); got != 7 {
		t.Fatalf("tiny diagonal effective bits: got %d want 7", got)
	}
	if got := precisionAwareFixedPointBits(33, []float64{0.0013, 0.0017}, 12); got != 12 {
		t.Fatalf("ordinary diagonal effective bits: got %d want 12", got)
	}
}

func TestCipherTensorMulPlainMatSquareAccumulateThenRescale(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		t.Fatal(err)
	}

	plain := make([][][]float64, 1)
	plain[0] = make([][]float64, 50)
	for col := range plain[0] {
		plain[0][col] = []float64{0.01 * float64(col+1), math.Sin(float64(col)) * 0.1, math.Cos(float64(col)) * 0.2}
	}
	matrix := [][]float64{{0.3, -0.7, 0.1}, {1.1, 0.2, -0.6}, {-0.4, 0.9, 0.8}}
	cipherTensor, err := encryption.EncryptTensorValue(publicKeys, plain)
	if err != nil {
		t.Fatal(err)
	}
	left, right, err := CipherTensorRotationByColsNotAddMultiThread(publicKeys.Evaluator.ShallowCopy(), cipherTensor, 1, 1, publicKeys.Params.MaxSlots())
	if err != nil {
		t.Fatal(err)
	}
	wantCipher := cipherTensorMulPlainMatRectReference(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1)
	gotCipher, err := CipherTensorMulPlainMatWithLeftAndRightTensorMultiThread(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1, publicKeys.Params.MaxSlots())
	if err != nil {
		t.Fatal(err)
	}
	if gotCipher.Ciphertexts[0].Level() != wantCipher.Ciphertexts[0].Level() {
		t.Fatalf("level mismatch: got %d want %d", gotCipher.Ciphertexts[0].Level(), wantCipher.Ciphertexts[0].Level())
	}
	want, err := encryption.DecryptTensorValue(secretKeys, wantCipher)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encryption.DecryptTensorValue(secretKeys, gotCipher)
	if err != nil {
		t.Fatal(err)
	}
	maxError := 0.0
	for row := range want {
		for col := range want[row] {
			for depth := range want[row][col] {
				if error := math.Abs(got[row][col][depth] - want[row][col][depth]); error > maxError {
					maxError = error
				}
			}
		}
	}
	if maxError > 1e-5 {
		t.Fatalf("square accumulate-then-rescale differs from reference: max abs error %.3g", maxError)
	}

	fixedCipher, err := CipherTensorMulPlainMatRectFixedPoint(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), left, right, matrix, 1, 12)
	if err != nil {
		t.Fatal(err)
	}
	if fixedCipher.Ciphertexts[0].Level() != wantCipher.Ciphertexts[0].Level() {
		t.Fatalf("square fixed-point level mismatch: got %d want %d", fixedCipher.Ciphertexts[0].Level(), wantCipher.Ciphertexts[0].Level())
	}
	fixed, err := encryption.DecryptTensorValue(secretKeys, fixedCipher)
	if err != nil {
		t.Fatal(err)
	}
	maxFixedError := 0.0
	for row := range want {
		for col := range want[row] {
			for depth := range want[row][col] {
				if error := math.Abs(fixed[row][col][depth] - want[row][col][depth]); error > maxFixedError {
					maxFixedError = error
				}
			}
		}
	}
	t.Logf("square fixed-point bits=12 max abs error=%.6g", maxFixedError)
	if maxFixedError > 0.01 {
		t.Fatalf("square fixed-point error too large: %.6g", maxFixedError)
	}
}

func TestEncodedPlainMatrixColumns(t *testing.T) {
	publicKeys, secretKeys, err := encryption.SetHERealParams()
	if err != nil {
		t.Fatal(err)
	}
	plain := [][][]float64{{
		{0.1, 0.2, 0.3},
		{0.4, -0.5, 0.6},
		{0.7, 0.8, -0.9},
	}}
	matrix := [][]float64{{0.3, -0.7, 0.1}, {1.1, 0.2, -0.6}, {-0.4, 0.9, 0.8}}
	cipherTensor, err := encryption.EncryptTensorValue(publicKeys, plain)
	if err != nil {
		t.Fatal(err)
	}
	wantCipher, err := PlainMatMultiplyCiphertextTensorThenAdd(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), matrix, cipherTensor)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodePlainMatColumnsForTensor(publicKeys.Params, publicKeys.Encoder.ShallowCopy(), matrix, cipherTensor)
	if err != nil {
		t.Fatal(err)
	}
	gotCipher, err := PlaintextsMultiplyCiphertextTensorThenAdd(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), encoded, cipherTensor)
	if err != nil {
		t.Fatal(err)
	}
	wantTensor := &encryption.CiphertextTensor{Ciphertexts: []*rlwe.Ciphertext{wantCipher}, NumRows: 1, NumCols: 3, NumDepth: 1}
	gotTensor := &encryption.CiphertextTensor{Ciphertexts: []*rlwe.Ciphertext{gotCipher}, NumRows: 1, NumCols: 3, NumDepth: 1}
	want, err := encryption.DecryptTensorValue(secretKeys, wantTensor)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encryption.DecryptTensorValue(secretKeys, gotTensor)
	if err != nil {
		t.Fatal(err)
	}
	maxError := 0.0
	for col := range want[0] {
		if error := math.Abs(got[0][col][0] - want[0][col][0]); error > maxError {
			maxError = error
		}
	}
	if gotCipher.Level() != wantCipher.Level() || maxError > 1e-5 {
		t.Fatalf("encoded cache mismatch: levels got=%d want=%d max abs error=%.3g", gotCipher.Level(), wantCipher.Level(), maxError)
	}

	wantRowsCipher, err := CiphertextTensorMultiplyPlainMatThenAdd(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), cipherTensor, matrix)
	if err != nil {
		t.Fatal(err)
	}
	encodedRows, err := EncodePlainMatRowsForTensor(publicKeys.Params, publicKeys.Encoder.ShallowCopy(), matrix, cipherTensor)
	if err != nil {
		t.Fatal(err)
	}
	gotRowsCipher, err := PlaintextsMultiplyCiphertextTensorThenAdd(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), encodedRows, cipherTensor)
	if err != nil {
		t.Fatal(err)
	}
	wantRowsTensor := &encryption.CiphertextTensor{Ciphertexts: []*rlwe.Ciphertext{wantRowsCipher}, NumRows: 1, NumCols: 3, NumDepth: 1}
	gotRowsTensor := &encryption.CiphertextTensor{Ciphertexts: []*rlwe.Ciphertext{gotRowsCipher}, NumRows: 1, NumCols: 3, NumDepth: 1}
	wantRows, err := encryption.DecryptTensorValue(secretKeys, wantRowsTensor)
	if err != nil {
		t.Fatal(err)
	}
	gotRows, err := encryption.DecryptTensorValue(secretKeys, gotRowsTensor)
	if err != nil {
		t.Fatal(err)
	}
	maxError = 0
	for col := range wantRows[0] {
		if error := math.Abs(gotRows[0][col][0] - wantRows[0][col][0]); error > maxError {
			maxError = error
		}
	}
	if gotRowsCipher.Level() != wantRowsCipher.Level() || maxError > 1e-5 {
		t.Fatalf("encoded row cache mismatch: levels got=%d want=%d max abs error=%.3g", gotRowsCipher.Level(), wantRowsCipher.Level(), maxError)
	}

	wantFused, err := publicKeys.Evaluator.AddNew(wantCipher, wantRowsCipher)
	if err != nil {
		t.Fatal(err)
	}
	gotFused, err := TwoPlaintextProductsAddThenRescale(publicKeys.Params, publicKeys.Evaluator.ShallowCopy(), encoded, cipherTensor, encodedRows, cipherTensor)
	if err != nil {
		t.Fatal(err)
	}
	wantFusedTensor := &encryption.CiphertextTensor{Ciphertexts: []*rlwe.Ciphertext{wantFused}, NumRows: 1, NumCols: 3, NumDepth: 1}
	gotFusedTensor := &encryption.CiphertextTensor{Ciphertexts: []*rlwe.Ciphertext{gotFused}, NumRows: 1, NumCols: 3, NumDepth: 1}
	wantFusedValues, err := encryption.DecryptTensorValue(secretKeys, wantFusedTensor)
	if err != nil {
		t.Fatal(err)
	}
	gotFusedValues, err := encryption.DecryptTensorValue(secretKeys, gotFusedTensor)
	if err != nil {
		t.Fatal(err)
	}
	maxError = 0
	for col := range wantFusedValues[0] {
		if error := math.Abs(gotFusedValues[0][col][0] - wantFusedValues[0][col][0]); error > maxError {
			maxError = error
		}
	}
	if gotFused.Level() != wantFused.Level() || maxError > 1e-5 {
		t.Fatalf("fused plaintext products mismatch: levels got=%d want=%d max abs error=%.3g", gotFused.Level(), wantFused.Level(), maxError)
	}
}
