package maths

import (
	"dashformer/encryption"
	"dashformer/utils"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// LowRankFixedPointBits balances coefficient quantization against the CKKS
// precision left for the sparse boundary mask. At 12 bits, the real model's
// additional factorization error is below 0.035% in relative Frobenius norm.
const LowRankFixedPointBits = 12

// HoistedCipherTensorRotationsByColsNotAdd computes several logical row
// rotations of the same tensor. For every depth channel it decomposes the
// source ciphertext once and reuses that decomposition for all automorphisms.
// The returned left/right tensors are the two unmasked rotations consumed by
// the boundary-mask fusion routines below.
func HoistedCipherTensorRotationsByColsNotAdd(publicKeys *encryption.PublicParametersKeys, cipherTensor *encryption.CiphertextTensor, logicalRotations []int) (left, right map[int]*encryption.CiphertextTensor, err error) {
	if cipherTensor == nil || cipherTensor.NumCols < 1 || cipherTensor.NumDepth < 1 {
		return nil, nil, fmt.Errorf("invalid ciphertext tensor for hoisted rotations")
	}
	cols := cipherTensor.NumCols
	slots := publicKeys.Params.MaxSlots()
	type rotationPair struct{ left, right int }
	pairs := make(map[int]rotationPair, len(logicalRotations))
	actualSet := make(map[int]struct{}, 2*len(logicalRotations))
	left = make(map[int]*encryption.CiphertextTensor, len(logicalRotations))
	right = make(map[int]*encryption.CiphertextTensor, len(logicalRotations))
	for _, logical := range logicalRotations {
		if _, exists := pairs[logical]; exists {
			continue
		}
		r := logical % cols
		if r < 0 {
			r += cols
		}
		pair := rotationPair{left: r, right: (r - cols + slots) % slots}
		pairs[logical] = pair
		if pair.left != 0 {
			actualSet[pair.left] = struct{}{}
		}
		if pair.right != 0 {
			actualSet[pair.right] = struct{}{}
		}
		left[logical] = &encryption.CiphertextTensor{
			Ciphertexts: make([]*rlwe.Ciphertext, cipherTensor.NumDepth),
			NumRows:     cipherTensor.NumRows, NumCols: cols, NumDepth: cipherTensor.NumDepth,
		}
		right[logical] = &encryption.CiphertextTensor{
			Ciphertexts: make([]*rlwe.Ciphertext, cipherTensor.NumDepth),
			NumRows:     cipherTensor.NumRows, NumCols: cols, NumDepth: cipherTensor.NumDepth,
		}
	}
	actualRotations := make([]int, 0, len(actualSet))
	for rotation := range actualSet {
		actualRotations = append(actualRotations, rotation)
	}
	sort.Ints(actualRotations)

	jobs := make(chan int)
	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex
	workers := runtime.GOMAXPROCS(0)
	if workers > cipherTensor.NumDepth {
		workers = cipherTensor.NumDepth
	}
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			for depth := range jobs {
				rotated, rotateErr := evaluator.RotateHoistedNew(cipherTensor.Ciphertexts[depth], actualRotations)
				if rotateErr != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("hoisted rotations at depth %d: %w", depth, rotateErr)
					}
					errMu.Unlock()
					continue
				}
				for logical, pair := range pairs {
					if pair.left == 0 {
						left[logical].Ciphertexts[depth] = cipherTensor.Ciphertexts[depth].CopyNew()
					} else {
						left[logical].Ciphertexts[depth] = rotated[pair.left]
					}
					if pair.right == 0 {
						right[logical].Ciphertexts[depth] = cipherTensor.Ciphertexts[depth].CopyNew()
					} else {
						right[logical].Ciphertexts[depth] = rotated[pair.right]
					}
				}
			}
		}()
	}
	for depth := 0; depth < cipherTensor.NumDepth; depth++ {
		jobs <- depth
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return nil, nil, firstErr
	}
	return left, right, nil
}

/*
 * CipherTensorMulPlainMatRectWithLeftAndRightTensorMultiThread
 * 矩形版 LeftRight 明文矩阵变换:
 *   out[t] = Σ_j PlainMat[j][t] · (maskL·ctLeft[j] + maskR·ctRight[j])
 * 输入张量深度 = len(PlainMat), 输出张量深度 = len(PlainMat[0]).
 * 与方阵版相同, 消耗 1 层 (一次 rescale).
 */
func CipherTensorMulPlainMatRectWithLeftAndRightTensorMultiThread(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensorLeft, cipherTensorRight *encryption.CiphertextTensor, PlainMat [][]float64, rotNumber int, Slots int) (*encryption.CiphertextTensor, error) {

	cipherTensorRows := cipherTensorLeft.NumRows
	cipherTensorCols := cipherTensorLeft.NumCols
	inDepth := cipherTensorLeft.NumDepth

	rotNumber = (rotNumber + cipherTensorCols) % cipherTensorCols
	if rotNumber > cipherTensorCols {
		return &encryption.CiphertextTensor{}, fmt.Errorf("the rot size %d is too large, require <%d", rotNumber, cipherTensorCols)
	}
	if inDepth != len(PlainMat) {
		return &encryption.CiphertextTensor{}, fmt.Errorf("rect LeftRight: PlainMat rows %d != input depth %d", len(PlainMat), inDepth)
	}
	outDepth := len(PlainMat[0])

	newCiphertexts := make([]*rlwe.Ciphertext, outDepth)
	for t := 0; t < outDepth; t++ {
		ct := hefloat.NewCiphertext(*param, cipherTensorLeft.Ciphertexts[0].Degree(), cipherTensorLeft.Ciphertexts[0].Level())
		for j := 0; j < inDepth; j++ {
			rotLeftVector, rotRightVector := GeneratePlainVecLeftAndRight(cipherTensorRows, cipherTensorCols, rotNumber, PlainMat[j][t])
			// All products have the same input level and plaintext scale. Accumulate
			// them before rescaling: Rescale is linear, so this is mathematically
			// identical to rescaling every term and then adding, while requiring only
			// one modulus-switch per output channel instead of 2*inDepth temporary
			// ciphertexts and inDepth rescales.
			if err := evaluator.MulThenAdd(cipherTensorLeft.Ciphertexts[j], rotLeftVector, ct); err != nil {
				return nil, fmt.Errorf("rect LeftRight left multiply output %d input %d: %w", t, j, err)
			}
			if err := evaluator.MulThenAdd(cipherTensorRight.Ciphertexts[j], rotRightVector, ct); err != nil {
				return nil, fmt.Errorf("rect LeftRight right multiply output %d input %d: %w", t, j, err)
			}
		}
		if err := evaluator.Rescale(ct, ct); err != nil {
			return nil, fmt.Errorf("rect LeftRight rescale output %d: %w", t, err)
		}
		newCiphertexts[t] = ct
	}

	return &encryption.CiphertextTensor{
		Ciphertexts: newCiphertexts,
		NumRows:     cipherTensorRows,
		NumCols:     cipherTensorCols,
		NumDepth:    outDepth,
	}, nil
}

// CipherTensorMulPlainMatRectFixedPoint factors each masked coefficient as
//
//	coefficient * mask ~= round(coefficient * 2^bits) * (mask / 2^bits).
//
// The integer channel combinations do not increase the CKKS scale or consume
// a level. Therefore only two SIMD plaintext products (left/right mask) are
// needed per output channel, instead of two per input/output pair. The final
// rescale schedule and output scale are identical to the original transform.
func CipherTensorMulPlainMatRectFixedPoint(param *hefloat.Parameters, evaluator *hefloat.Evaluator, cipherTensorLeft, cipherTensorRight *encryption.CiphertextTensor, plainMat [][]float64, rotNumber, fixedPointBits int) (*encryption.CiphertextTensor, error) {
	if fixedPointBits < 1 || fixedPointBits > 24 {
		return nil, fmt.Errorf("fixed-point bits must be in [1,24], got %d", fixedPointBits)
	}
	if cipherTensorLeft.NumDepth != len(plainMat) || cipherTensorRight.NumDepth != len(plainMat) {
		return nil, fmt.Errorf("fixed-point LeftRight matrix rows=%d tensor depths=%d/%d", len(plainMat), cipherTensorLeft.NumDepth, cipherTensorRight.NumDepth)
	}
	rows, cols, inDepth := cipherTensorLeft.NumRows, cipherTensorLeft.NumCols, cipherTensorLeft.NumDepth
	rotNumber = (rotNumber + cols) % cols
	outDepth := len(plainMat[0])
	fixedScale := math.Ldexp(1, fixedPointBits)
	maskScale := math.Ldexp(1, -fixedPointBits)
	maskLeft, maskRight := GeneratePlainVecLeftAndRight(rows, cols, rotNumber, maskScale)
	result := make([]*rlwe.Ciphertext, outDepth)

	for output := 0; output < outDepth; output++ {
		leftCombination := hefloat.NewCiphertext(*param, cipherTensorLeft.Ciphertexts[0].Degree(), cipherTensorLeft.Ciphertexts[0].Level())
		rightCombination := hefloat.NewCiphertext(*param, cipherTensorRight.Ciphertexts[0].Degree(), cipherTensorRight.Ciphertexts[0].Level())
		leftCombination.Scale = cipherTensorLeft.Ciphertexts[0].Scale
		rightCombination.Scale = cipherTensorRight.Ciphertexts[0].Scale
		for input := 0; input < inDepth; input++ {
			scaled := math.Round(plainMat[input][output] * fixedScale)
			if math.IsNaN(scaled) || math.IsInf(scaled, 0) || math.Abs(scaled) > float64(1<<62) {
				return nil, fmt.Errorf("fixed-point coefficient out of range at [%d,%d]", input, output)
			}
			coefficient := int64(scaled)
			if coefficient == 0 {
				continue
			}
			if err := evaluator.MulThenAdd(cipherTensorLeft.Ciphertexts[input], coefficient, leftCombination); err != nil {
				return nil, fmt.Errorf("fixed-point left combination output %d input %d: %w", output, input, err)
			}
			if err := evaluator.MulThenAdd(cipherTensorRight.Ciphertexts[input], coefficient, rightCombination); err != nil {
				return nil, fmt.Errorf("fixed-point right combination output %d input %d: %w", output, input, err)
			}
		}

		ct := hefloat.NewCiphertext(*param, leftCombination.Degree(), leftCombination.Level())
		if err := evaluator.MulThenAdd(leftCombination, maskLeft, ct); err != nil {
			return nil, fmt.Errorf("fixed-point left mask output %d: %w", output, err)
		}
		if err := evaluator.MulThenAdd(rightCombination, maskRight, ct); err != nil {
			return nil, fmt.Errorf("fixed-point right mask output %d: %w", output, err)
		}
		if err := evaluator.Rescale(ct, ct); err != nil {
			return nil, fmt.Errorf("fixed-point rescale output %d: %w", output, err)
		}
		result[output] = ct
	}

	return &encryption.CiphertextTensor{Ciphertexts: result, NumRows: rows, NumCols: cols, NumDepth: outDepth}, nil
}

/*
 * GenerateCipherTensorRotLowRank
 * 低秩版旋转预计算 (替代 GenerateCipherTensorRot):
 *   giant-step: X0RotLeft/Right[i] (RotateNew, level 10), X0RotTensor[i] (mask 组合, level 9, diag2 用)
 *   baby-step : rotBL/rotBR[j] (RotateNew, level 10), X0TRot[j] (mask 组合, level 9, term3 用)
 * Y / ZrotB 在各头内由 rotL/R 与 rotBL/BR 融合 A / Bt 直接计算.
 */
func GenerateCipherTensorRotLowRank(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, babyStep, giantStep int) (X0RotTensor, X0TRotTensor, X0RotLeft, X0RotRight, rotBL, rotBR []*encryption.CiphertextTensor, err error) {

	X0RotLeft = make([]*encryption.CiphertextTensor, giantStep)
	X0RotRight = make([]*encryption.CiphertextTensor, giantStep)
	X0RotTensor = make([]*encryption.CiphertextTensor, giantStep)
	rotBL = make([]*encryption.CiphertextTensor, babyStep)
	rotBR = make([]*encryption.CiphertextTensor, babyStep)
	X0TRotTensor = make([]*encryption.CiphertextTensor, babyStep)

	runtime.GOMAXPROCS(4)
	var wg sync.WaitGroup

	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			l, r, err := CipherTensorRotationByColsNotAddMultiThread(evaluator, X0, -i*babyStep, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			rotX0, err := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, l, r, -i*babyStep, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			X0RotLeft[i] = l
			X0RotRight[i] = r
			X0RotTensor[i] = rotX0
		}(i)
	}
	for j := 0; j < babyStep; j++ {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			l, r, err := CipherTensorRotationByColsNotAddMultiThread(evaluator, X0, j, 1, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			rotX0T, err := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, l, r, j, publicKeys.Params.MaxSlots())
			if err != nil {
				panic(err)
			}
			rotBL[j] = l
			rotBR[j] = r
			X0TRotTensor[j] = rotX0T
		}(j)
	}
	wg.Wait()
	return X0RotTensor, X0TRotTensor, X0RotLeft, X0RotRight, rotBL, rotBR, nil
}

// GenerateCipherTensorRotLowRankHoisted is the batched-automorphism variant of
// GenerateCipherTensorRotLowRank. It preserves the same tensors and levels but
// amortizes RNS decomposition across all baby/giant rotations of each X0
// ciphertext.
func GenerateCipherTensorRotLowRankHoisted(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, babyStep, giantStep int) (X0RotTensor, X0TRotTensor, X0RotLeft, X0RotRight, rotBL, rotBR []*encryption.CiphertextTensor, err error) {
	logical := make([]int, 0, babyStep+giantStep)
	for i := 0; i < giantStep; i++ {
		logical = append(logical, -i*babyStep)
	}
	for j := 0; j < babyStep; j++ {
		logical = append(logical, j)
	}
	left, right, err := HoistedCipherTensorRotationsByColsNotAdd(publicKeys, X0, logical)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	X0RotTensor = make([]*encryption.CiphertextTensor, giantStep)
	X0TRotTensor = make([]*encryption.CiphertextTensor, babyStep)
	X0RotLeft = make([]*encryption.CiphertextTensor, giantStep)
	X0RotRight = make([]*encryption.CiphertextTensor, giantStep)
	rotBL = make([]*encryption.CiphertextTensor, babyStep)
	rotBR = make([]*encryption.CiphertextTensor, babyStep)
	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex

	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			rotation := -i * babyStep
			masked, localErr := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, left[rotation], right[rotation], rotation, publicKeys.Params.MaxSlots())
			if localErr != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = localErr
				}
				errMu.Unlock()
				return
			}
			X0RotLeft[i], X0RotRight[i], X0RotTensor[i] = left[rotation], right[rotation], masked
		}(i)
	}
	for j := 0; j < babyStep; j++ {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			masked, localErr := CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, left[j], right[j], j, publicKeys.Params.MaxSlots())
			if localErr != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = localErr
				}
				errMu.Unlock()
				return
			}
			rotBL[j], rotBR[j], X0TRotTensor[j] = left[j], right[j], masked
		}(j)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, nil, nil, nil, nil, nil, firstErr
	}
	return X0RotTensor, X0TRotTensor, X0RotLeft, X0RotRight, rotBL, rotBR, nil
}

/*
 * CipherTensorUnfoldX0ToAttentionWithBSGSLowRankMultiThread
 *
 * 与 CipherTensorUnfoldX0ToAttentionWithBSGSMultiThread 数学等价,
 * 但 X0·Item_1·X0^T 一项用 rank-r 分解 Item_1 ≈ A·Bt^T 计算:
 *
 *   Y[i][t]     = Σ_j A[j][t]  · maskedRot_{-7i}(X0[j])   (融合 A 与 mask-乘, level 10→9)
 *   ZrotB[j][t] = Σ_c Bt[c][t] · maskedRot_j(X0[c])       (融合 Bt 与 mask-乘, level 10→9)
 *   diag1       = Σ_t Y[i][t] ⊙ ZrotB[j][t]               (r 次 ct-ct, 原 25 次)
 *
 * term3 (Item_3·X0^T 对角线) 用 X0TRot 显式计算; diag2/term4/softmax/×V 不变.
 * level 链与原实现完全一致 (diag1 输入双侧 level 9 → 输出 level 8).
 */
func CipherTensorUnfoldX0ToAttentionWithBSGSLowRankMultiThread(publicKeys *encryption.PublicParametersKeys, X0 *encryption.CiphertextTensor, X0Tensor, X0TRotTensor, X0RotTensorLeft, X0RotTensorRight, rotBL, rotBR []*encryption.CiphertextTensor,
	A, Bt, WQBKT, BQWKT, BQBKT [][]float64,
	X0_rear_V, Constant_V [][]float64, babyStep, giantStep int, b, c float64, fixedPointBits int, hoistedRotations, hoistedGiantRescale bool) (*encryption.CiphertextTensor, error) {
	if giantStep < 1 {
		return nil, fmt.Errorf("giantStep must be positive, got %d", giantStep)
	}

	V, err := CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, X0, X0_rear_V, Constant_V)
	if err != nil {
		panic(err)
	}

	VCols := V.NumCols
	VRows := V.NumRows
	VDepth := V.NumDepth

	var VRotTensor = make([]*encryption.CiphertextTensor, babyStep)
	var ZrotBTensor = make([]*encryption.CiphertextTensor, babyStep)
	var vLeft, vRight map[int]*encryption.CiphertextTensor
	if hoistedRotations {
		logical := make([]int, babyStep)
		for j := range logical {
			logical[j] = j
		}
		vLeft, vRight, err = HoistedCipherTensorRotationsByColsNotAdd(publicKeys, V, logical)
		if err != nil {
			return nil, fmt.Errorf("hoisted V rotations: %w", err)
		}
	}

	runtime.GOMAXPROCS(4)
	var wg sync.WaitGroup

	// 生成 V 旋转 与 ZrotB (本头专用, 因 Bt 按头不同)
	for j := 0; j < babyStep; j++ {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()

			var rotV *encryption.CiphertextTensor
			var err error
			if hoistedRotations {
				rotV, err = CipherTensorMulPlainMatAll1WithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, vLeft[j], vRight[j], j, publicKeys.Params.MaxSlots())
			} else {
				rotV, err = CiphertextTensorRotationByColsNewMultiThread(evaluator, V, j, 1, publicKeys.Params.MaxSlots())
			}
			if err != nil {
				panic(err)
			}
			VRotTensor[j] = rotV

			// ZrotB[j] = 融合 Bt 与 mask 的矩形变换, 直接从 level-10 旋转计算 (level 10→9)
			var zb *encryption.CiphertextTensor
			if fixedPointBits == 0 {
				zb, err = CipherTensorMulPlainMatRectWithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, rotBL[j], rotBR[j], Bt, j, publicKeys.Params.MaxSlots())
			} else {
				zb, err = CipherTensorMulPlainMatRectFixedPoint(publicKeys.Params, evaluator, rotBL[j], rotBR[j], Bt, j, fixedPointBits)
			}
			if err != nil {
				panic(err)
			}
			ZrotBTensor[j] = zb
		}(j)
	}
	wg.Wait()
	utils.ProfileMemory("attention-v-zrot-ready")

	// WQBKT depends only on the baby step. Cache the encoded row operands once
	// per head and share them read-only across all eight giant-step workers.
	encodedWQBKT := make([][]*rlwe.Plaintext, babyStep)
	encoder := publicKeys.Encoder.ShallowCopy()
	for j := 0; j < babyStep; j++ {
		encodedWQBKT[j], err = EncodePlainMatRowsForTensor(publicKeys.Params, encoder, rotateMatrixRows(WQBKT, j), X0Tensor[0])
		if err != nil {
			return nil, fmt.Errorf("encode WQBKT baby step %d: %w", j, err)
		}
	}
	utils.ProfileMemory("attention-plaintext-cache-ready")

	// Each giant step writes to its own slot. The previous shared accumulator
	// was a data race; fixed-order reduction after the parallel region keeps the
	// output deterministic.
	giantResults := make([]*encryption.CiphertextTensor, giantStep)

	// BSGS.
	for i := 0; i < giantStep; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			encoder := publicKeys.Encoder.ShallowCopy()

			localNewCiphertexts := make([]*rlwe.Ciphertext, VDepth)
			for k := 0; k < VDepth; k++ {
				localNewCiphertexts[k] = hefloat.NewCiphertext(*publicKeys.Params, V.Ciphertexts[0].Degree(), V.Ciphertexts[0].Level())
			}
			QKV := &encryption.CiphertextTensor{Ciphertexts: localNewCiphertexts, NumRows: VRows, NumCols: VCols, NumDepth: VDepth}

			var Yi *encryption.CiphertextTensor
			var err error
			if fixedPointBits == 0 {
				Yi, err = CipherTensorMulPlainMatRectWithLeftAndRightTensorMultiThread(publicKeys.Params, evaluator, X0RotTensorLeft[i], X0RotTensorRight[i], A, -i*babyStep, publicKeys.Params.MaxSlots())
			} else {
				Yi, err = CipherTensorMulPlainMatRectFixedPoint(publicKeys.Params, evaluator, X0RotTensorLeft[i], X0RotTensorRight[i], A, -i*babyStep, fixedPointBits)
			}
			if err != nil {
				panic(err)
			}
			rotatedBQWKT := rotateMatrixColumns(BQWKT, -i*babyStep)
			encodedBQWKT, err := EncodePlainMatColumnsForTensor(publicKeys.Params, encoder, rotatedBQWKT, X0TRotTensor[0])
			if err != nil {
				panic(err)
			}

			for j := 0; j < babyStep; j++ {
				if (i*babyStep + j) < VCols {
					diagMatrix_1, err := CiphertextTensorMultiplyCiphertextTensorThenAdd(publicKeys.Params, evaluator, Yi, ZrotBTensor[j])
					if err != nil {
						panic(err)
					}
					diagMatrix_23, err := TwoPlaintextProductsAddThenRescale(publicKeys.Params, evaluator, encodedBQWKT, X0TRotTensor[j], encodedWQBKT[j], X0Tensor[i])
					if err != nil {
						panic(err)
					}
					diagMatrix, err := evaluator.AddNew(diagMatrix_1, diagMatrix_23)
					if err != nil {
						panic(err)
					}
					diagPlain, err := GetDiagRotVector(BQBKT, i*babyStep+j, -i*babyStep)
					if err != nil {
						panic(err)
					}
					diagPlain, err = utils.ReaptVector(diagPlain, VRows)
					if err != nil {
						panic(err)
					}
					if err = evaluator.Add(diagMatrix, diagPlain, diagMatrix); err != nil {
						panic(err)
					}
					diagMatrixSoftMax, err := ApproximateSoftmaxCiphertext(evaluator, diagMatrix, b/math.Sqrt(c), 1)
					if err != nil {
						panic(err)
					}
					if err = CiphertextTensorMultiplyCiphertextTensorAddToRes(publicKeys.Params, evaluator, diagMatrixSoftMax, VRotTensor[j], QKV); err != nil {
						panic(err)
					}
				}
			}
			if hoistedGiantRescale {
				// Keep mask multiplication and rotation inside the giant worker so
				// they overlap with the other giant computations. Only the shared
				// raw reduction and one rescale per channel remain after the barrier.
				giantResults[i], err = RotateGiantStepRaw(evaluator, QKV, babyStep*i, publicKeys.Params.MaxSlots())
				if err != nil {
					panic(err)
				}
			} else {
				QKVRotKi, err := CiphertextTensorRotationByColsNewMultiThread(evaluator, QKV, babyStep*i, 1, publicKeys.Params.MaxSlots())
				if err != nil {
					panic(err)
				}
				giantResults[i] = QKVRotKi
			}
		}(i)
	}
	wg.Wait()
	utils.ProfileMemory("attention-giants-ready")
	if hoistedGiantRescale {
		result, err := ReduceRawGiantStepsAndRescale(publicKeys, giantResults)
		if err != nil {
			return nil, err
		}
		utils.ProfileMemory("attention-reduced")
		return result, nil
	}
	newCiphertexts := make([]*rlwe.Ciphertext, VDepth)
	reducer := publicKeys.Evaluator.ShallowCopy()
	for k := 0; k < VDepth; k++ {
		newCiphertexts[k] = giantResults[0].Ciphertexts[k].CopyNew()
		for i := 1; i < giantStep; i++ {
			if err := reducer.Add(newCiphertexts[k], giantResults[i].Ciphertexts[k], newCiphertexts[k]); err != nil {
				return nil, fmt.Errorf("reduce giant step %d output %d: %w", i, k, err)
			}
		}
	}
	utils.ProfileMemory("attention-reduced")
	return &encryption.CiphertextTensor{Ciphertexts: newCiphertexts, NumRows: VRows, NumCols: VCols, NumDepth: VDepth}, nil
}

// RotateGiantStepRaw applies exactly the same pre-rotation boundary masks as
// CiphertextTensorRotationByColsNewMultiThread but deliberately leaves the
// products unrescaled. Keeping this call in each giant worker preserves the
// original overlap between the diagonal work and the final row rotation.
func RotateGiantStepRaw(evaluator *hefloat.Evaluator, tensor *encryption.CiphertextTensor, rotNumber, slots int) (*encryption.CiphertextTensor, error) {
	if tensor == nil || tensor.NumRows < 1 || tensor.NumCols < 1 || tensor.NumDepth < 1 {
		return nil, fmt.Errorf("invalid giant-step tensor")
	}
	rotNumber = (rotNumber + tensor.NumCols) % tensor.NumCols
	leftMask := make([]float64, tensor.NumRows*tensor.NumCols)
	rightMask := make([]float64, tensor.NumRows*tensor.NumCols)
	for row := 0; row < tensor.NumRows; row++ {
		for col := 0; col < tensor.NumCols; col++ {
			if col < rotNumber {
				rightMask[row*tensor.NumCols+col] = 1.0
			} else {
				leftMask[row*tensor.NumCols+col] = 1.0
			}
		}
	}
	outputs := make([]*rlwe.Ciphertext, tensor.NumDepth)
	for channel := 0; channel < tensor.NumDepth; channel++ {
		left, err := evaluator.MulRelinNew(tensor.Ciphertexts[channel], leftMask)
		if err != nil {
			return nil, fmt.Errorf("channel %d left mask: %w", channel, err)
		}
		if rotNumber != 0 {
			if err = evaluator.Rotate(left, rotNumber, left); err != nil {
				return nil, fmt.Errorf("channel %d left rotate: %w", channel, err)
			}
			right, err := evaluator.MulRelinNew(tensor.Ciphertexts[channel], rightMask)
			if err != nil {
				return nil, fmt.Errorf("channel %d right mask: %w", channel, err)
			}
			rightRotation := (rotNumber - tensor.NumCols + slots) % slots
			if err = evaluator.Rotate(right, rightRotation, right); err != nil {
				return nil, fmt.Errorf("channel %d right rotate: %w", channel, err)
			}
			if err = evaluator.Add(left, right, left); err != nil {
				return nil, fmt.Errorf("channel %d join rotations: %w", channel, err)
			}
		}
		outputs[channel] = left
	}
	return &encryption.CiphertextTensor{Ciphertexts: outputs, NumRows: tensor.NumRows, NumCols: tensor.NumCols, NumDepth: tensor.NumDepth}, nil
}

// ReduceRawGiantStepsAndRescale adds same-level, same-scale raw row-rotation
// products and performs the single level-consuming rescale for each channel.
func ReduceRawGiantStepsAndRescale(publicKeys *encryption.PublicParametersKeys, rawGiants []*encryption.CiphertextTensor) (*encryption.CiphertextTensor, error) {
	if len(rawGiants) == 0 || rawGiants[0] == nil {
		return nil, fmt.Errorf("no raw giant-step results")
	}
	rows, cols, depth := rawGiants[0].NumRows, rawGiants[0].NumCols, rawGiants[0].NumDepth
	for giant, tensor := range rawGiants {
		if tensor == nil || tensor.NumRows != rows || tensor.NumCols != cols || tensor.NumDepth != depth {
			return nil, fmt.Errorf("raw giant step %d has inconsistent tensor shape", giant)
		}
	}
	outputs := make([]*rlwe.Ciphertext, depth)
	jobs := make(chan int)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error
	workerCount := runtime.GOMAXPROCS(0)
	if workerCount > depth {
		workerCount = depth
	}
	if workerCount > 4 {
		workerCount = 4
	}
	recordError := func(err error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}
	for worker := 0; worker < workerCount; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			for channel := range jobs {
				accumulator := rawGiants[0].Ciphertexts[channel].CopyNew()
				for giant := 1; giant < len(rawGiants); giant++ {
					if err := evaluator.Add(accumulator, rawGiants[giant].Ciphertexts[channel], accumulator); err != nil {
						recordError(fmt.Errorf("raw giant %d channel %d reduction: %w", giant, channel, err))
						return
					}
				}
				if err := evaluator.Rescale(accumulator, accumulator); err != nil {
					recordError(fmt.Errorf("channel %d final rescale: %w", channel, err))
					return
				}
				outputs[channel] = accumulator
			}
		}()
	}
	for channel := 0; channel < depth; channel++ {
		jobs <- channel
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return &encryption.CiphertextTensor{Ciphertexts: outputs, NumRows: rows, NumCols: cols, NumDepth: depth}, nil
}

// RotateAndReduceGiantStepsRescaleHoisted applies the row-boundary masks and
// physical rotations for all BSGS giant steps, adds their same-level raw
// plaintext products, and performs one rescale per output channel. The legacy
// schedule rescales each nonzero giant step independently. For an 8x22 head,
// this changes 7*22=154 final-stage rescales into 22 without changing depth.
func RotateAndReduceGiantStepsRescaleHoisted(publicKeys *encryption.PublicParametersKeys, giantResults []*encryption.CiphertextTensor, babyStep int) (*encryption.CiphertextTensor, error) {
	if len(giantResults) == 0 || giantResults[0] == nil {
		return nil, fmt.Errorf("no giant-step results")
	}
	if babyStep < 1 {
		return nil, fmt.Errorf("babyStep must be positive, got %d", babyStep)
	}
	rows := giantResults[0].NumRows
	cols := giantResults[0].NumCols
	depth := giantResults[0].NumDepth
	if rows < 1 || cols < 1 || depth < 1 {
		return nil, fmt.Errorf("invalid giant-step tensor shape %dx%dx%d", rows, cols, depth)
	}

	rotations := make([]int, len(giantResults))
	leftMasks := make([][]float64, len(giantResults))
	rightMasks := make([][]float64, len(giantResults))
	for giant, tensor := range giantResults {
		if tensor == nil || tensor.NumRows != rows || tensor.NumCols != cols || tensor.NumDepth != depth {
			return nil, fmt.Errorf("giant step %d has inconsistent tensor shape", giant)
		}
		rotations[giant] = (babyStep * giant) % cols
		leftMasks[giant] = make([]float64, rows*cols)
		rightMasks[giant] = make([]float64, rows*cols)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				if col < rotations[giant] {
					rightMasks[giant][row*cols+col] = 1.0
				} else {
					leftMasks[giant][row*cols+col] = 1.0
				}
			}
		}
	}

	outputs := make([]*rlwe.Ciphertext, depth)
	jobs := make(chan int)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error
	workerCount := runtime.GOMAXPROCS(0)
	if workerCount > depth {
		workerCount = depth
	}
	if workerCount > 4 {
		workerCount = 4
	}
	recordError := func(err error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}
	for worker := 0; worker < workerCount; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			evaluator := publicKeys.Evaluator.ShallowCopy()
			for channel := range jobs {
				var accumulator *rlwe.Ciphertext
				for giant, tensor := range giantResults {
					left, err := evaluator.MulRelinNew(tensor.Ciphertexts[channel], leftMasks[giant])
					if err != nil {
						recordError(fmt.Errorf("giant %d channel %d left mask: %w", giant, channel, err))
						return
					}
					rotation := rotations[giant]
					if rotation != 0 {
						if err = evaluator.Rotate(left, rotation, left); err != nil {
							recordError(fmt.Errorf("giant %d channel %d left rotate: %w", giant, channel, err))
							return
						}
						right, err := evaluator.MulRelinNew(tensor.Ciphertexts[channel], rightMasks[giant])
						if err != nil {
							recordError(fmt.Errorf("giant %d channel %d right mask: %w", giant, channel, err))
							return
						}
						rightRotation := (rotation - cols + publicKeys.Params.MaxSlots()) % publicKeys.Params.MaxSlots()
						if err = evaluator.Rotate(right, rightRotation, right); err != nil {
							recordError(fmt.Errorf("giant %d channel %d right rotate: %w", giant, channel, err))
							return
						}
						if err = evaluator.Add(left, right, left); err != nil {
							recordError(fmt.Errorf("giant %d channel %d join rotations: %w", giant, channel, err))
							return
						}
					}
					if accumulator == nil {
						accumulator = left
					} else if err = evaluator.Add(accumulator, left, accumulator); err != nil {
						recordError(fmt.Errorf("giant %d channel %d raw reduction: %w", giant, channel, err))
						return
					}
				}
				if err := evaluator.Rescale(accumulator, accumulator); err != nil {
					recordError(fmt.Errorf("channel %d final rescale: %w", channel, err))
					return
				}
				outputs[channel] = accumulator
			}
		}()
	}
	for channel := 0; channel < depth; channel++ {
		jobs <- channel
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return &encryption.CiphertextTensor{Ciphertexts: outputs, NumRows: rows, NumCols: cols, NumDepth: depth}, nil
}
