package encryption

import (
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type PublicParametersKeys struct {
	Params    *hefloat.Parameters
	Encoder   *hefloat.Encoder
	Encryptor *rlwe.Encryptor
	Evaluator *hefloat.Evaluator
}

type SecretParametersKeys struct {
	Params    *hefloat.Parameters
	Sk        *rlwe.SecretKey
	Encoder   *hefloat.Encoder
	Decryptor *rlwe.Decryptor
}

func CopyKeyGenerator(params rlwe.ParameterProvider, enc *rlwe.Encryptor) *rlwe.KeyGenerator {
	return &rlwe.KeyGenerator{
		Encryptor: enc,
	}
}

// RequiredRotationSteps returns the direct column rotations used by the
// low-rank BSGS graph. A logical rotation within a row of length cols needs
// both the non-wrapping rotation r and the wrapping rotation r-cols.
func RequiredRotationSteps(cols, babyStep, giantStep int) []int {
	steps := make(map[int]struct{})
	addColumnRotation := func(rotation int) {
		r := rotation % cols
		if r < 0 {
			r += cols
		}
		if r != 0 {
			steps[r] = struct{}{}
		}
		if wrap := r - cols; wrap != 0 {
			steps[wrap] = struct{}{}
		}
	}

	// Precomputed X0 giant steps, baby steps/V rotations, and the rotation of
	// each accumulated giant-step attention result.
	for i := 0; i < giantStep; i++ {
		addColumnRotation(-i * babyStep)
		addColumnRotation(i * babyStep)
	}
	for j := 0; j < babyStep; j++ {
		addColumnRotation(j)
	}

	result := make([]int, 0, len(steps))
	for step := range steps {
		result = append(result, step)
	}
	sort.Ints(result)
	return result
}

func deduplicateGaloisElements(elements []uint64) []uint64 {
	seen := make(map[uint64]struct{}, len(elements))
	result := make([]uint64, 0, len(elements))
	for _, element := range elements {
		if _, ok := seen[element]; ok {
			continue
		}
		seen[element] = struct{}{}
		result = append(result, element)
	}
	return result
}

// NewHERealParameters returns the deployed CKKS profile with a selectable
// special-prime basis. Keeping LogQ fixed lets experiments tighten security or
// tune key switching without changing the multiplicative-depth budget.
func NewHERealParameters(logP []int) (hefloat.Parameters, error) {
	if len(logP) == 0 {
		return hefloat.Parameters{}, fmt.Errorf("LogP must contain at least one special prime")
	}
	for i, bits := range logP {
		if bits < 2 || bits > 60 {
			return hefloat.Parameters{}, fmt.Errorf("LogP[%d]=%d is outside [2,60]", i, bits)
		}
	}

	params, err := hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN: 14,
			LogQ: []int{38, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33},
			LogP: append([]int(nil), logP...),
			// RingType:        ring.ConjugateInvariant,
			LogDefaultScale: 33,
		})
	if err != nil {
		return hefloat.Parameters{}, err
	}

	// This is the legacy HES/Lattigo bound. The newer HES-v2 profile and an
	// independent lattice-estimator audit are reported separately; experiments
	// targeting that stricter classical-128 profile use LogP=[31,31] (LogQP=430).
	const legacyMaxLogQP128 = 438.0
	if params.LogQP() > legacyMaxLogQP128 {
		return hefloat.Parameters{}, fmt.Errorf("unsafe parameter profile: logQP %.3f exceeds legacy 128-bit bound %.0f for logN=14", params.LogQP(), legacyMaxLogQP128)
	}
	return params, nil
}

func SetHERealParams() (*PublicParametersKeys, *SecretParametersKeys, error) {
	return SetHERealParamsWithLogP([]int{35, 34})
}

func SetHERealParamsWithLogP(logP []int) (*PublicParametersKeys, *SecretParametersKeys, error) {
	return SetHERealParamsWithLogPAndBSGS(logP, 50, 7, 8)
}

// SetHERealParamsWithLogPAndBSGS generates the exact rotation-key set for a
// selected BSGS shape. Keeping key generation coupled to the evaluated graph
// prevents parameter sweeps from silently falling back to composed rotations.
func SetHERealParamsWithLogPAndBSGS(logP []int, cols, babyStep, giantStep int) (*PublicParametersKeys, *SecretParametersKeys, error) {
	if cols < 1 || babyStep < 1 || giantStep < 1 || babyStep*giantStep < cols {
		return nil, nil, fmt.Errorf("invalid BSGS shape: cols=%d baby=%d giant=%d", cols, babyStep, giantStep)
	}
	fmt.Printf("CKKS initialization ...")
	ckksIniStartTime := time.Now()
	params, err := NewHERealParameters(logP)
	if err != nil {
		return nil, nil, err
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	ecd := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)

	// Generate only the keys used by the deployed low-rank BSGS graph, plus
	// the exact set required by final pooling. The original implementation
	// generated every rotation in [-50,50].
	rotNumbers := RequiredRotationSteps(cols, babyStep, giantStep)
	galEls := params.GaloisElements(rotNumbers)
	galEls = append(galEls, params.GaloisElementsForInnerSum(1, cols)...)
	galEls = deduplicateGaloisElements(galEls)
	fmt.Printf(" rotation keys=%d (direct steps=%d);", len(galEls), len(rotNumbers))
	// fmt.Printf("galEls: %v\n", galEls)
	// eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

	// // BEGIN

	// single thread
	// galoisKeys := make([]*rlwe.GaloisKey, len(galEls))
	// for i, galEl := range galEls {
	// 	if galoisKeys[i] == nil {
	// 		galoisKeys[i] = kgen.GenGaloisKeyNew(galEl, sk)
	// 	} else {
	// 		kgen.GenGaloisKey(galEl, sk, galoisKeys[i])
	// 	}
	// }
	// fmt.Printf("Galois keys generation ... completed\n")

	// FOUR threads

	galoisKeys := make([]*rlwe.GaloisKey, len(galEls))
	runtime.GOMAXPROCS(4)
	var wg sync.WaitGroup
	numThreads := 4
	chunkSize := (len(galEls) + numThreads - 1) / numThreads
	// fmt.Printf("dimension of galEls: %d\n", len(galEls))
	// fmt.Printf("dimension of chunkSize: %d\n", chunkSize)

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > len(galEls) {
			end = len(galEls)
		}
		// fmt.Printf("(start, end): (%d, %d)\n", start, end)
		wg.Add(1)
		go func(start, end, threadNum int) {
			defer wg.Done()
			localEncryptor := enc.ShallowCopy()
			localKgen := CopyKeyGenerator(params, localEncryptor)

			// localKgen := OurKeyGenerator(params, enc)
			for i := start; i < end; i++ {
				galEl := galEls[i]
				// fmt.Printf("galEl @ thread %d: %d\n", threadNum, galEl)
				if galoisKeys[i] == nil {
					// fmt.Printf("I AM thread %d ... OK\n    -- current i: %d\n", threadNum, i)
					galoisKeys[i] = localKgen.GenGaloisKeyNew(galEl, sk)
				} else {
					// fmt.Printf("I AM thread %d ... OK\n", threadNum)
					kgen.GenGaloisKey(galEl, sk, galoisKeys[i])
				}
				// fmt.Printf("Generated Galois Key for rotation %d\n", galEl)
			}
		}(start, end, t)
	}

	wg.Wait()
	evaluationKeyBytes := rlk.BinarySize()
	for _, key := range galoisKeys {
		evaluationKeyBytes += key.BinarySize()
	}
	fmt.Printf(" evaluation-key bytes=%d;", evaluationKeyBytes)

	eval := hefloat.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...))
	// // fmt.Println("I AM HERE")
	// // END

	dec := rlwe.NewDecryptor(params, sk)
	fmt.Printf(" takes %s\n", time.Since(ckksIniStartTime))
	fmt.Printf("  - log N = %d, log QP = %.3f, log P = %v, max_level = %d, log_scale = %d\n",
		params.LogN(), params.LogQP(), logP, params.MaxLevel(), params.LogDefaultScale())

	return &PublicParametersKeys{
			Params:    &params,
			Encoder:   ecd,
			Encryptor: enc,
			Evaluator: eval,
		}, &SecretParametersKeys{
			Params:    &params,
			Sk:        sk,
			Encoder:   ecd,
			Decryptor: dec,
		}, nil
}

// func SetBSGSHERealParams(babyStep int, giantStep int, cols int) (*PublicParametersKeys, *SecretParametersKeys, error) {
// 	var err error
// 	var params hefloat.Parameters

// 	if params, err = hefloat.NewParametersFromLiteral(
// 		hefloat.ParametersLiteral{
// 			LogN:            13,
// 			LogQ:            []int{51, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
// 			LogP:            []int{50, 50, 50},
// 			LogDefaultScale: 40,
// 		}); err != nil {
// 		panic(err)
// 	}

// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()
// 	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
// 	rlk := kgen.GenRelinearizationKeyNew(sk)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk)
// 	ecd := hefloat.NewEncoder(params)
// 	enc := rlwe.NewEncryptor(params, pk)
// 	eval := hefloat.NewEvaluator(params, evk)

// 	// 生成旋转步数
// 	var rotNumbers []int
// 	// 生成大步小步所需的步长
// 	for i := 1; i*babyStep < cols; i++ {
// 		rotNumbers = append(rotNumbers, i)
// 		rotNumbers = append(rotNumbers, -i)
// 		rotNumbers = append(rotNumbers, i*babyStep)
// 		rotNumbers = append(rotNumbers, -i*babyStep)
// 	}
// 	fmt.Print(rotNumbers)
// 	galEls := params.GaloisElements(rotNumbers)
// 	fmt.Println(galEls)
// 	// 假如pooling旋转密钥
// 	// galEls = append(galEls, params.GaloisElementsForInnerSum(1, cols)...)
// 	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

// 	dec := rlwe.NewDecryptor(params, sk)

// 	return &PublicParametersKeys{
// 			Params:    &params,
// 			Encoder:   ecd,
// 			Encryptor: enc,
// 			Evaluator: eval,
// 		}, &SecretParametersKeys{
// 			Params:    &params,
// 			Sk:        sk,
// 			Encoder:   ecd,
// 			Decryptor: dec,
// 		}, nil
// }
