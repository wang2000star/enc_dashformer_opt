package coefficient

import (
	"dashformer/utils"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Create_Coefficient_input(dash utils.DashformerModelParameters) Coefficient_input {

	d := 128.0

	sigma_1_diag := dash.LayerNormSqrtVariance1
	sigma_2_diag := dash.LayerNormSqrtVariance2
	for i := 0; i < len(sigma_1_diag); i++ {
		sigma_1_diag[i] = sigma_1_diag[i] / d
	}
	for i := 0; i < len(sigma_2_diag); i++ {
		sigma_2_diag[i] = sigma_2_diag[i] / d
	}

	gam_1 := Compute_Gamma(d, dash.LayerNormVectorR1)
	gam_2 := Compute_Gamma(d, dash.LayerNormVectorR2)

	one_row := make([]float64, 50)
	for i := range one_row {
		one_row[i] = 1.0
	}
	one_coloum := make([][]float64, 50)
	for i := 0; i < 50; i++ {
		one_coloum[i] = append(one_coloum[i], 1.0)
	}

	dash.ClassifierWeightMatrix = utils.ScaleMatrix(dash.ClassifierWeightMatrix, 1/2649.372705)
	dash.ClassifierBiasVector = utils.ScaleVector(dash.ClassifierBiasVector, 1/2649.372705)

	return Coefficient_input{
		One_50_row:    one_row,
		One_50_coloum: one_coloum,

		W_e: dash.EmbeddingMatrix,
		P:   dash.EncodingMatrix,

		W_Q: dash.QueryWeightAttentionMatrixs,
		B_Q: dash.QueryBiasAttentionVectors,
		W_K: dash.KeyWeightAttentionMatrixs,
		B_K: dash.KeyBiasAttentionVectors,
		W_V: dash.ValueWeightAttentionMatrixs,
		B_V: dash.ValueBiasAttentionVectors,

		W_c: dash.CombineWeightMatrixs,
		B_c: dash.CombineBiasVectors,

		W_1: dash.FeedForwardWeightMatrix1,
		B_1: dash.FeedForwardBiasVector1,
		W_2: dash.FeedForwardWeightMatrix2,
		B_2: dash.FeedForwardBiasVector2,

		Sigma_1_diag: sigma_1_diag,
		Sigma_2_diag: sigma_2_diag,
		Sigma_1:      ToDiagonalMatrix(sigma_1_diag),
		Sigma_2:      ToDiagonalMatrix(sigma_2_diag),
		Gamma_1:      gam_1,
		Gamma_2:      gam_2,
		Beta_1:       dash.LayerNormVectorB1,
		Beta_2:       dash.LayerNormVectorB2,

		W_d: dash.ClassifierWeightMatrix,
		B_d: dash.ClassifierBiasVector,
	}
}

func Compute_coefficient_dash(in Coefficient_input) Coefficient_dash {

	c_y2 := MatrixChainAdd_slice(MatrixChainMultiply_slice(in.Sigma_1, in.One_50_coloum, in.B_c, in.Gamma_1),
		MatrixChainMultiply_slice(in.Sigma_1, in.P, in.Gamma_1),
		MatrixChainMultiply_slice(in.One_50_coloum, in.Beta_1))

	c_relu := MatrixChainAdd_slice(MatrixChainMultiply_slice(c_y2, in.W_1),
		MatrixChainMultiply_slice(in.One_50_coloum, in.B_1))

	c_dash := MatrixChainAdd_slice(MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, c_y2, in.Gamma_2, in.W_d),
		MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, in.One_50_coloum, in.B_2, in.Gamma_2, in.W_d),
		MatrixChainMultiply_slice(in.One_50_row, in.One_50_coloum, in.Beta_2, in.W_d))[0]
	for i := 0; i < 25; i++ {
		c_dash[i] = c_dash[i] + in.B_d[i]
	}

	// inverse, err := InverseMatrix(MatrixChainMultiply_slice(in.W_1, Transp(in.W_1)) )
	// if err != nil {
	// 	panic(err)
	// }

	return Coefficient_dash{
		Relu_before: MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2)[0],
		Relu_rear:   MatrixChainMultiply_slice(in.W_2, in.Gamma_2, in.W_d),

		Head_before_relu: in.Sigma_1_diag,
		Head_rear_relu:   MatrixChainMultiply_slice(in.W_c, in.Gamma_1, in.W_1),

		Head_before: MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, in.Sigma_1)[0],
		Head_rear:   MatrixChainMultiply_slice(in.W_c, in.Gamma_1, in.Gamma_2, in.W_d),
		X0_before:   MatrixChainMultiply_slice(in.One_50_row, in.Sigma_2, in.Sigma_1)[0],
		X0_rear:     MatrixChainMultiply_slice(in.W_e, in.Gamma_1, in.Gamma_2, in.W_d),

		X0_before_relu: in.Sigma_1_diag,
		X0_rear_relu:   MatrixChainMultiply_slice(in.W_e, in.Gamma_1, in.W_1),

		Constant_Dash: c_dash,
		Constant_Relu: c_relu,
	}
}

func Compute_coefficient_QKV(in Coefficient_input) Coefficient_QKV {

	var a_Q [][][]float64
	var a_K [][][]float64
	var a_V [][][]float64
	for i := 0; i < 4; i++ {
		a_Q = append(a_Q, MatrixChainMultiply_slice(in.W_e, in.W_Q[i]))
		a_K = append(a_K, MatrixChainMultiply_slice(in.W_e, in.W_K[i]))
		a_V = append(a_V, MatrixChainMultiply_slice(in.W_e, in.W_V[i]))
	}

	var constant_Q [][][]float64
	var constant_K [][][]float64
	var constant_V [][][]float64
	for i := 0; i < 4; i++ {
		constant_Q = append(constant_Q, MatrixChainAdd_slice(MatrixChainMultiply_slice(in.P, in.W_Q[i]), MatrixChainMultiply_slice(in.One_50_coloum, in.B_Q[i])))
		constant_K = append(constant_K, MatrixChainAdd_slice(MatrixChainMultiply_slice(in.P, in.W_K[i]), MatrixChainMultiply_slice(in.One_50_coloum, in.B_K[i])))
		constant_V = append(constant_V, MatrixChainAdd_slice(MatrixChainMultiply_slice(in.P, in.W_V[i]), MatrixChainMultiply_slice(in.One_50_coloum, in.B_V[i])))
	}

	return Coefficient_QKV{
		A_Q: a_Q,
		A_K: a_K,
		A_V: a_V,

		Constant_Q: constant_Q,
		Constant_K: constant_K,
		Constant_V: constant_V,
	}
}

func Compute_coefficient_sqmax(coeffi_QKV Coefficient_QKV, b [4]float64, c [4]float64) Coefficient_sqmax {

	if len(b) != 4 || len(c) != 4 {
		fmt.Println("len(b) != 4 or len(c) != 4 !!!")
	}

	g := make([]float64, 4)
	// h := make([]float64, 4)
	for i := 0; i < 4; i++ {
		g[i] = 1.0 / (math.Sqrt(32.0 * c[i]))
		// h[i] = b[i] / math.Sqrt(c[i])
	}

	item_1 := make([][][]float64, 4)
	item_2 := make([][][]float64, 4)
	item_3 := make([][][]float64, 4)
	item_4 := make([][][]float64, 4)
	for i := 0; i < 4; i++ {
		item_1[i] = MatrixChainMultiply_slice(coeffi_QKV.A_Q[i], Transp(coeffi_QKV.A_K[i]))
		item_1[i] = MultiplyByScalar(item_1[i], g[i])

		item_2[i] = MatrixChainMultiply_slice(coeffi_QKV.A_Q[i], Transp(coeffi_QKV.Constant_K[i]))
		item_2[i] = MultiplyByScalar(item_2[i], g[i])

		item_3[i] = MatrixChainMultiply_slice(coeffi_QKV.Constant_Q[i], Transp(coeffi_QKV.A_K[i]))
		item_3[i] = MultiplyByScalar(item_3[i], g[i])

		item_4[i] = MatrixChainMultiply_slice(coeffi_QKV.Constant_Q[i], Transp(coeffi_QKV.Constant_K[i]))
		item_4[i] = MultiplyByScalar(item_4[i], g[i])
		// item_4[i] = AddWithScalar(item_4[i], h[i])
	}

	return Coefficient_sqmax{
		// G_g: g,

		Item_1: item_1,
		Item_2: item_2,
		Item_3: item_3,
		Item_4: item_4,
	}
}

// func Compute_coefficient_head(coeffi_in Coefficient_input) (Coefficient_head) {

// 	W_Head_whole := MatrixChainMultiply_slice(coeffi_in.W_c, coeffi_in.Gamma_1, coeffi_in.W_1)

// 	return Coefficient_head{
// 		Sigma_1_diag: coeffi_in.Sigma_1_diag,

//			W_Head: SplitMatrixIntoFourChunks_byRow(W_Head_whole),
//		}
//	}
//
// ComputeCoefficientLowRank 对每个头的 Item_1 (25×25) 做截断 SVD,
// 返回 A[h] = U_r·√Σ_r (25×r) 与 Bt[h] = V_r·√Σ_r (25×r).
// 数学上 A[h]·Bt[h]^T 是 Item_1[h] 的最优 rank-r 近似.
// 该近似只以 σ_{r+1} 的谱误差扰动注意力分数; 不改变乘法深度
// (两个因子分别融合进已有的两个 mask-乘阶段).
func ComputeCoefficientLowRank(coeffi_sqmax Coefficient_sqmax, r []int) Coefficient_lowrank {
	if len(r) != 4 {
		panic(fmt.Sprintf("ComputeCoefficientLowRank: expected 4 per-head ranks, got %d", len(r)))
	}
	for _, v := range r {
		if v <= 0 || v > 25 {
			panic(fmt.Sprintf("ComputeCoefficientLowRank: invalid rank %d", v))
		}
	}

	A := make([][][]float64, 4)
	Bt := make([][][]float64, 4)

	for h := 0; h < 4; h++ {
		rh := r[h]                     // 按头自适应秩
		item := coeffi_sqmax.Item_1[h] // 25×25
		n := len(item)
		data := make([]float64, 0, n*n)
		for i := 0; i < n; i++ {
			data = append(data, item[i]...)
		}
		m := mat.NewDense(n, n, data)

		var svd mat.SVD
		if ok := svd.Factorize(m, mat.SVDFull); !ok {
			panic(fmt.Sprintf("SVD failed for head %d", h))
		}
		var U, V mat.Dense
		svd.UTo(&U)
		svd.VTo(&V)
		sv := svd.Values(nil) // 降序

		A[h] = make([][]float64, n)
		Bt[h] = make([][]float64, n)
		for i := 0; i < n; i++ {
			A[h][i] = make([]float64, rh)
			Bt[h][i] = make([]float64, rh)
			for t := 0; t < rh; t++ {
				s := math.Sqrt(sv[t])
				A[h][i][t] = U.At(i, t) * s
				Bt[h][i][t] = V.At(i, t) * s
			}
		}

		// 打印近似质量
		fmt.Printf("  [lowrank] head %d: r=%d, sigma_1=%.4f, sigma_%d=%.6f, rel spectral err=%.4f%%\n",
			h, rh, sv[0], rh+1, sv[rh], sv[rh]/sv[0]*100)
	}

	maxR := r[0]
	for _, v := range r[1:] {
		if v > maxR {
			maxR = v
		}
	}
	return Coefficient_lowrank{A: A, Bt: Bt, R: maxR}
}

// ComputeCoefficientValueBasis factors each 75×32 value map and fuses the
// r_h×32 recovery into the corresponding 32-row block of both consumers.
func ComputeCoefficientValueBasis(coeffQKV Coefficient_QKV, coeffDash Coefficient_dash, ranks []int) Coefficient_valuebasis {
	if len(ranks) != 4 {
		panic(fmt.Sprintf("ComputeCoefficientValueBasis: expected 4 per-head ranks, got %d", len(ranks)))
	}

	tokenBasis := make([][][]float64, 4)
	positionBasis := make([][][]float64, 4)
	recovery := make([][][]float64, 4)
	fusedRelu := make([][]float64, 0)
	fusedHead := make([][]float64, 0)

	for h, rank := range ranks {
		if rank <= 0 || rank > 32 {
			panic(fmt.Sprintf("ComputeCoefficientValueBasis: invalid rank %d for head %d", rank, h))
		}
		if len(coeffQKV.A_V[h]) != 25 || len(coeffQKV.Constant_V[h]) != 50 {
			panic(fmt.Sprintf("ComputeCoefficientValueBasis: unexpected value-map shape for head %d", h))
		}

		data := make([]float64, 0, 75*32)
		for _, row := range coeffQKV.A_V[h] {
			data = append(data, row...)
		}
		for _, row := range coeffQKV.Constant_V[h] {
			data = append(data, row...)
		}
		g := mat.NewDense(75, 32, data)
		var svd mat.SVD
		if ok := svd.Factorize(g, mat.SVDThin); !ok {
			panic(fmt.Sprintf("value-basis SVD failed for head %d", h))
		}
		var u, v mat.Dense
		svd.UTo(&u)
		svd.VTo(&v)
		singular := svd.Values(nil)

		// Balanced factorization avoids placing the entire singular-value
		// dynamic range on either CKKS plaintext operand.
		factor := make([][]float64, 75)
		for i := range factor {
			factor[i] = make([]float64, rank)
			for t := 0; t < rank; t++ {
				factor[i][t] = u.At(i, t) * math.Sqrt(singular[t])
			}
		}
		recovery[h] = make([][]float64, rank)
		for t := 0; t < rank; t++ {
			recovery[h][t] = make([]float64, 32)
			for j := 0; j < 32; j++ {
				recovery[h][t][j] = math.Sqrt(singular[t]) * v.At(j, t)
			}
		}
		tokenBasis[h] = factor[:25]
		positionBasis[h] = factor[25:]

		reluBlock := coeffDash.Head_rear_relu[h*32 : (h+1)*32]
		headBlock := coeffDash.Head_rear[h*32 : (h+1)*32]
		fusedRelu = append(fusedRelu, MatrixChainMultiply_slice(recovery[h], reluBlock)...)
		fusedHead = append(fusedHead, MatrixChainMultiply_slice(recovery[h], headBlock)...)

		relErr := 0.0
		if rank < len(singular) {
			relErr = singular[rank] / singular[0] * 100
		}
		fmt.Printf("  [value-basis] head %d: r=%d, sigma_1=%.4f, relative spectral error=%.4f%%\n",
			h, rank, singular[0], relErr)
	}

	return Coefficient_valuebasis{
		TokenBasis: tokenBasis, PositionBasis: positionBasis, Recovery: recovery,
		Head_rear_relu_fused: fusedRelu, Head_rear_fused: fusedHead,
		Ranks: append([]int(nil), ranks...),
	}
}

func GenerateCoefficient(dashModelParam utils.DashformerModelParameters) (Coefficient_dash, Coefficient_QKV, Coefficient_sqmax) {
	coeff_in := Create_Coefficient_input(dashModelParam)
	// coeff_in.PrintDimensions()

	coeff_dash := Compute_coefficient_dash(coeff_in)
	// coeff_dash.PrintDimensions()

	coeff_QKV := Compute_coefficient_QKV(coeff_in)
	// coeff_QKV.PrintDimensions()

	coeff_sqmax := Compute_coefficient_sqmax(coeff_QKV, dashModelParam.SoftMaxB, dashModelParam.SoftMaxC)
	// coeff_sqmax.PrintDimensions()

	return coeff_dash, coeff_QKV, coeff_sqmax
}
