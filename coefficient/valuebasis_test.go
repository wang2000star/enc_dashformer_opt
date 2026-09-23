package coefficient

import (
	"math"
	"testing"
)

func deterministicMatrix(rows, cols int, phase float64) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = math.Sin(float64((i+1)*(j+2))*0.017+phase) + 0.1*math.Cos(float64(i+j)+phase)
		}
	}
	return m
}

func TestValueBasisFullRankAndFusion(t *testing.T) {
	qkv := Coefficient_QKV{A_V: make([][][]float64, 4), Constant_V: make([][][]float64, 4)}
	for h := 0; h < 4; h++ {
		qkv.A_V[h] = deterministicMatrix(25, 32, float64(h))
		qkv.Constant_V[h] = deterministicMatrix(50, 32, float64(h)+0.5)
	}
	dash := Coefficient_dash{
		Head_rear_relu: deterministicMatrix(128, 17, 0.25),
		Head_rear:      deterministicMatrix(128, 11, 0.75),
	}
	got := ComputeCoefficientValueBasis(qkv, dash, []int{32, 32, 32, 32})

	for h := 0; h < 4; h++ {
		original := append([][]float64{}, qkv.A_V[h]...)
		original = append(original, qkv.Constant_V[h]...)
		factor := append([][]float64{}, got.TokenBasis[h]...)
		factor = append(factor, got.PositionBasis[h]...)
		reconstructed := MatrixChainMultiply_slice(factor, got.Recovery[h])
		for i := range original {
			for j := range original[i] {
				if diff := math.Abs(original[i][j] - reconstructed[i][j]); diff > 1e-10 {
					t.Fatalf("head %d reconstruction [%d,%d] diff=%g", h, i, j, diff)
				}
			}
		}

		expectedRelu := MatrixChainMultiply_slice(got.Recovery[h], dash.Head_rear_relu[h*32:(h+1)*32])
		expectedHead := MatrixChainMultiply_slice(got.Recovery[h], dash.Head_rear[h*32:(h+1)*32])
		for i := 0; i < 32; i++ {
			for j := range expectedRelu[i] {
				if diff := math.Abs(expectedRelu[i][j] - got.Head_rear_relu_fused[h*32+i][j]); diff > 1e-12 {
					t.Fatalf("relu fusion mismatch: head=%d row=%d col=%d diff=%g", h, i, j, diff)
				}
			}
			for j := range expectedHead[i] {
				if diff := math.Abs(expectedHead[i][j] - got.Head_rear_fused[h*32+i][j]); diff > 1e-12 {
					t.Fatalf("head fusion mismatch: head=%d row=%d col=%d diff=%g", h, i, j, diff)
				}
			}
		}
	}
}
