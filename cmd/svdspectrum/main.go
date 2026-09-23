package main

import (
	"dashformer/coefficient"
	"dashformer/utils"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func main() {
	mp, err := utils.ReadModelParameterFile("data/dashformer_model_parameters")
	if err != nil {
		panic(err)
	}
	_, _, sqmax := coefficient.GenerateCoefficient(mp)
	for h := 0; h < 4; h++ {
		item := sqmax.Item_1[h] // 25×25
		n := len(item)
		data := make([]float64, 0, n*n)
		for i := 0; i < n; i++ {
			data = append(data, item[i]...)
		}
		m := mat.NewDense(n, n, data)
		var svd mat.SVD
		if !svd.Factorize(m, mat.SVDFull) {
			panic("SVD failed")
		}
		sv := svd.Values(nil) // 降序
		s1 := sv[0]
		fmt.Printf("head%d", h)
		for k := 0; k < n; k++ {
			fmt.Printf(" %.6g", sv[k]/s1)
		}
		fmt.Println()
	}
}
