package main

import (
	"dashformer/coefficient"
	"dashformer/encryption"
	"dashformer/maths"
	"dashformer/utils"
	"flag"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"time"
)

// /*
func evalDashformerMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys,
	exampleData [][][]float64, dashModelParam utils.DashformerModelParameters) (*encryption.CiphertextTensor, error) {
	// 2.2.加密example数据
	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Encrypte data already, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)

	// 3.1.Embedding
	startTime := time.Now()
	embeddingCipherTensor, err := maths.CiphertextTensorMultiplyPlaintextMatrixMultiThread(publicKeys, ciphertextTensor, dashModelParam.EmbeddingMatrix)
	if err != nil {
		panic(err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Printf("Embedding took %s to run.\n", elapsedTime)

	// 3.2.Encoding
	startTime = time.Now()
	encodingCipherTensor, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, embeddingCipherTensor, dashModelParam.EncodingMatrix)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("Encoding took %s to run.\n", elapsedTime)

	var concatenateHeader *encryption.CiphertextTensor
	// 3.3.Attention
	startTime = time.Now()
	for i := 0; i < 4; i++ {
		// 3.3.1. Compute Q,K,V
		multiAttentionQ, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, encodingCipherTensor, dashModelParam.QueryWeightAttentionMatrixs[i], dashModelParam.QueryBiasAttentionVectors[i])
		if err != nil {
			panic(err)
		}

		multiAttentionK, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, encodingCipherTensor, dashModelParam.KeyWeightAttentionMatrixs[i], dashModelParam.KeyBiasAttentionVectors[i])
		if err != nil {
			panic(err)
		}

		multiAttentionV, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, encodingCipherTensor, dashModelParam.ValueWeightAttentionMatrixs[i], dashModelParam.ValueBiasAttentionVectors[i])
		if err != nil {
			panic(err)
		}

		// 3.3.2 Compute Q X K^T --> Halevi-Shoup
		// **为了优化矩阵乘法层数，将softMax(X)=(x+0.95)^2/400 --> softMax(X)= (QXK^T/(sqrt(32)*20)+ 0.95/20 )^2 **
		multiAttentionQMulKT, err := maths.CiphertextTensorMultiplyCiphertextTensorToHalveiShoupMultiThread(publicKeys, multiAttentionQ, multiAttentionK, 1/(math.Sqrt(32)*20))
		if err != nil {
			panic(err)
		}

		// 3.3.3 Compute SoftMax
		multiAttentionQMulKTSoftMax, err := maths.ApproximateSoftmax(publicKeys, multiAttentionQMulKT, 0.95/20, 1)
		if err != nil {
			panic(err)
		}

		// 3.3.4 Compute softMax(Q X K^T/sqrt(32)) X V --> X
		multiAttentionHeader, err := maths.CiphertextTensorHSMultiplyCiphertextTensorMultiThread(publicKeys, multiAttentionQMulKTSoftMax, multiAttentionV)
		if err != nil {
			panic(err)
		}

		multiAttentionHeader, err = maths.CiphertextTensorQKVToAttentionWithBSGSMultiThread(publicKeys, multiAttentionQ, multiAttentionK, multiAttentionV, 7, 8, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i])
		if err != nil {
			panic(err)
		}
		// 3.3.5 concatenate header
		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
		if err != nil {
			panic(err)
		}
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("Attention took %s to run.\n", elapsedTime)

	// 3.3. Combining Header
	startTime = time.Now()
	combiningHeader, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, concatenateHeader, dashModelParam.CombineWeightMatrixs, dashModelParam.CombineBiasVectors)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("Combining Header took %s to run.\n", elapsedTime)

	// 3.4. Add and LayNorm-1  ***近似1/x需要在layerNorm函数内部进行修改
	startTime = time.Now()
	headerAddEncoding, err := encryption.AddTwoCipherTensorNewMultiThread(publicKeys, combiningHeader, encodingCipherTensor)
	if err != nil {
		panic(err)
	}
	layerNorm1, err := maths.CiphertextTensorLayerNormReplaceVarianceMultiThread(publicKeys, headerAddEncoding, dashModelParam.LayerNormVectorR1, dashModelParam.LayerNormVectorB1, dashModelParam.LayerNormSqrtVariance1)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("LayerNorm-1 took %s to run.\n", elapsedTime)

	// 3.5. Feed Forward with ReLu
	startTime = time.Now()
	feedForwardRelu1, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, layerNorm1, dashModelParam.FeedForwardWeightMatrix1, dashModelParam.FeedForwardBiasVector1)
	if err != nil {
		panic(err)
	}
	// Relu
	feedForwardReluAppromate, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, feedForwardRelu1, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
	if err != nil {
		panic(err)
	}

	feedForwardRelu2, err := maths.CiphertextTensorMultiplyWeightAndAddBiasMultiThread(publicKeys, feedForwardReluAppromate, dashModelParam.FeedForwardWeightMatrix2, dashModelParam.FeedForwardBiasVector2)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("Feed Forward with ReLu took %s to run.\n", elapsedTime)

	// 3.6. Add and LayNorm-2
	startTime = time.Now()
	feedForwardAddlayerNorm1, err := encryption.AddTwoCipherTensorNewMultiThread(publicKeys, layerNorm1, feedForwardRelu2)
	if err != nil {
		panic(err)
	}
	// fmt.Printf("feedForwardAddlayerNorm1 scale :%f\n", feedForwardAddlayerNorm1.Ciphertexts[0].LogScale())
	layerNorm2, err := maths.CiphertextTensorLayerNormReplaceVarianceMultiThread(publicKeys, feedForwardAddlayerNorm1, dashModelParam.LayerNormVectorR2, dashModelParam.LayerNormVectorB2, dashModelParam.LayerNormSqrtVariance2)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("LayerNorm-2 took %s to run.\n", elapsedTime)

	// 3.7. Average Pooling
	// 3.8. Dense Classification
	startTime = time.Now()
	poolingAndClassification, err := maths.CiphertextTensorMultiplyClassificationAndPoolingMultiThread(publicKeys, layerNorm2, dashModelParam.ClassifierWeightMatrix, dashModelParam.ClassifierBiasVector)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("Pooling and dense classification took %s to run.\n", elapsedTime)

	return poolingAndClassification, nil
}

// */

func evalUnfoldDashformerWithBSGSMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys, exampleData [][][]float64,
	dashModelParam utils.DashformerModelParameters, coeff_dash coefficient.Coefficient_dash, coeff_QKV coefficient.Coefficient_QKV, coeff_sqmax coefficient.Coefficient_sqmax, fixedPointBits int) (*encryption.CiphertextTensor, error) {
	//! NOTE: Here, secretKeys are just for debug and is not used for encrypted computation.
	// 2.2.加密example数据
	fmt.Println("Encrypting data ... ")
	startTime := time.Now()
	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
	if err != nil {
		panic(err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Printf("Encrypting data ... takes %s\n", elapsedTime)
	fmt.Printf("Data encryption completed, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)
	fmt.Printf("Ciphertext Tensor X0 Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())
	utils.ProfileMemory("encrypted-input")

	fmt.Println("Start computing with encrypted data")
	fmt.Printf("  ...")
	startTime = time.Now()
	startEncryptedComputation := time.Now()
	X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, err := maths.GenerateCipherTensorRot(publicKeys, ciphertextTensor, 7, 8)
	if err != nil {
		panic(err)
	}

	var concatenateHeader *encryption.CiphertextTensor
	// 3.3.Attention
	fmt.Printf("...")
	for i := 0; i < 4; i++ {

		multiAttentionHeader, err := maths.CipherTensorUnfoldX0ToAttentionWithBSGSMultiThread(publicKeys, ciphertextTensor, X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, coeff_sqmax.Item_1[i], coeff_sqmax.Item_2[i], coeff_sqmax.Item_3[i], coeff_sqmax.Item_4[i], coeff_QKV.A_V[i], coeff_QKV.Constant_V[i], 7, 8, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i], fixedPointBits)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Ciphertext Tensor multiAttentionHeader Level:%d\n", multiAttentionHeader.Ciphertexts[0].Level())
		// 3.3.5 concatenate header
		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
		if err != nil {
			panic(err)
		}
	}

	// fmt.Printf("concatenateHeader Rows:%d, Cols:%d, Depth:%d\n", concatenateHeader.NumRows, concatenateHeader.NumCols, concatenateHeader.NumDepth)

	// 开始计算展开式
	// 1.计算rulu里面的内容
	fmt.Printf("...")
	cipherTensorHeaderBeforeRelu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coeff_dash.Head_before_relu, coeff_dash.Head_rear_relu)
	if err != nil {
		panic(err)
	}

	fmt.Printf("...")
	cipherTensorX0BeforeRulu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_dash.X0_before_relu, coeff_dash.X0_rear_relu)
	if err != nil {
		panic(err)
	}
	// cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNewMultiThread(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
	cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNew(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
	if err != nil {
		panic(err)
	}

	fmt.Printf("...\n")
	cipherTensorBeforeReluResult, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, cipherTensorBeforeRelu, coeff_dash.Constant_Relu)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - before relu takes %s \n", elapsedTime)
	startTime = time.Now()
	// 2.1进行relu
	cipherTensorRelu, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, cipherTensorBeforeReluResult, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
	if err != nil {
		panic(err)
	}
	cipherTensorReluResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, cipherTensorRelu, coeff_dash.Relu_before, coeff_dash.Relu_rear)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - relu takes %s\n", elapsedTime)
	startTime = time.Now()
	// 2.2对Head进行计算
	cipherTensorHeadResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coeff_dash.Head_before, coeff_dash.Head_rear)
	if err != nil {
		panic(err)
	}

	// 2.3对X0进行计算
	cipherTensorX0Result, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_dash.X0_before, coeff_dash.X0_rear)
	if err != nil {
		panic(err)
	}

	// valueTensor, err = encryption.DecryptTensorValue(secretKeys, cipherTensorX0Result)
	// if err != nil {
	// 	panic(err)
	// }
	// utils.PrintSliceInfo(valueTensor, "cipherTensorX0Result")
	// fmt.Println(valueTensor[0])

	// 3.1将所有结果相加
	cipherTensorBeforePooling, err := encryption.AddThreeCipherTensorNewMultiThread(publicKeys, cipherTensorReluResult, cipherTensorHeadResult, cipherTensorX0Result)
	if err != nil {
		panic(err)
	}

	// 3.2进行pooling
	cipherTensorPoolingResult, err := maths.CipherTensorPoolingAndAddConstantMultiThread(publicKeys, cipherTensorBeforePooling, coeff_dash.Constant_Dash)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - after relu takes %s\n", elapsedTime)
	fmt.Printf("Encrypted computation takes %s\n", time.Since(startEncryptedComputation))
	return cipherTensorPoolingResult, nil
}

// evalUnfoldDashformerWithBSGSLowRankMultiTread
// 与 evalUnfoldDashformerWithBSGSMultiTread 数学等价 (相差 M1 的 rank-r 近似),
// 注意力用 level 中性的融合低秩算法; 移除 debug 解密.
func evalUnfoldDashformerWithBSGSLowRankMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys, exampleData [][][]float64,
	dashModelParam utils.DashformerModelParameters, coeff_dash coefficient.Coefficient_dash, coeff_QKV coefficient.Coefficient_QKV, coeff_sqmax coefficient.Coefficient_sqmax, coeff_lowrank coefficient.Coefficient_lowrank, coeffValue *coefficient.Coefficient_valuebasis, fixedPointBits, babyStep, giantStep int, hoistedRotations, hoistedVRotations, hoistedGiantRescale bool) (*encryption.CiphertextTensor, error) {
	// 2.2.加密example数据
	fmt.Println("Encrypting data ... ")
	startTime := time.Now()
	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
	if err != nil {
		panic(err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Printf("Encrypting data ... takes %s\n", elapsedTime)
	fmt.Printf("Data encryption completed, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)
	fmt.Printf("Ciphertext Tensor X0 Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())
	utils.ProfileMemory("encrypted-input")

	fmt.Println("Start computing with encrypted data (LOW-RANK fused attention)")
	fmt.Printf("  ...")
	startTime = time.Now()
	startEncryptedComputation := time.Now()
	var X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, rotBL, rotBR []*encryption.CiphertextTensor
	if hoistedRotations {
		X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, rotBL, rotBR, err = maths.GenerateCipherTensorRotLowRankHoisted(publicKeys, ciphertextTensor, babyStep, giantStep)
	} else {
		X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, rotBL, rotBR, err = maths.GenerateCipherTensorRotLowRank(publicKeys, ciphertextTensor, babyStep, giantStep)
	}
	if err != nil {
		panic(err)
	}
	fmt.Printf("  - GenerateCipherTensorRotLowRank takes %s\n", time.Since(startTime))
	utils.ProfileMemory("rotation-cache-ready")

	var concatenateHeader *encryption.CiphertextTensor
	// 3.3.Attention
	fmt.Printf("...")
	headStart := time.Now()
	for i := 0; i < 4; i++ {
		valueRear := coeff_QKV.A_V[i]
		valueConstant := coeff_QKV.Constant_V[i]
		if coeffValue != nil {
			valueRear = coeffValue.TokenBasis[i]
			valueConstant = coeffValue.PositionBasis[i]
		}

		multiAttentionHeader, err := maths.CipherTensorUnfoldX0ToAttentionWithBSGSLowRankMultiThread(publicKeys, ciphertextTensor, X0RotTensor, X0TRotTensor, cipherTensorLeft, cipherTensorRight, rotBL, rotBR, coeff_lowrank.A[i], coeff_lowrank.Bt[i], coeff_sqmax.Item_2[i], coeff_sqmax.Item_3[i], coeff_sqmax.Item_4[i], valueRear, valueConstant, babyStep, giantStep, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i], fixedPointBits, hoistedVRotations, hoistedGiantRescale)
		if err != nil {
			panic(err)
		}
		fmt.Printf("\n  - head %d attention takes %s, level=%d", i, time.Since(headStart), multiAttentionHeader.Ciphertexts[0].Level())
		headStart = time.Now()

		// 3.3.5 concatenate header
		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
		if err != nil {
			panic(err)
		}
		utils.ProfileMemory(fmt.Sprintf("head-%d-merged", i))
	}
	X0RotTensor, X0TRotTensor = nil, nil
	cipherTensorLeft, cipherTensorRight = nil, nil
	rotBL, rotBR = nil, nil
	utils.ReleaseMemory("attention-cache")

	// 开始计算展开式
	// 1.计算rulu里面的内容
	fmt.Printf("...")
	headRearRelu := coeff_dash.Head_rear_relu
	headRear := coeff_dash.Head_rear
	if coeffValue != nil {
		headRearRelu = coeffValue.Head_rear_relu_fused
		headRear = coeffValue.Head_rear_fused
	}
	cipherTensorHeaderBeforeRelu, err := maths.PlainVecMulCipherTensorMulPlainMatFixedPoint(publicKeys, concatenateHeader, coeff_dash.Head_before_relu, headRearRelu, fixedPointBits)
	if err != nil {
		panic(err)
	}

	fmt.Printf("...")
	cipherTensorX0BeforeRulu, err := maths.PlainVecMulCipherTensorMulPlainMatFixedPoint(publicKeys, ciphertextTensor, coeff_dash.X0_before_relu, coeff_dash.X0_rear_relu, fixedPointBits)
	if err != nil {
		panic(err)
	}
	cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNew(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
	if err != nil {
		panic(err)
	}

	fmt.Printf("...\n")
	cipherTensorBeforeReluResult, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, cipherTensorBeforeRelu, coeff_dash.Constant_Relu)
	if err != nil {
		panic(err)
	}
	cipherTensorHeaderBeforeRelu = nil
	cipherTensorX0BeforeRulu = nil
	cipherTensorBeforeRelu = nil
	utils.ReleaseMemory("before-relu-intermediates")
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - before relu takes %s \n", elapsedTime)
	utils.ProfileMemory("before-relu-ready")
	startTime = time.Now()
	// 2.1进行relu
	cipherTensorRelu, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, cipherTensorBeforeReluResult, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
	if err != nil {
		panic(err)
	}
	cipherTensorReluResult, err := maths.PlainVecMulCipherTensorMulPlainMatFixedPoint(publicKeys, cipherTensorRelu, coeff_dash.Relu_before, coeff_dash.Relu_rear, fixedPointBits)
	if err != nil {
		panic(err)
	}
	cipherTensorBeforeReluResult = nil
	cipherTensorRelu = nil
	utils.ReleaseMemory("relu-intermediates")
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - relu takes %s\n", elapsedTime)
	utils.ProfileMemory("relu-ready")
	startTime = time.Now()

	// 2.2对Head进行计算
	cipherTensorHeadResult, err := maths.PlainVecMulCipherTensorMulPlainMatFixedPoint(publicKeys, concatenateHeader, coeff_dash.Head_before, headRear, fixedPointBits)
	if err != nil {
		panic(err)
	}

	// 2.3对X0进行计算
	cipherTensorX0Result, err := maths.PlainVecMulCipherTensorMulPlainMatFixedPoint(publicKeys, ciphertextTensor, coeff_dash.X0_before, coeff_dash.X0_rear, fixedPointBits)
	if err != nil {
		panic(err)
	}

	// 3.1将所有结果相加
	cipherTensorBeforePooling, err := encryption.AddThreeCipherTensorNewMultiThread(publicKeys, cipherTensorReluResult, cipherTensorHeadResult, cipherTensorX0Result)
	if err != nil {
		panic(err)
	}

	// 3.2进行pooling
	cipherTensorPoolingResult, err := maths.CipherTensorPoolingAndAddConstantMultiThread(publicKeys, cipherTensorBeforePooling, coeff_dash.Constant_Dash)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - after relu takes %s\n", elapsedTime)
	fmt.Printf("Encrypted computation takes %s\n", time.Since(startEncryptedComputation))
	utils.ProfileMemory("encrypted-result-ready")
	return cipherTensorPoolingResult, nil
}

// /*
func evalUnfoldDashformerMultiTread(publicKeys *encryption.PublicParametersKeys, secretKeys *encryption.SecretParametersKeys, exampleData [][][]float64,
	dashModelParam utils.DashformerModelParameters, coefficient coefficient.Coefficient_dash, coeff_QKV coefficient.Coefficient_QKV, coeff_sqmax coefficient.Coefficient_sqmax) (*encryption.CiphertextTensor, error) {
	// 2.2.加密example数据
	ciphertextTensor, err := encryption.EncryptTensorValueMultiTread(publicKeys, exampleData)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Encrypte data already, Rows:%d, Cols:%d, Depths:%d\n", ciphertextTensor.NumRows, ciphertextTensor.NumCols, ciphertextTensor.NumDepth)
	// fmt.Printf("Ciphertext Tensor X0 Level:%d\n", ciphertextTensor.Ciphertexts[0].Level())

	var concatenateHeader *encryption.CiphertextTensor
	// 3.3.Attention
	startTime := time.Now()
	for i := 0; i < 4; i++ {

		multiAttentionQ, err := maths.CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_QKV.A_Q[i], coeff_QKV.Constant_Q[i])
		if err != nil {
			panic(err)
		}

		multiAttentionK, err := maths.CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_QKV.A_K[i], coeff_QKV.Constant_K[i])
		if err != nil {
			panic(err)
		}

		multiAttentionV, err := maths.CipherTensorMulPlainMatAndAddPlainMatMultiThread(publicKeys, ciphertextTensor, coeff_QKV.A_V[i], coeff_QKV.Constant_V[i])
		if err != nil {
			panic(err)
		}

		fmt.Printf("Ciphertext Tensor Q Level:%d\n", multiAttentionQ.Ciphertexts[0].Level())
		fmt.Printf("Ciphertext Tensor K Level:%d\n", multiAttentionK.Ciphertexts[0].Level())
		fmt.Printf("Ciphertext Tensor V Level:%d\n", multiAttentionV.Ciphertexts[0].Level())

		multiAttentionHeader, err := maths.CiphertextTensorQKVToAttentionWithBSGSMultiThread(publicKeys, multiAttentionQ, multiAttentionK, multiAttentionV, 7, 8, dashModelParam.SoftMaxB[i], dashModelParam.SoftMaxC[i])
		if err != nil {
			panic(err)
		}

		fmt.Printf("Ciphertext Tensor multiAttentionHeader Level:%d\n", multiAttentionHeader.Ciphertexts[0].Level())
		// 3.3.5 concatenate header
		concatenateHeader, err = encryption.MergeAndAddCiphertextTensors(concatenateHeader, multiAttentionHeader)
		if err != nil {
			panic(err)
		}
	}
	elapsedTime := time.Since(startTime)
	fmt.Printf("Attention took %s to run.\n", elapsedTime)

	// 开始计算展开式
	// 1.计算rulu里面的内容
	cipherTensorHeaderBeforeRelu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coefficient.Head_before_relu, coefficient.Head_rear_relu)
	if err != nil {
		panic(err)
	}
	cipherTensorX0BeforeRulu, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_before_relu, coefficient.X0_rear_relu)
	if err != nil {
		panic(err)
	}
	cipherTensorBeforeRelu, err := encryption.AddTwoCipherTensorNew(publicKeys, cipherTensorHeaderBeforeRelu, cipherTensorX0BeforeRulu)
	if err != nil {
		panic(err)
	}
	cipherTensorBeforeReluResult, err := maths.CiphertextTensorAddPlaintextMatrixMultiThread(publicKeys, cipherTensorBeforeRelu, coefficient.Constant_Relu)
	if err != nil {
		panic(err)
	}

	// 2.1进行relu
	cipherTensorRelu, err := maths.ApproximatePolynomialCipherTensorMultiThread(publicKeys, cipherTensorBeforeReluResult, dashModelParam.ReluCoefficients, [2]float64{-50, 40})
	if err != nil {
		panic(err)
	}
	cipherTensorReluResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, cipherTensorRelu, coefficient.Relu_before, coefficient.Relu_rear)
	if err != nil {
		panic(err)
	}
	elapsedTime = time.Since(startTime)
	fmt.Printf("  - relu takes %s\n", elapsedTime)
	startTime = time.Now()
	// 2.2对Head进行计算
	cipherTensorHeadResult, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, concatenateHeader, coefficient.Head_before, coefficient.Head_rear)
	if err != nil {
		panic(err)
	}

	// 2.3对X0进行计算
	cipherTensorX0Result, err := maths.PlainVecMulCipherTensorMulPlainMatMultiThread(publicKeys, ciphertextTensor, coefficient.X0_before, coefficient.X0_rear)
	if err != nil {
		panic(err)
	}

	// 3.1将所有结果相加
	cipherTensorBeforePooling, err := encryption.AddThreeCipherTensorNewMultiThread(publicKeys, cipherTensorReluResult, cipherTensorHeadResult, cipherTensorX0Result)
	if err != nil {
		panic(err)
	}

	// 3.2进行pooling
	cipherTensorPoolingResult, err := maths.CipherTensorPoolingAndAddConstantMultiThread(publicKeys, cipherTensorBeforePooling, coefficient.Constant_Dash)
	if err != nil {
		panic(err)
	}

	return cipherTensorPoolingResult, nil
}

// */

func main() {
	startTime := time.Now()
	// Parse execution controls before reading data so experiments can select
	// datasets and output directories without modifying tracked inputs.
	baselineMode := flag.Bool("baseline", false, "run the champion unfold+BSGS path without score/value low rank")
	baselineFixedPointMode := flag.Bool("baseline-fixed-point", false, "apply fixed-point boundary-mask factorization to the full-rank champion path")
	valueBasisMode := flag.Bool("value-basis", false, "enable level-neutral value-basis compression")
	scoreRanksFlag := flag.String("score-ranks", "10,10,8,10", "four comma-separated per-head score ranks")
	valueRanksFlag := flag.String("value-ranks", "24,24,24,24", "four comma-separated per-head value ranks")
	fixedPointBitsFlag := flag.Int("fixed-point-bits", maths.LowRankFixedPointBits, "low-rank mask factorization bits; 0 disables fixed-point factorization")
	memoryMode := flag.Bool("memory-mode", false, "use more frequent Go garbage collection to reduce CKKS peak RSS")
	profileMemoryMode := flag.Bool("profile-memory", false, "print stage-level Go heap and process RSS telemetry")
	logPFlag := flag.String("log-p", "35,34", "comma-separated special-prime bit sizes; 31,31 selects the HES-v2 logQP=430 experiment")
	babyStepFlag := flag.Int("bsgs-baby-step", 7, "attention BSGS baby-step width; giant-step count is derived from the input column count")
	hoistedRotationsFlag := flag.Bool("hoisted-rotations", false, "reuse one RNS decomposition across batched X0 rotations")
	hoistedVRotationsFlag := flag.Bool("hoisted-v-rotations", false, "experimental: also hoist per-head V rotations")
	hoistedGiantRescaleFlag := flag.Bool("hoisted-giant-rescale", false, "experimental: add same-level raw giant-step masks before one rescale per output")
	examplesFlag := flag.String("examples", "", "override the input sequence file")
	outputFlag := flag.String("output", "", "override the output directory")
	dataDirFlag := flag.String("data-dir", "data", "directory containing tokenizer, model parameters and default examples")
	checkInputsFlag := flag.Bool("check-inputs", false, "read inputs and check model/input dimensions without generating keys or running inference")
	flag.Parse()
	if flag.NArg() != 0 {
		log.Fatal("unexpected positional arguments; use --help for supported options")
	}
	if *valueBasisMode && *baselineMode {
		log.Fatal("--value-basis cannot be combined with --baseline")
	}
	if *hoistedGiantRescaleFlag && *baselineMode {
		log.Fatal("--hoisted-giant-rescale applies only to the low-rank path")
	}
	if *memoryMode {
		previousGCPercent := debug.SetGCPercent(50)
		fmt.Printf("Memory mode enabled: GOGC 50 (previous %d)\n", previousGCPercent)
	}
	utils.SetMemoryProfiling(*profileMemoryMode)
	utils.SetAggressiveMemoryMode(*memoryMode)
	if *fixedPointBitsFlag < 0 || *fixedPointBitsFlag > 24 {
		log.Fatalf("--fixed-point-bits must be 0 or in [1,24], got %d", *fixedPointBitsFlag)
	}
	if *baselineFixedPointMode && !*baselineMode {
		log.Fatal("--baseline-fixed-point requires --baseline")
	}
	if *babyStepFlag < 1 {
		log.Fatalf("--bsgs-baby-step must be positive, got %d", *babyStepFlag)
	}
	if *hoistedVRotationsFlag && !*hoistedRotationsFlag {
		log.Fatal("--hoisted-v-rotations requires --hoisted-rotations")
	}
	if *baselineMode && (*babyStepFlag != 7 || *hoistedRotationsFlag || *hoistedVRotationsFlag) {
		log.Fatal("BSGS shape and hoisted rotations currently apply only to the low-rank path")
	}
	logPParts := strings.Split(*logPFlag, ",")
	logP := make([]int, len(logPParts))
	for i, part := range logPParts {
		bits, parseErr := strconv.Atoi(strings.TrimSpace(part))
		if parseErr != nil {
			log.Fatalf("invalid --log-p entry %q: %v", part, parseErr)
		}
		logP[i] = bits
	}
	fmt.Println("Data reading ...")

	// 初始化配置
	examplePath := filepath.Join(*dataDirFlag, "example_AA_sequences.list")
	tokenizerPath := filepath.Join(*dataDirFlag, "dashformer_tokenizer.json")
	modelParamPath := filepath.Join(*dataDirFlag, "dashformer_model_parameters")
	outputPath := filepath.Join(*dataDirFlag, "output")
	if *examplesFlag != "" {
		examplePath = *examplesFlag
	}
	if *outputFlag != "" {
		outputPath = *outputFlag
	}
	fmt.Println("Example Path:", examplePath)
	fmt.Println("Tokenizer Path:", tokenizerPath)
	fmt.Println("Model Parameter Path:", modelParamPath)
	fmt.Println("Output Path:", outputPath)

	// 1. 读取文件数据
	// 1.1.读词向量文件，并解析成字典
	tokenizerDate, err := utils.ReadWordIndex(tokenizerPath)
	if err != nil {
		fmt.Println("Error reading dashformer_tokenizer.json file:", err)
		panic(err)
	}

	// 1.2.读输入示例，根据字典进行转换
	exampleData, err := utils.ReadExampleData(examplePath, tokenizerDate)
	if err != nil {
		log.Fatalf("Error reading example_AA_sequences.list file: %v", err)
		panic(err)
	}
	utils.PrintSliceInfo(exampleData, "exampleDataTensor")
	if len(exampleData) == 0 || len(exampleData[0]) == 0 {
		log.Fatal("input tensor has no rows or columns")
	}
	attentionCols := len(exampleData[0])
	giantStep := (attentionCols + *babyStepFlag - 1) / *babyStepFlag
	if *babyStepFlag > attentionCols {
		log.Fatalf("--bsgs-baby-step=%d exceeds input columns=%d", *babyStepFlag, attentionCols)
	}
	fmt.Printf("Attention BSGS: baby=%d giant=%d cols=%d hoisted-x0=%v hoisted-v=%v hoisted-giant-rescale=%v\n", *babyStepFlag, giantStep, attentionCols, *hoistedRotationsFlag, *hoistedVRotationsFlag, *hoistedGiantRescaleFlag)

	// 1.3.读模型参数文件
	dashModelParam, err := utils.ReadModelParameterFile(modelParamPath)
	if err != nil {
		panic(err)
	}
	// 显示读取结果
	dashModelParam.PrintDimensions()
	if len(dashModelParam.EmbeddingMatrix) != len(tokenizerDate)+1 {
		log.Fatal("tokenizer vocabulary does not match model embedding rows")
	}
	if len(dashModelParam.EncodingMatrix) != attentionCols {
		log.Fatal("sequence length does not match model positional encoding rows")
	}
	if *checkInputsFlag {
		fmt.Printf("Input check passed: %d sequences, %d positions, %d token channels. No keys generated.\n", len(exampleData), attentionCols, len(tokenizerDate)+1)
		return
	}

	// 1.4.生成系数(unfold)
	coeff_dash, coeff_QKV, coeff_sqmax := coefficient.GenerateCoefficient(dashModelParam)

	var coeff_lowrank coefficient.Coefficient_lowrank
	if !*baselineMode {
		parts := strings.Split(*scoreRanksFlag, ",")
		if len(parts) != 4 {
			log.Fatalf("--score-ranks requires four comma-separated ranks, got %q", *scoreRanksFlag)
		}
		lowRankR := make([]int, 4)
		for i, part := range parts {
			rank, parseErr := strconv.Atoi(strings.TrimSpace(part))
			if parseErr != nil || rank < 1 || rank > 25 {
				log.Fatalf("invalid --score-ranks entry %q", part)
			}
			lowRankR[i] = rank
		}
		fmt.Printf("Computing low-rank (r=%v) factorization of M1 per head ...\n", lowRankR)
		coeff_lowrank = coefficient.ComputeCoefficientLowRank(coeff_sqmax, lowRankR)
	}
	var coeffValue *coefficient.Coefficient_valuebasis
	if *valueBasisMode {
		if *baselineMode {
			log.Fatal("--value-basis cannot be combined with --baseline; it extends the score-low-rank path")
		}
		parts := strings.Split(*valueRanksFlag, ",")
		if len(parts) != 4 {
			log.Fatalf("--value-ranks requires four comma-separated ranks, got %q", *valueRanksFlag)
		}
		ranks := make([]int, 4)
		for i, part := range parts {
			rank, parseErr := strconv.Atoi(strings.TrimSpace(part))
			if parseErr != nil {
				log.Fatalf("invalid --value-ranks entry %q: %v", part, parseErr)
			}
			ranks[i] = rank
		}
		valueCoeff := coefficient.ComputeCoefficientValueBasis(coeff_QKV, coeff_dash, ranks)
		coeffValue = &valueCoeff
	}

	// 2.1.生成加密参数
	publicKeys, secretKeys, err := encryption.SetHERealParamsWithLogPAndBSGS(logP, attentionCols, *babyStepFlag, giantStep)
	if err != nil {
		panic(err)
	}
	utils.ProfileMemory("keys-ready")

	// 进行密文计算
	var poolingAndClassification *encryption.CiphertextTensor
	var evalErr error
	if *baselineMode {
		baselineFixedPointBits := 0
		if *baselineFixedPointMode {
			baselineFixedPointBits = *fixedPointBitsFlag
		}
		fmt.Printf("Running BASELINE (champion unfold+BSGS, no low-rank approximation; fixed-point bits=%d) ...\n", baselineFixedPointBits)
		poolingAndClassification, evalErr = evalUnfoldDashformerWithBSGSMultiTread(publicKeys, secretKeys, exampleData, dashModelParam, coeff_dash, coeff_QKV, coeff_sqmax, baselineFixedPointBits)
	} else {
		fmt.Printf("Running LOW-RANK fused attention (value-basis=%v) ...\n", *valueBasisMode)
		fmt.Printf("Low-rank fixed-point mask bits: %d (0=disabled)\n", *fixedPointBitsFlag)
		poolingAndClassification, evalErr = evalUnfoldDashformerWithBSGSLowRankMultiTread(publicKeys, secretKeys, exampleData, dashModelParam, coeff_dash, coeff_QKV, coeff_sqmax, coeff_lowrank, coeffValue, *fixedPointBitsFlag, *babyStepFlag, giantStep, *hoistedRotationsFlag, *hoistedVRotationsFlag, *hoistedGiantRescaleFlag)
	}
	if evalErr != nil {
		log.Fatalf("encrypted evaluation failed: %v", evalErr)
	}

	fmt.Printf("  - ciphertexts now at level:%d\n", poolingAndClassification.Ciphertexts[0].Level())

	fmt.Println("Decrypting and writing the result ...")
	// 解密结果
	valueTensor, err := encryption.DecryptTensorValueMultiThread(secretKeys, poolingAndClassification)
	if err != nil {
		panic(err)
	}

	// 解密到文件中
	if err := utils.WriteResultToFile(outputPath, valueTensor); err != nil {
		log.Fatalf("WriteResultToFile failed: %v", err)
	}
	// 详细分类输出（不会破坏原有行为），输出 top-3
	if err := utils.WriteClassificationDetailed(outputPath, valueTensor, 3); err != nil {
		log.Printf("WriteClassificationDetailed failed: %v", err)
	}

	elapsedTime := time.Since(startTime)
	fmt.Printf("Total running time is %s.\n", elapsedTime)

}
