package utils

/*
* This package include read file function
* 1. ReadWordIndex: reads a JSON file from the given path
* 2. ReadExampleData: read example file from the give path
 */

import (
	"bufio"
	"dashformer/config"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// 读取 JSON 文件并解析 word_index 字段的函数
func ReadWordIndex(filePath string) (map[string]int, error) {
	// 打开文件
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 读取文件内容
	byteValue, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	// 定义一个空接口来存储解码后的 JSON 数据
	var result map[string]interface{}
	if err := json.Unmarshal(byteValue, &result); err != nil {
		return nil, err
	}

	// 提取 config 字段
	config, ok := result["config"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("config field not found or is not a JSON object")
	}

	// 提取 word_index 字段
	wordIndexStr, ok := config["word_index"].(string)
	if !ok {
		return nil, fmt.Errorf("word_index field not found or is not a string")
	}

	// 将 word_index 字段解析为 map[string]int
	var wordIndex map[string]int
	if err := json.Unmarshal([]byte(wordIndexStr), &wordIndex); err != nil {
		return nil, err
	}

	return wordIndex, nil
}

// ReadExampleData reads space-separated tokens with an optional comma-separated
// label. It rejects malformed input instead of terminating the caller or panicking.
func ReadExampleData(filePath string, tokenizer map[string]int) ([][][]float64, error) {
	if len(tokenizer) == 0 {
		return nil, fmt.Errorf("tokenizer is empty")
	}
	seen := make(map[int]bool, len(tokenizer))
	for token, index := range tokenizer {
		if index < 1 || index > len(tokenizer) || seen[index] {
			return nil, fmt.Errorf("invalid or duplicate tokenizer index %d for token %q", index, token)
		}
		seen[index] = true
	}
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("open sequences: %w", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	var tensor [][][]float64
	width := 0
	for lineNumber := 1; scanner.Scan(); lineNumber++ {
		sequence := strings.SplitN(scanner.Text(), ",", 2)[0]
		tokens := strings.Fields(sequence)
		if len(tokens) == 0 {
			return nil, fmt.Errorf("line %d: empty sequence", lineNumber)
		}
		if width == 0 {
			width = len(tokens)
		}
		if len(tokens) != width {
			return nil, fmt.Errorf("line %d: sequence length %d differs from expected %d", lineNumber, len(tokens), width)
		}
		row := make([][]float64, len(tokens))
		for position, token := range tokens {
			index, ok := tokenizer[strings.ToLower(token)]
			if !ok {
				return nil, fmt.Errorf("line %d, position %d: unknown token %q", lineNumber, position+1, token)
			}
			row[position] = make([]float64, len(tokenizer)+1)
			row[position][index] = 1
		}
		tensor = append(tensor, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read sequences: %w", err)
	}
	if len(tensor) == 0 {
		return nil, fmt.Errorf("sequence file is empty")
	}
	return tensor, nil
}

// 将文件转换成二维切片
func parseMatrix(lines []string) ([][]float64, error) {
	if len(lines) == 0 {
		return nil, fmt.Errorf("matrix has no rows")
	}
	var matrixSlices [][]float64
	width := len(strings.Fields(lines[0]))
	if width == 0 {
		return nil, fmt.Errorf("matrix has no columns")
	}

	for i, line := range lines {
		cols := strings.Fields(line)
		if len(cols) != width {
			return nil, fmt.Errorf("matrix row %d has %d columns; expected %d", i+1, len(cols), width)
		}
		row := make([]float64, len(cols))
		for j, col := range cols {
			num, err := strconv.ParseFloat(col, 64)
			if err != nil {
				return [][]float64{}, fmt.Errorf("error converting string to float64 at line %d, column %d: %v", i+1, j+1, err)
			}
			if math.IsNaN(num) || math.IsInf(num, 0) {
				return nil, fmt.Errorf("non-finite matrix value at row %d, column %d", i+1, j+1)
			}
			row[j] = num
		}
		matrixSlices = append(matrixSlices, row)
	}

	return matrixSlices, nil
}

// 将文件转换成一维向量
func parseVector(lines []string) ([]float64, error) {
	var vectorSlices []float64

	for i, line := range lines {
		cols := strings.Fields(line)
		if len(cols) != 1 {
			return []float64{}, fmt.Errorf("line %d does not contain 1 column", i+129)
		}
		num, err := strconv.ParseFloat(cols[0], 64)
		if err != nil {
			return []float64{}, fmt.Errorf("error converting string to float64: %v", err)
		}
		vectorSlices = append(vectorSlices, num)
	}

	return vectorSlices, nil
}

// 读取W_Q, W_K, W_V 三个文件
func ReadMultiAttentionFile(filename string) ([4][][]float64, [4][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) != 256 {
		return [4][][]float64{}, [4][]float64{}, fmt.Errorf("the file does not contain 256 lines")
	}

	// 解析前128行
	var matrixSlices [4][][]float64
	for i := 0; i < 4; i++ {
		matrixSlices[i] = make([][]float64, 128)
		for j := range matrixSlices[i] {
			matrixSlices[i][j] = make([]float64, 32)
		}
	}

	for i := 0; i < 128; i++ {
		cols := strings.Fields(lines[i])
		if len(cols) != 128 {
			return [4][][]float64{}, [4][]float64{}, fmt.Errorf("line %d does not contain 128 columns", i+1)
		}
		for j := 0; j < 128; j++ {
			num, err := strconv.ParseFloat(cols[j], 64)
			if err != nil {
				return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error converting string to float64: %v", err)
			}
			matrixSlices[j/32][i][j%32] = num
		}
	}

	// 解析后128行
	var MatrixSlices [4][]float64
	for i := 0; i < 4; i++ {
		MatrixSlices[i] = make([]float64, 32)
	}

	for i := 0; i < 128; i++ {
		cols := strings.Fields(lines[i+128])
		if len(cols) != 1 {
			return [4][][]float64{}, [4][]float64{}, fmt.Errorf("line %d does not contain 1 column", i+129)
		}
		num, err := strconv.ParseFloat(cols[0], 64)
		if err != nil {
			return [4][][]float64{}, [4][]float64{}, fmt.Errorf("error converting string to float64: %v", err)
		}
		MatrixSlices[i/32][i%32] = num
	}

	return matrixSlices, MatrixSlices, nil
}

// 读取文件combineHead和classifier
func ReadCombineAndClassifierFile(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [][]float64{}, []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) <= 128 {
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 取前128行
	first128Lines := lines[:128]

	// 取128行后的所有行
	remainingLines := lines[128:]

	// 解析前128行
	matrixWeightSlices, err := parseMatrix(first128Lines)
	if err != nil {
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// fmt.Printf("First 128 lines parsed successfully with dimensions: %d rows x varying columns\n", len(matrixWeightSlices))

	// 解析128行后的所有行
	vectorBaisSlices, err := parseVector(remainingLines)
	if err != nil {
		fmt.Println("Error parsing remaining lines:", err)
		return [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	return matrixWeightSlices, vectorBaisSlices, nil
}

// 读取文件combineHead和classifier
func ReadFeedFowardFile(filename string) ([][]float64, []float64, [][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) != 768 {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("feed-forward parameters require 768 lines, got %d", len(lines))
	}

	// 取W_1和b_1
	weightMatrixLines1 := lines[:128]
	biasVectorLines1 := lines[128:384]

	// 取W_2和b_2
	weightMatrixLines2 := lines[384:640]
	biasVectorLines2 := lines[640:]

	// 解析W_1和b_1
	matrixWeightSlice1, err := parseMatrix(weightMatrixLines1)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}
	vectorBaisSlice1, err := parseVector(biasVectorLines1)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 解析W_2和b_2
	matrixWeightSlice2, err := parseMatrix(weightMatrixLines2)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}
	vectorBaisSlice2, err := parseVector(biasVectorLines2)
	if err != nil {
		return [][]float64{}, []float64{}, [][]float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	return matrixWeightSlice1, vectorBaisSlice1, matrixWeightSlice2, vectorBaisSlice2, nil
}

// 读取文件LayerNorm
func ReadLayerNormFile(filename string) ([]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return []float64{}, []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) <= 128 {
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 取前128行
	first128Lines := lines[:128]

	// 取128行后的所有行
	remainingLines := lines[128:]

	// 解析前128行
	vectorRSlices, err := parseVector(first128Lines)
	if err != nil {
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// fmt.Printf("First 128 lines parsed successfully with dimensions: %d rows x varying columns\n", len(matrixWeightSlices))

	// 解析128行后的所有行
	vectorBSlices, err := parseVector(remainingLines)
	if err != nil {
		fmt.Println("Error parsing remaining lines:", err)
		return []float64{}, []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	return vectorRSlices, vectorBSlices, nil
}

// 读取文件LayerNorm
func ReadLayerNormSqrtVarianceFile(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return []float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return []float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 解析所有行
	vectorVaranceSlices, err := parseVector(lines)
	if err != nil {
		return []float64{}, fmt.Errorf("error reading file: %v", err)
	}
	return vectorVaranceSlices, nil
}

// 读取文件embedding和encoding
func ReadEcodeingFile(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return [][]float64{}, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	// 使用bufio.Scanner逐行读取文件
	scanner := bufio.NewScanner(file)
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	// 检查是否有读取错误
	if err := scanner.Err(); err != nil {
		return [][]float64{}, fmt.Errorf("error reading file: %v", err)
	}

	// 将文件读成矩阵
	matrixSlices, err := parseMatrix(lines)
	if err != nil {
		return [][]float64{}, fmt.Errorf("error reading file: %v", err)
	}
	return matrixSlices, nil
}

// 读取Transformer参数文件
func ReadModelParameterFile(fileDir string) (DashformerModelParameters, error) {

	// 读取文件embedding
	embeddingData, err := ReadEcodeingFile(fileDir + "/embedding_Embedding_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("embedding: (%d , %d)\n", len(embeddingData), len(embeddingData[0]))

	// 读取文件encoding
	encodingData, err := ReadEcodeingFile(fileDir + "/positional_encoding_Lookup.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("encoding: (%d , %d)\n", len(encodingData), len(encodingData[0]))

	// 读取文件W_Q, W_K, W_V
	queryWeightMatrixs, queryBiasVectors, err := ReadMultiAttentionFile(fileDir + "/transformer_block_Query_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("queryWeightMatrixs: (%d, %d, %d)   ", len(queryWeightMatrixs), len(queryWeightMatrixs[0]), len(queryWeightMatrixs[0][0]))
	// fmt.Printf("queryBiasVectors: (%d, %d, )\n", len(queryBiasVectors), len(queryBiasVectors[0]))

	keyWeightMatrixs, keyBiasVectors, err := ReadMultiAttentionFile(fileDir + "/transformer_block_Key_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("keyWeightMatrixs: (%d, %d, %d)   ", len(keyWeightMatrixs), len(keyWeightMatrixs[0]), len(keyWeightMatrixs[0][0]))
	// fmt.Printf("keyBiasVectors: (%d, %d, )\n", len(keyBiasVectors), len(keyBiasVectors[0]))

	valueWeightMatrixs, valueBiasVectors, err := ReadMultiAttentionFile(fileDir + "/transformer_block_Value_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("valueWeightMatrixs: (%d, %d, %d)   ", len(valueWeightMatrixs), len(valueWeightMatrixs[0]), len(valueWeightMatrixs[0][0]))
	// fmt.Printf("valueBiasVectors: (%d, %d, )\n", len(valueBiasVectors), len(valueBiasVectors[0]))

	// 读取文件combineHead和classifier
	combineWeightMatrixs, combineBiasVectors, err := ReadCombineAndClassifierFile(fileDir + "/transformer_block_CombineHead_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("combineWeightMatrixs: (%d, %d)   ", len(combineWeightMatrixs), len(combineWeightMatrixs[0]))
	// fmt.Printf("combineBiasVectors: (%d, )\n", len(combineBiasVectors))

	classifierWeightMatrixs, classifierBiasVectors, err := ReadCombineAndClassifierFile(fileDir + "/Dense_Classifier_DenseClassifier_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("classifierWeightMatrixs: (%d, %d)   ", len(classifierWeightMatrixs), len(classifierWeightMatrixs[0]))
	// fmt.Printf("classifierBiasVectors: (%d, )\n", len(classifierBiasVectors))

	// 读取文件LayerNorm
	LayerNormMatrixsR1, LayerNormMatrixsB1, err := ReadLayerNormFile(fileDir + "/transformer_block_LayerNorm1_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("LayerNormMatrixsR1: (%d, )   ", len(LayerNormMatrixsR1))
	// fmt.Printf("LayerNormMatrixsB1: (%d, )\n", len(LayerNormMatrixsB1))

	LayerNormMatrixsR2, LayerNormMatrixsB2, err := ReadLayerNormFile(fileDir + "/transformer_block_LayerNorm2_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	// fmt.Printf("LayerNormMatrixsR1: (%d, )   ", len(LayerNormMatrixsR2))
	// fmt.Printf("LayerNormMatrixsB1: (%d, )\n", len(LayerNormMatrixsB2))

	// 读取文件FeedForward
	feedForwardWeightMatrix1, feedForwardBiasVector1, feedForwardWeightMatrix2, feedForwardBiasVector2, err := ReadFeedFowardFile(fileDir + "/transformer_block_FFN_weights.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}

	// dashModelParam.PrintDimensions()

	// 设置多项式拟合系数
	reluCoefficients, sqrtLayerCoefficients1, sqrtLayerCoefficients2, softMaxB, softMaxC := config.InitCoeffients()
	layerNormSqrtVariance1, err := ReadLayerNormSqrtVarianceFile(fileDir + "/layerNorm1_Reciprocal_SqrtVariance.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}
	layerNormSqrtVariance2, err := ReadLayerNormSqrtVarianceFile(fileDir + "/layerNorm2_Reciprocal_SqrtVariance.txt")
	if err != nil {
		return DashformerModelParameters{}, fmt.Errorf("read model parameters: %w", err)
	}

	return DashformerModelParameters{
		EmbeddingMatrix: embeddingData,
		EncodingMatrix:  encodingData,

		QueryWeightAttentionMatrixs: queryWeightMatrixs,
		QueryBiasAttentionVectors:   queryBiasVectors,
		KeyWeightAttentionMatrixs:   keyWeightMatrixs,
		KeyBiasAttentionVectors:     keyBiasVectors,
		ValueWeightAttentionMatrixs: valueWeightMatrixs,
		ValueBiasAttentionVectors:   valueBiasVectors,

		CombineWeightMatrixs: combineWeightMatrixs,
		CombineBiasVectors:   combineBiasVectors,

		LayerNormVectorR1: LayerNormMatrixsR1,
		LayerNormVectorB1: LayerNormMatrixsB1,
		LayerNormVectorR2: LayerNormMatrixsR2,
		LayerNormVectorB2: LayerNormMatrixsB2,

		FeedForwardWeightMatrix1: feedForwardWeightMatrix1,
		FeedForwardBiasVector1:   feedForwardBiasVector1,
		FeedForwardWeightMatrix2: feedForwardWeightMatrix2,
		FeedForwardBiasVector2:   feedForwardBiasVector2,

		ClassifierWeightMatrix: classifierWeightMatrixs,
		ClassifierBiasVector:   classifierBiasVectors,

		ReluCoefficients:       reluCoefficients,
		SqrtLayerCoefficients1: sqrtLayerCoefficients1,
		SqrtLayerCoefficients2: sqrtLayerCoefficients2,

		LayerNormSqrtVariance1: layerNormSqrtVariance1,
		LayerNormSqrtVariance2: layerNormSqrtVariance2,
		SoftMaxB:               softMaxB,
		SoftMaxC:               softMaxC,
	}, nil
}

// 写三维张量到文件中

func WriteResultToFile(fileDir string, valueTensor [][][]float64) error {
	if err := os.MkdirAll(fileDir, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	// 打开文件用于写入
	file, err := os.Create(filepath.Join(fileDir, "output.txt"))
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer func() {
		_ = file.Close()
	}()
	//

	// 遍历 valueTensor 并写入文件
	for i := range valueTensor {
		for j := range valueTensor[i][0] {
			if j != 0 {
				if _, err := fmt.Fprint(file, "\t"); err != nil { // 在每个值之间添加制表符作为分隔符
					return fmt.Errorf("write output separator: %w", err)
				}
			}
			valueTensor[i][0][j] = valueTensor[i][0][j] * 2649.372705
			if _, err := fmt.Fprint(file, valueTensor[i][0][j]); err != nil {
				return fmt.Errorf("write output value: %w", err)
			}
		}
		if _, err := fmt.Fprintln(file); err != nil { // 每行结束后写入一个换行符
			return fmt.Errorf("write output newline: %w", err)
		}
	}
	return nil
}

// 新增：写分类结果到文件（对每个样本做 argmax 并输出预测类索引和置信度）
func WriteClassificationToFile(fileDir string, valueTensor [][][]float64) {
	file, err := os.Create(fileDir + "/classification.txt")
	if err != nil {
		fmt.Println("Error creating classification file:", err)
		return
	}
	defer file.Close()

	for i := range valueTensor {
		// 保护性检查
		if len(valueTensor[i]) == 0 || len(valueTensor[i][0]) == 0 {
			fmt.Fprintln(file, "-	-")
			continue
		}
		// 取第一行作为输出向量（不要对 logits 做额外放大）
		row := make([]float64, len(valueTensor[i][0]))
		copy(row, valueTensor[i][0])

		// 计算 softmax 概率并取 argmax（在概率上取最大）
		probs := softmax(row)
		maxIdx := 0
		maxProb := probs[0]
		for j := 1; j < len(probs); j++ {
			if probs[j] > maxProb {
				maxProb = probs[j]
				maxIdx = j
			}
		}
		// 只写预测类别索引和置信度
		fmt.Fprintf(file, "%d\t%.6f\n", maxIdx, maxProb)
	}
}

// 新增：写详细分类结果到文件（现在只包含 pred_index 和 pred_prob，保证置信度合理）
func WriteClassificationDetailed(fileDir string, valueTensor [][][]float64, topK int) error {
	// 创建输出目录
	if err := os.MkdirAll(fileDir, 0o755); err != nil {
		return err
	}
	fp := filepath.Join(fileDir, "classification.txt")
	f, err := os.Create(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	// 写表头（只包含预测类别与置信度）
	if _, err := fmt.Fprintln(f, strings.Join([]string{"pred_index", "pred_prob"}, "\t")); err != nil {
		return err
	}

	// 逐样本处理
	for i := range valueTensor {
		if len(valueTensor[i]) == 0 || len(valueTensor[i][0]) == 0 {
			fmt.Fprintln(f, "-\t-")
			continue
		}
		vec := make([]float64, len(valueTensor[i][0]))
		copy(vec, valueTensor[i][0])

		// 不对 logits 做额外放大，直接计算 softmax
		probs := softmax(vec)

		// 在概率上取最大
		maxIdx := 0
		maxProb := probs[0]
		for j := 1; j < len(probs); j++ {
			if probs[j] > maxProb {
				maxProb = probs[j]
				maxIdx = j
			}
		}

		// 只输出预测类别编号和置信度
		if _, err := fmt.Fprintf(f, "%d\t%.6f\n", maxIdx, maxProb); err != nil {
			return err
		}
	}
	return nil
}

// 下面是工具函数，供 WriteClassificationDetailed 使用
func argmax(vec []float64) (int, float64) {
	bestIdx := 0
	best := vec[0]
	for i := 1; i < len(vec); i++ {
		if vec[i] > best {
			best = vec[i]
			bestIdx = i
		}
	}
	return bestIdx, best
}

func softmax(vec []float64) []float64 {
	// numerically stable softmax
	maxv := vec[0]
	for _, v := range vec {
		if v > maxv {
			maxv = v
		}
	}
	exps := make([]float64, len(vec))
	sum := 0.0
	for i, v := range vec {
		e := math.Exp(v - maxv)
		exps[i] = e
		sum += e
	}
	if sum == 0 {
		out := make([]float64, len(vec))
		for i := range out {
			out[i] = 1.0 / float64(len(vec))
		}
		return out
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

type kv struct {
	idx int
	v   float64
}

func topKFrom(probs []float64, k int) ([]int, []float64) {
	if k <= 0 {
		return nil, nil
	}
	if k > len(probs) {
		k = len(probs)
	}
	arr := make([]kv, len(probs))
	for i, v := range probs {
		arr[i] = kv{idx: i, v: v}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].v > arr[j].v })
	idxs := make([]int, k)
	vals := make([]float64, k)
	for i := 0; i < k; i++ {
		idxs[i] = arr[i].idx
		vals[i] = arr[i].v
	}
	return idxs, vals
}
