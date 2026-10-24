package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

type Metrics struct {
	TotalPredictions int     `json:"total_predictions"`
	Matching         int     `json:"matching_predictions"`
	Agreement        float64 `json:"prediction_agreement_percent"`
	GoAccuracy       float64 `json:"go_accuracy_percent"`
	InferenceMS      float64 `json:"inference_time_ms"`
	InferenceUS      float64 `json:"inference_time_microseconds"`
	Backend          string  `json:"backend"`
}

type ScalerParameters struct {
	Mean  []float64 `json:"mean"`
	Scale []float64 `json:"scale"`
}

func readFeatures(path string) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("test feature CSV has no data rows")
	}

	var data [][]float32

	for rowIndex, row := range rows[1:] {
		values := make([]float32, len(row))

		for columnIndex, value := range row {
			number, err := strconv.ParseFloat(strings.TrimSpace(value), 32)
			if err != nil {
				return nil, fmt.Errorf(
					"invalid feature at row %d, column %d: %w",
					rowIndex+2, columnIndex+1, err,
				)
			}

			values[columnIndex] = float32(number)
		}

		data = append(data, values)
	}

	return data, nil
}

func readScalerParameters(path string) ScalerParameters {
	data, err := os.ReadFile(path)
	check(err)

	var params ScalerParameters
	check(json.Unmarshal(data, &params))

	if len(params.Mean) != 30 || len(params.Scale) != 30 {
		check(fmt.Errorf("expected 30 scaler parameters"))
	}

	for i, scale := range params.Scale {
		if scale == 0 {
			check(fmt.Errorf("scaler value at feature %d is zero", i))
		}
	}

	return params
}

func readPythonPredictions(path string) ([]int, []int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	rows, err := csv.NewReader(file).ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("Python predictions CSV has no data rows")
	}

	header := rows[0]
	actualColumn, predictedColumn := -1, -1

	for i, name := range header {
		switch strings.ToLower(strings.TrimSpace(name)) {
		case "actual", "true", "target", "y_true", "actual_label":
			actualColumn = i
		case "predicted", "prediction", "pred", "y_pred",
			"predicted_label", "python_prediction":
			predictedColumn = i
		}
	}

	if actualColumn < 0 || predictedColumn < 0 {
		return nil, nil, fmt.Errorf(
			"could not identify actual and predicted columns in %s; headers: %v",
			path, header,
		)
	}

	var actual, predicted []int

	for rowIndex, row := range rows[1:] {
		if actualColumn >= len(row) || predictedColumn >= len(row) {
			return nil, nil, fmt.Errorf("incomplete prediction row %d", rowIndex+2)
		}

		a, err := strconv.Atoi(strings.TrimSpace(row[actualColumn]))
		if err != nil {
			return nil, nil, err
		}

		p, err := strconv.Atoi(strings.TrimSpace(row[predictedColumn]))
		if err != nil {
			return nil, nil, err
		}

		actual = append(actual, a)
		predicted = append(predicted, p)
	}

	return actual, predicted, nil
}

func flattenLabels(value any) ([]int, error) {
	var labels []int

	var visit func(reflect.Value) error

	visit = func(v reflect.Value) error {
		if !v.IsValid() {
			return fmt.Errorf("empty tensor output")
		}

		if v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer {
			if v.IsNil() {
				return fmt.Errorf("nil tensor output")
			}
			return visit(v.Elem())
		}

		if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
			for i := 0; i < v.Len(); i++ {
				if err := visit(v.Index(i)); err != nil {
					return err
				}
			}
			return nil
		}

		switch v.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16,
			reflect.Int32, reflect.Int64:
			labels = append(labels, int(v.Int()))

		case reflect.Uint, reflect.Uint8, reflect.Uint16,
			reflect.Uint32, reflect.Uint64:
			labels = append(labels, int(v.Uint()))

		default:
			return fmt.Errorf(
				"unexpected prediction value type: %s", v.Type(),
			)
		}

		return nil
	}

	if err := visit(reflect.ValueOf(value)); err != nil {
		return nil, err
	}

	return labels, nil
}

func predictionsFromProbabilities(
	value any,
	expectedRows int,
) ([]int, error) {
	var probabilities []float64

	var visit func(reflect.Value) error

	visit = func(v reflect.Value) error {
		if !v.IsValid() {
			return fmt.Errorf("empty probability output")
		}

		if v.Kind() == reflect.Interface || v.Kind() == reflect.Pointer {
			if v.IsNil() {
				return fmt.Errorf("nil probability output")
			}
			return visit(v.Elem())
		}

		switch v.Kind() {
		case reflect.Slice, reflect.Array:
			for i := 0; i < v.Len(); i++ {
				if err := visit(v.Index(i)); err != nil {
					return err
				}
			}

		case reflect.Float32, reflect.Float64:
			probabilities = append(probabilities, v.Float())

		default:
			return fmt.Errorf(
				"unexpected probability value type: %s",
				v.Type(),
			)
		}

		return nil
	}

	if err := visit(reflect.ValueOf(value)); err != nil {
		return nil, err
	}

	if len(probabilities) != expectedRows*2 {
		return nil, fmt.Errorf(
			"expected %d probability values, received %d",
			expectedRows*2,
			len(probabilities),
		)
	}

	predictions := make([]int, expectedRows)

	for i := 0; i < expectedRows; i++ {
		class0 := probabilities[i*2]
		class1 := probabilities[i*2+1]

		if class1 > class0 {
			predictions[i] = 1
		} else {
			predictions[i] = 0
		}
	}

	return predictions, nil
}

func writePredictions(
	path string,
	actual []int,
	predictions []int,
	pythonPredictions []int,
) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write([]string{
		"actual",
		"go_prediction",
		"python_prediction",
		"match",
	}); err != nil {
		return err
	}

	for i, prediction := range predictions {
		match := "false"

		if prediction == pythonPredictions[i] {
			match = "true"
		}

		if err := writer.Write([]string{
			strconv.Itoa(actual[i]),
			strconv.Itoa(prediction),
			strconv.Itoa(pythonPredictions[i]),
			match,
		}); err != nil {
			return err
		}
	}

	return writer.Error()
}

func main() {
	root := ".."

	modelPath := filepath.Join(
		root, "models", "breast_cancer_model.onnx",
	)
	featuresPath := filepath.Join(
		root, "data", "test_features.csv",
	)
	scalerPath := filepath.Join(
		root, "models", "scaler_parameters.json",
	)
	pythonPredictionsPath := filepath.Join(
		root, "results", "python_predictions.csv",
	)
	outputPath := filepath.Join(
		root, "results", "go_predictions.csv",
	)
	metricsPath := filepath.Join(
		root, "results", "go_metrics.json",
	)

	// Load test observations.
	fmt.Println("Loading test observations...")

	inputData, err := readFeatures(featuresPath)
	check(err)

	actual, pythonPredictions, err := readPythonPredictions(
		pythonPredictionsPath,
	)
	check(err)

	if len(inputData) != len(actual) ||
		len(actual) != len(pythonPredictions) {
		check(fmt.Errorf(
			"row count mismatch: features=%d, actual=%d, Python predictions=%d",
			len(inputData), len(actual), len(pythonPredictions),
		))
	}

	if len(inputData[0]) != 30 {
		check(fmt.Errorf(
			"expected 30 features per observation, got %d",
			len(inputData[0]),
		))
	}

	fmt.Printf(
		"Loaded %d observations with %d features each.\n",
		len(inputData), len(inputData[0]),
	)

	// Load Python StandardScaler parameters.
	fmt.Println("Loading scaler parameters...")

	scaler := readScalerParameters(scalerPath)

	// Apply the same standardization used during Python training.
	for i := range inputData {
		for j := range inputData[i] {
			inputData[i][j] = (inputData[i][j] - float32(scaler.Mean[j])) / float32(scaler.Scale[j])
		}
	}

	fmt.Println("Feature standardization completed.")

	// Load ONNX neural network.
	fmt.Println("Loading ONNX model...")

	onnxModel, err := parser.ParseFile(modelPath)
	check(err)
	defer onnxModel.Close()

	inputNames, _ := onnxModel.Inputs()
	outputNames, _ := onnxModel.Outputs()

	fmt.Printf("ONNX inputs: %v\n", inputNames)
	fmt.Printf("ONNX outputs: %v\n", outputNames)

	if len(inputNames) != 1 {
		check(fmt.Errorf(
			"expected one ONNX input, found %d",
			len(inputNames),
		))
	}

	// Initialize GoMLX.
	fmt.Println("Initializing GoMLX backend...")

	backend, err := compute.New()
	check(err)
	defer backend.Finalize()

	fmt.Printf("Backend: %s\n", backend.Name())

	store := model.NewStore()
	check(onnxModel.VariablesToScope(store.RootScope()))

	// Convert standardized features to a GoMLX tensor.
	inputTensor := tensors.FromValue(inputData)

	// Run ONNX inference through GoMLX.
	fmt.Println("Running Go inference...")

	execFn := func(scope *model.Scope, inputs []*graph.Node) []*graph.Node {
		return onnxModel.CallGraph(
			scope,
			inputs[0].Graph(),
			map[string]*graph.Node{
				inputNames[0]: inputs[0],
			},
			"probabilities",
		)
	}

	// Warm up once to exclude initial graph compilation.
	_, err = model.CallOnceN(
		backend,
		store,
		execFn,
		inputTensor,
	)
	check(err)

	// Measure repeated inference over the same observations.
	const benchmarkRuns = 100

	var outputTensors []*tensors.Tensor

	start := time.Now()

	for i := 0; i < benchmarkRuns; i++ {
		outputTensors, err = model.CallOnceN(
			backend,
			store,
			execFn,
			inputTensor,
		)
		check(err)
	}

	inferenceDuration := time.Since(start) / benchmarkRuns

	if len(outputTensors) != 1 {
		check(fmt.Errorf(
			"expected one output tensor, received %d",
			len(outputTensors),
		))
	}

	outputValue, err := outputTensors[0].ValueSafe()
	check(err)

	goPredictions, err := predictionsFromProbabilities(
		outputValue,
		len(inputData),
	)
	check(err)

	if len(goPredictions) != len(actual) {
		check(fmt.Errorf(
			"Go returned %d predictions; expected %d",
			len(goPredictions), len(actual),
		))
	}

	// Compare Go predictions against Python and actual labels.
	matching := 0
	correct := 0

	for i, prediction := range goPredictions {
		if prediction == pythonPredictions[i] {
			matching++
		}

		if prediction == actual[i] {
			correct++
		}
	}

	agreement := float64(matching) / float64(len(actual)) * 100
	accuracy := float64(correct) / float64(len(actual)) * 100

	durationMS := float64(inferenceDuration.Microseconds()) / 1000
	durationUS := float64(inferenceDuration.Microseconds())

	// Save performance and accuracy metrics.
	metrics := Metrics{
		TotalPredictions: len(goPredictions),
		Matching:         matching,
		Agreement:        agreement,
		GoAccuracy:       accuracy,
		InferenceMS:      durationMS,
		InferenceUS:      durationUS,
		Backend:          backend.Name(),
	}

	check(writePredictions(
		outputPath,
		actual,
		goPredictions,
		pythonPredictions,
	))

	metricsFile, err := os.Create(metricsPath)
	check(err)

	encoder := json.NewEncoder(metricsFile)
	encoder.SetIndent("", "  ")

	err = encoder.Encode(metrics)
	closeErr := metricsFile.Close()

	check(err)
	check(closeErr)

	// Display results.
	fmt.Println("\nGo inference results")
	fmt.Println("-----------------------")

	fmt.Printf("Total predictions: %d\n", metrics.TotalPredictions)
	fmt.Printf(
		"Matching Python predictions: %d/%d\n",
		matching, len(actual),
	)
	fmt.Printf("Prediction agreement: %.2f%%\n", agreement)
	fmt.Printf("Go prediction accuracy: %.2f%%\n", accuracy)

	fmt.Printf(
		"Inference time: %.3f ms (%.0f microseconds)\n",
		metrics.InferenceMS, metrics.InferenceUS,
	)

	fmt.Printf("Go predictions saved to: %s\n", outputPath)
	fmt.Printf("Go metrics saved to: %s\n", metricsPath)
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}
