package main

import (
	"reflect"
	"testing"
)

func TestPredictionsFromProbabilities(t *testing.T) {
	tests := []struct {
		name          string
		probabilities any
		expectedRows  int
		expected      []int
	}{
		{
			name:          "class zero predictions",
			probabilities: [][]float64{{0.9, 0.1}, {0.8, 0.2}},
			expectedRows:  2,
			expected:      []int{0, 0},
		},
		{
			name:          "class one predictions",
			probabilities: [][]float32{{0.1, 0.9}, {0.2, 0.8}},
			expectedRows:  2,
			expected:      []int{1, 1},
		},
		{
			name:          "mixed predictions",
			probabilities: [][]float64{{0.9, 0.1}, {0.1, 0.9}},
			expectedRows:  2,
			expected:      []int{0, 1},
		},
		{
			name:          "equal probabilities",
			probabilities: [][]float64{{0.5, 0.5}},
			expectedRows:  1,
			expected:      []int{0},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual, err := predictionsFromProbabilities(
				test.probabilities,
				test.expectedRows,
			)

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if !reflect.DeepEqual(actual, test.expected) {
				t.Errorf(
					"expected %v, received %v",
					test.expected,
					actual,
				)
			}
		})
	}
}

func TestPredictionsFromProbabilitiesInvalidInput(t *testing.T) {
	_, err := predictionsFromProbabilities(
		[][]float64{{0.9, 0.1}},
		2,
	)

	if err == nil {
		t.Error("expected an error for mismatched row count")
	}
}
