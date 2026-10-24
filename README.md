# MSDS 431 - Assignment 5: Python and Go Machine Learning Interoperability

## Project Overview

This project demonstrates machine learning interoperability between Python and Go by training a classification model in Python, exporting it to the ONNX (Open Neural Network Exchange) format, and performing inference in Go using GoMLX.

The project compares prediction consistency and inference performance between Python's ONNX Runtime and GoMLX.

The Wisconsin Breast Cancer dataset from scikit-learn is used to train and evaluate a binary classification model.

## Project Architecture

The workflow consists of five main stages:

1. **Model training:** Train a Multi-Layer Perceptron (MLP) classifier in Python using scikit-learn.
2. **Model export:** Export the trained neural network to ONNX format using `skl2onnx`.
3. **Python inference:** Run the exported ONNX model using ONNX Runtime and benchmark its performance.
4. **Go inference:** Load the same ONNX model in Go using GoMLX, standardize the input features, and generate predictions.
5. **Validation:** Compare Python and Go predictions against each other and the actual test labels.

Feature standardization is performed outside the ONNX model. The scaler parameters are exported to JSON and applied in Go before inference.

## Machine Learning Model

- **Dataset:** Wisconsin Breast Cancer dataset
- **Source:** scikit-learn
- **Model:** Multi-Layer Perceptron (MLP) classifier
- **Hidden layers:** 64 and 32 neurons
- **Activation:** ReLU
- **Optimizer:** Adam
- **Training/testing split:** 80% training and 20% testing
- **Random state:** 42
- **Training observations:** 455
- **Testing observations:** 114
- **Input features:** 30

The model achieved **97.37% test accuracy**.

## Technologies and Libraries

### Python

- Python
- NumPy
- Pandas
- scikit-learn
- skl2onnx
- ONNX
- ONNX Runtime
- Joblib
- psutil

### Go

- Go
- GoMLX
- ONNX-GoMLX
- Go compute backend

The Go implementation uses the pure Go backend, avoiding a CGO dependency.

## Repository Structure

```text
Assignment 5/
│
├── data/
│   └── test_features.csv
│
├── go/
│   ├── main.go
│   ├── main_test.go
│   ├── go.mod
│   └── go.sum
│
├── llm_dialogs/
│   └── LLM conversation documentation
│
├── models/
│   ├── breast_cancer_model.joblib
│   ├── feature_scaler.joblib
│   ├── breast_cancer_model.onnx
│   └── scaler_parameters.json
│
├── python/
│   ├── train_model.py
│   └── export_onnx.py
│
├── results/
│   ├── model_metrics.json
│   ├── python_predictions.csv
│   ├── onnx_predictions.csv
│   ├── on