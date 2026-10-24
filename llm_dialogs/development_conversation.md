# LLM-Assisted Development Conversation Log

## Purpose

This document records the use of an LLM (ChatGPT) during the development of Assignment 5. The LLM was used as a development assistant for implementation guidance, troubleshooting, testing, and documentation. All code was reviewed, executed, and validated by the student.

## 1. Python Model Development

**Topic:** Training a machine learning classifier.

**Development work:**
- Selected the Wisconsin Breast Cancer dataset from scikit-learn.
- Developed an MLP classification model using scikit-learn.
- Implemented feature standardization and training/testing separation.
- Saved the trained model, scaler, and prediction results.

**LLM assistance:** Guidance on structuring the Python training workflow, selecting model components, and saving model artifacts for interoperability.

## 2. ONNX Export and Compatibility

**Topic:** Exporting the trained model for use outside Python.

**Development work:**
- Used `skl2onnx` to export the trained neural network.
- Used ONNX Runtime to validate the exported model.
- Separated feature standardization from the ONNX model.
- Adjusted ONNX outputs to expose class probabilities rather than classifier labels.

**LLM assistance:** Troubleshooting ONNX operator compatibility issues and helping adapt the exported model to the operators supported by GoMLX.

## 3. GoMLX Implementation

**Topic:** Running ONNX inference in Go.

**Development work:**
- Integrated GoMLX and ONNX-GoMLX into the Go application.
- Loaded the ONNX model and scaler parameters.
- Implemented feature standardization in Go.
- Converted probability outputs into predicted classes.
- Generated Go prediction and metrics files.

**LLM assistance:** Guidance on GoMLX integration, dependency compatibility, tensor handling, and ONNX inference implementation.

## 4. Testing and Benchmarking

**Topic:** Validating prediction consistency and comparing runtimes.

**Development work:**
- Added Go unit tests for probability conversion and invalid input.
- Built and executed a standalone Go executable.
- Compared Python and Go predictions.
- Benchmarked both runtimes using 100 inference runs after warm-up.

**LLM assistance:** Assistance developing test cases, troubleshooting execution errors, and documenting benchmark methodology and results.

## 5. Student Validation

The student executed the Python and Go applications, reviewed generated outputs, ran the Go unit tests, and built and executed the standalone Windows binary.

Final validation showed:
- 114 test observations.
- 100% prediction agreement between Python and Go.
- 97.37% test accuracy.
- Successful Go unit tests.
- Successful standalone executable build and execution.

The LLM served as a development aid. The student was responsible for reviewing the implementation, executing the code, and validating the results.
