
"""
Export the trained MLP neural network to ONNX.

StandardScaler preprocessing is performed separately.
Only the neural network probability output is exported
to avoid unsupported ONNX label-generation operators.

The exported model is validated against the original
Python model and benchmarked using ONNX Runtime.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_PATH = MODEL_DIR / "breast_cancer_model.onnx"


def main():
    print("Loading trained model and scaler...")

    model = joblib.load(MODEL_DIR / "breast_cancer_model.joblib")
    scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")

    X_test = pd.read_csv(DATA_DIR / "test_features.csv")

    # Standardize features outside the ONNX model.
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    initial_types = [
        ("float_input", FloatTensorType([None, 30]))
    ]

    print("Converting neural network to ONNX...")

    # Export the MLP itself, not the full classifier pipeline.
    # zipmap=False produces a standard probability tensor.
    mlp = model

    onnx_model = convert_sklearn(
        mlp,
        initial_types=initial_types,
        options={id(mlp): {"zipmap": False}},
        target_opset=15,
    )

    # Remove the predicted-label output. Go will calculate
    # the predicted class from the two class probabilities.
    probability_output = onnx_model.graph.output[-1]
    del onnx_model.graph.output[:]
    onnx_model.graph.output.append(probability_output)

    onnx.checker.check_model(onnx_model)

    with open(MODEL_PATH, "wb") as file:
        file.write(onnx_model.SerializeToString())

    print(f"ONNX model saved: {MODEL_PATH}")

    # Load the exported model using ONNX Runtime.
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    print(f"Model input: {input_name}")
    print(f"Model outputs: {output_names}")

    # Warm up ONNX Runtime before measuring inference.
    session.run(
        None,
        {input_name: X_test_scaled}
    )

    # Benchmark repeated inference over the same test observations.
    benchmark_runs = 100

    start = time.perf_counter()

    for _ in range(benchmark_runs):
        probabilities = session.run(
            None,
            {input_name: X_test_scaled}
        )[0]

    inference_duration = (
        time.perf_counter() - start
    ) / benchmark_runs

    inference_ms = inference_duration * 1000
    inference_us = inference_duration * 1_000_000

    print(f"\nPython ONNX Runtime benchmark ({benchmark_runs} runs)")
    print(f"Average inference time: {inference_ms:.3f} ms")
    print(f"Average inference time: {inference_us:.0f} microseconds")

    # Convert probability outputs into predicted class labels.
    print("\nModel class ordering:", model.classes_)
    print("ONNX probability shape:", probabilities.shape)

    onnx_predictions = model.classes_[
        np.argmax(probabilities, axis=1)
    ].astype(np.int64)

    # Validate against the original Python model.
    python_predictions = model.predict(
        X_test_scaled
    ).astype(np.int64)

    matches = np.sum(onnx_predictions == python_predictions)
    total = len(python_predictions)
    agreement = matches / total * 100

    print("\nONNX validation results")
    print("-----------------------")
    print(f"Total predictions: {total}")
    print(f"Matching predictions: {matches}")
    print(f"Prediction agreement: {agreement:.2f}%")

    if matches != total:
        raise RuntimeError(
            "ONNX predictions do not match Python predictions."
        )

    # Save prediction comparison.
    pd.DataFrame({
        "python_prediction": python_predictions,
        "onnx_prediction": onnx_predictions
    }).to_csv(
        RESULTS_DIR / "onnx_predictions.csv",
        index=False
    )

    # Save validation and benchmark metadata.
    metadata = {
        "format": "ONNX",
        "runtime": "ONNX Runtime",
        "backend": "CPUExecutionProvider",
        "benchmark_runs": benchmark_runs,
        "inference_time_ms": inference_ms,
        "inference_time_microseconds": inference_us,
        "input_name": input_name,
        "input_features": 30,
        "output_names": output_names,
        "output_type": "Class probabilities",
        "preprocessing": "StandardScaler applied externally",
        "test_observations": total,
        "matching_predictions": int(matches),
        "prediction_agreement_percent": agreement,
        "opset_version": 15
    }

    with open(
        RESULTS_DIR / "onnx_validation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nONNX export and validation completed successfully.")
    print("Validation metadata saved to results/onnx_validation.json")


if __name__ == "__main__":
    main()
