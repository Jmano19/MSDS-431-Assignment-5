
"""
Train a neural network for breast cancer classification.

The model is trained on standardized Wisconsin Breast Cancer data.
The scaler and model are saved separately so Go can reproduce
the same preprocessing before ONNX inference.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

for directory in (MODEL_DIR, DATA_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def main():
    # Load the built-in Wisconsin Breast Cancer dataset.
    dataset = load_breast_cancer()

    features = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )
    labels = dataset.target

    # Create reproducible training and testing datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    # Standardize using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a feedforward neural network.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=30,
        random_state=42
    )

    print("Training neural network...")
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on previously unseen observations.
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel evaluation")
    print("----------------")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations:  {len(X_test)}")
    print(f"Input features:        {X_train.shape[1]}")
    print(f"Hidden layers:         {model.hidden_layer_sizes}")
    print(f"Training iterations:   {model.n_iter_}")
    print(f"Test accuracy:        {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names
        )
    )

    # Save the trained model and preprocessing parameters.
    joblib.dump(model, MODEL_DIR / "breast_cancer_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")

    # Save scaler parameters for Go preprocessing.
    scaler_parameters = {
        "feature_names": list(dataset.feature_names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }

    with open(
        MODEL_DIR / "scaler_parameters.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(scaler_parameters, file, indent=4)

    # Save the original, unscaled test observations for Go.
    X_test.to_csv(
        DATA_DIR / "test_features.csv",
        index=False
    )

    pd.DataFrame({
        "actual_label": y_test,
        "python_prediction": predictions
    }).to_csv(
        RESULTS_DIR / "python_predictions.csv",
        index=False
    )

    # Save model evaluation metrics.
    metrics = {
        "model": "Scikit-Learn MLPClassifier",
        "dataset": "Wisconsin Breast Cancer",
        "training_observations": len(X_train),
        "testing_observations": len(X_test),
        "input_features": X_train.shape[1],
        "hidden_layers": list(model.hidden_layer_sizes),
        "training_iterations": model.n_iter_,
        "test_accuracy": float(accuracy)
    }

    with open(
        RESULTS_DIR / "model_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved model, scaler, test data, and evaluation results.")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
