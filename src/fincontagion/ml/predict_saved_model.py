
from pathlib import Path
import json
import hashlib

import joblib
import pandas as pd


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

DATASET_PATH = DATA_DIR / "ml_scenario_dataset.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_PATHS = {
    "Logistic Regression": (
        MODEL_DIR / "logistic_regression_pipeline.joblib"
    ),
    "Random Forest": (
        MODEL_DIR / "random_forest_pipeline.joblib"
    ),
}

OUTPUT_PATH = DATA_DIR / "ml_saved_model_predictions.csv"


# ---------------------------------------------------------
# 2. Validate artifacts and dataset
# ---------------------------------------------------------

def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    if metadata.get("project") != "FINCONTAGION-X":
        raise ValueError("Metadata belongs to an unexpected project.")

    if metadata.get("target_column") != "distress_label":
        raise ValueError("Unexpected target definition in metadata.")

    return metadata


def load_prediction_data(metadata):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    current_hash = hashlib.sha256(
        DATASET_PATH.read_bytes()
    ).hexdigest()

    expected_hash = metadata.get("dataset_sha256")

    if current_hash != expected_hash:
        raise ValueError(
            "Dataset hash does not match the training metadata. "
            "The dataset may have changed since the models were trained. "
            "Retrain the models or restore the matching dataset."
        )

    df = pd.read_csv(DATASET_PATH)

    required = {
        "entity_id",
        "scenario_id",
        "distress_label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    feature_columns = metadata["feature_columns"]

    missing_features = set(feature_columns) - set(df.columns)

    if missing_features:
        raise ValueError(
            "Dataset is missing model features: "
            f"{sorted(missing_features)}"
        )

    X = df[feature_columns].copy()

    y = df["distress_label"].astype(int)

    return df, X, y


# ---------------------------------------------------------
# 3. Load saved pipelines and predict
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("FINCONTAGION-X: SAVED MODEL INFERENCE")
    print("=" * 65)

    metadata = load_metadata()
    df, X, y = load_prediction_data(metadata)

    print(f"\nDataset rows: {len(df)}")
    print(f"Unique entities: {df['entity_id'].nunique()}")
    print(f"Training seed in metadata: {metadata['training_seed']}")

    # This is a reproduction check on the same dataset used
    # for training/evaluation, not an independent test.
    print("\nDataset hash verified.")

    predictions = df[
        ["entity_id", "scenario_id", "distress_label"]
    ].copy()

    predictions = predictions.rename(
        columns={"distress_label": "actual_distress"}
    )

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}"
            )

        print(f"\nLoading {model_name}...")
        pipeline = joblib.load(model_path)

        saved_feature_columns = metadata["models"][
            model_name
        ]["feature_columns"]

        if list(X.columns) != saved_feature_columns:
            raise ValueError(
                f"Feature schema mismatch for {model_name}."
            )

        probabilities = pipeline.predict_proba(X)[:, 1]
        predicted_labels = pipeline.predict(X)

        prefix = (
            "logistic_regression"
            if model_name == "Logistic Regression"
            else "random_forest"
        )

        predictions[f"{prefix}_probability"] = probabilities
        predictions[f"{prefix}_prediction"] = predicted_labels

        print(f"Loaded classifier: {type(pipeline.named_steps['classifier']).__name__}")
        print(f"Predictions generated: {len(predicted_labels)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved inference output:\n{OUTPUT_PATH}")

    print("\nSample output:")
    print(predictions.head(10).to_string(index=False))

    print(
        "\nIMPORTANT: These predictions use synthetic scenario data. "
        "This run checks model loading and inference, not real-world "
        "predictive validity."
    )


if __name__ == "__main__":
    main()
