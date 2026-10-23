
from pathlib import Path

import numpy as np
import pandas as pd

from fincontagion.ml.train_early_warning import (
    DATASET_PATH,
    LOGISTIC_MODEL_PATH,
    FOREST_MODEL_PATH,
)
from fincontagion.ml.predict_saved_model import OUTPUT_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ORIGINAL_PREDICTIONS_PATH = (
    PROCESSED_DIR / "ml_early_warning_predictions.csv"
)

KEYS = ["entity_id", "scenario_id"]

MODEL_COLUMNS = [
    "logistic_regression_probability",
    "logistic_regression_prediction",
    "random_forest_probability",
    "random_forest_prediction",
]


def main():
    print("=" * 65)
    print("FINCONTAGION-X: SAVED MODEL REPRODUCIBILITY CHECK")
    print("=" * 65)

    required_files = [
        DATASET_PATH,
        LOGISTIC_MODEL_PATH,
        FOREST_MODEL_PATH,
        ORIGINAL_PREDICTIONS_PATH,
        OUTPUT_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact is missing: {path}"
            )

    original = pd.read_csv(ORIGINAL_PREDICTIONS_PATH)
    reloaded = pd.read_csv(OUTPUT_PATH)

    print(f"Original held-out rows: {len(original)}")
    print(f"Saved-model inference rows: {len(reloaded)}")

    if original.duplicated(KEYS).any():
        raise AssertionError(
            "Original predictions contain duplicate entity/scenario keys."
        )

    if reloaded.duplicated(KEYS).any():
        raise AssertionError(
            "Saved-model predictions contain duplicate entity/scenario keys."
        )

    required_original = KEYS + ["actual_distress"] + MODEL_COLUMNS
    required_reloaded = KEYS + ["actual_distress"] + MODEL_COLUMNS

    for column in required_original:
        if column not in original.columns:
            raise KeyError(
                f"Column missing from original predictions: {column}"
            )

    for column in required_reloaded:
        if column not in reloaded.columns:
            raise KeyError(
                f"Column missing from saved-model predictions: {column}"
            )

    comparison = original.merge(
        reloaded[required_reloaded],
        on=KEYS,
        how="left",
        suffixes=("_original", "_reloaded"),
        validate="one_to_one",
        indicator=True,
    )

    if not (comparison["_merge"] == "both").all():
        missing = comparison.loc[
            comparison["_merge"] != "both", KEYS
        ]
        raise AssertionError(
            "Some original test rows were not found in saved-model "
            f"inference output:\n{missing.head(10)}"
        )

    if not np.array_equal(
        comparison["actual_distress_original"].to_numpy(),
        comparison["actual_distress_reloaded"].to_numpy(),
    ):
        raise AssertionError(
            "Actual target labels differ for matched rows."
        )

    checks = [
        (
            "Logistic Regression",
            "logistic_regression_probability",
            "logistic_regression_prediction",
        ),
        (
            "Random Forest",
            "random_forest_probability",
            "random_forest_prediction",
        ),
    ]

    for model_name, probability_column, prediction_column in checks:
        print(f"\nChecking {model_name}...")

        probability_match = np.allclose(
            comparison[f"{probability_column}_original"].to_numpy(),
            comparison[f"{probability_column}_reloaded"].to_numpy(),
            rtol=1e-7,
            atol=1e-10,
        )

        prediction_match = np.array_equal(
            comparison[f"{prediction_column}_original"].to_numpy(),
            comparison[f"{prediction_column}_reloaded"].to_numpy(),
        )

        print(f"Probabilities match: {probability_match}")
        print(f"Class predictions match: {prediction_match}")

        if not probability_match or not prediction_match:
            raise AssertionError(
                f"Reproducibility check failed for {model_name}."
            )

    print("\nPASS: saved-model predictions match the original outputs.")
    print(f"Held-out rows verified: {len(comparison)}")
    print(f"Full inference rows available: {len(reloaded)}")
    print("\nBoth saved model artifacts passed the reproducibility check.")


if __name__ == "__main__":
    main()
