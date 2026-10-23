
"""Diagnostic threshold sensitivity for unseen stress scenarios.

Threshold results on held-out scenarios are for analysis only.
Do not select a production threshold from these test results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fincontagion.ml.train_early_warning import (
    build_models,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "processed" / "ml_scenario_dataset.csv"
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "ml_threshold_sensitivity.csv"
)
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]


def main():
    print("=" * 72)
    print("FINCONTAGION-X: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    X, y, _, _, preprocessor = prepare_features(df)
    results = []

    for scenario_id in sorted(df["scenario_id"].unique()):
        test_mask = df["scenario_id"].eq(scenario_id)
        train_mask = ~test_mask

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]

        scenario_name = str(
            df.loc[test_mask, "scenario_name"].iloc[0]
        )

        if y_train.nunique() < 2:
            print(f"Skipping {scenario_name}: training has one class.")
            continue

        print(f"\nHeld-out scenario: {scenario_name}")
        print(f"Actual distress cases: {int(y_test.sum())}")

        for model_name, base_pipeline in build_models(preprocessor).items():
            model = clone(base_pipeline)
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]

            for threshold in THRESHOLDS:
                predictions = (probabilities >= threshold).astype(int)

                result = {
                    "scenario_id": scenario_id,
                    "scenario_name": scenario_name,
                    "model": model_name,
                    "threshold": threshold,
                    "test_rows": len(y_test),
                    "actual_distress_cases": int(y_test.sum()),
                    "predicted_distress_cases": int(predictions.sum()),
                    "true_positives": int(
                        ((predictions == 1) & (y_test.to_numpy() == 1)).sum()
                    ),
                    "false_positives": int(
                        ((predictions == 1) & (y_test.to_numpy() == 0)).sum()
                    ),
                    "false_negatives": int(
                        ((predictions == 0) & (y_test.to_numpy() == 1)).sum()
                    ),
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(
                        y_test, predictions, zero_division=0
                    ),
                    "recall": recall_score(
                        y_test, predictions, zero_division=0
                    ),
                    "f1": f1_score(
                        y_test, predictions, zero_division=0
                    ),
                    "roc_auc": (
                        roc_auc_score(y_test, probabilities)
                        if y_test.nunique() == 2
                        else np.nan
                    ),
                    "average_precision": (
                        average_precision_score(y_test, probabilities)
                        if y_test.nunique() == 2
                        else np.nan
                    ),
                    "threshold_selection_basis": (
                        "diagnostic_only_not_selected_for_deployment"
                    ),
                }
                results.append(result)

    if not results:
        raise RuntimeError("No threshold results were generated.")

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    columns = [
        "scenario_name",
        "model",
        "threshold",
        "predicted_distress_cases",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    ]

    print("\n" + "=" * 72)
    print("THRESHOLD COMPARISON")
    print("=" * 72)
    print(output[columns].to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(
        "\nWarning: test-set threshold comparisons are diagnostic. "
        "Choose operational thresholds using training-only validation "
        "and a clearly defined cost of missed distress versus false alarms."
    )


if __name__ == "__main__":
    main()
