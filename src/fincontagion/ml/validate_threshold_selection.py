
"""Training-only threshold selection for held-out stress scenarios.

Threshold selection uses an entity-grouped validation split drawn only
from the non-held-out scenarios. The held-out scenario is used only
for final evaluation.

This remains a synthetic-data experiment, not real-world validation.
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
from sklearn.model_selection import GroupShuffleSplit

from fincontagion.ml.train_early_warning import (
    build_models,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "processed" / "ml_scenario_dataset.csv"
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "ml_threshold_selection_validation.csv"
)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def grouped_validation_split(X, y, groups):
    """Return an entity-disjoint inner training/validation split."""

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.25,
        random_state=42,
    )

    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if y_train.nunique() == 2 and y_val.nunique() == 2:
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            groups_train = groups.iloc[train_idx]
            groups_val = groups.iloc[val_idx]

            if set(groups_train) & set(groups_val):
                raise RuntimeError("Entity overlap in inner validation split.")

            return X_train, X_val, y_train, y_val

    raise ValueError(
        "Could not create an entity-grouped inner split containing both "
        "classes in training and validation. More suitable data is needed."
    )


def choose_threshold(y_true, probabilities):
    """Choose the threshold with highest validation F1.

    Ties are resolved in favor of the higher threshold to avoid selecting
    a more permissive threshold when F1 is identical.
    """

    candidates = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        candidates.append((score, threshold))

    best_score = max(score for score, _ in candidates)
    best_threshold = max(
        threshold
        for score, threshold in candidates
        if np.isclose(score, best_score)
    )

    return float(best_threshold), float(best_score)


def main():
    print("=" * 76)
    print("FINCONTAGION-X: TRAINING-ONLY THRESHOLD SELECTION")
    print("=" * 76)

    df = pd.read_csv(DATA_PATH)
    X, y, groups, _, preprocessor = prepare_features(df)

    results = []

    for held_out in sorted(df["scenario_id"].unique()):
        test_mask = df["scenario_id"].eq(held_out)
        train_mask = ~test_mask

        X_development = X.loc[train_mask]
        y_development = y.loc[train_mask]
        groups_development = groups.loc[train_mask]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        scenario_name = str(
            df.loc[test_mask, "scenario_name"].iloc[0]
        )

        print("\n" + "-" * 76)
        print(f"Held-out scenario: {scenario_name}")

        X_inner_train, X_val, y_inner_train, y_val = grouped_validation_split(
            X_development,
            y_development,
            groups_development,
        )

        inner_train_entities = set(
            groups_development.loc[X_inner_train.index]
        )
        validation_entities = set(
            groups_development.loc[X_val.index]
        )

        if inner_train_entities & validation_entities:
            raise RuntimeError("Entity leakage detected in threshold selection.")

        for model_name, base_pipeline in build_models(preprocessor).items():
            # Select threshold using only the inner training/validation data.
            selection_model = clone(base_pipeline)
            selection_model.fit(X_inner_train, y_inner_train)

            validation_probabilities = selection_model.predict_proba(X_val)[:, 1]
            threshold, validation_f1 = choose_threshold(
                y_val,
                validation_probabilities,
            )

            # Refit on all non-held-out scenarios.
            final_model = clone(base_pipeline)
            final_model.fit(X_development, y_development)

            test_probabilities = final_model.predict_proba(X_test)[:, 1]
            test_predictions = (test_probabilities >= threshold).astype(int)

            both_test_classes = y_test.nunique() == 2

            result = {
                "held_out_scenario_id": held_out,
                "held_out_scenario_name": scenario_name,
                "model": model_name,
                "selected_threshold": threshold,
                "validation_f1_at_selected_threshold": validation_f1,
                "validation_rows": len(X_val),
                "validation_entities": len(validation_entities),
                "inner_train_entities": len(inner_train_entities),
                "held_out_test_rows": len(X_test),
                "held_out_distress_cases": int(y_test.sum()),
                "predicted_distress_cases": int(test_predictions.sum()),
                "true_positives": int(
                    ((test_predictions == 1) & (y_test.to_numpy() == 1)).sum()
                ),
                "false_positives": int(
                    ((test_predictions == 1) & (y_test.to_numpy() == 0)).sum()
                ),
                "false_negatives": int(
                    ((test_predictions == 0) & (y_test.to_numpy() == 1)).sum()
                ),
                "test_accuracy": accuracy_score(y_test, test_predictions),
                "test_precision": precision_score(
                    y_test, test_predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, test_predictions, zero_division=0
                ),
                "test_f1": f1_score(
                    y_test, test_predictions, zero_division=0
                ),
                "test_roc_auc": (
                    roc_auc_score(y_test, test_probabilities)
                    if both_test_classes
                    else np.nan
                ),
                "test_average_precision": (
                    average_precision_score(y_test, test_probabilities)
                    if both_test_classes
                    else np.nan
                ),
                "threshold_basis": (
                    "inner_entity_grouped_validation_max_f1"
                ),
                "data_type": "synthetic",
            }
            results.append(result)

            print(f"\n{model_name}")
            print(f"Selected threshold: {threshold:.2f}")
            print(f"Inner validation F1: {validation_f1:.4f}")
            print(f"Predicted distress cases: {test_predictions.sum()}")
            print(f"Actual distress cases: {int(y_test.sum())}")
            print(f"Test precision: {result['test_precision']:.4f}")
            print(f"Test recall: {result['test_recall']:.4f}")
            print(f"Test F1: {result['test_f1']:.4f}")

    output = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)

    columns = [
        "held_out_scenario_name",
        "model",
        "selected_threshold",
        "validation_f1_at_selected_threshold",
        "held_out_distress_cases",
        "predicted_distress_cases",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_average_precision",
    ]
    print(output[columns].to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(
        "\nInterpretation: thresholds are selected without using the held-out "
        "scenario labels. However, synthetic labels and only four stress "
        "levels still limit the strength of any generalization claim."
    )


if __name__ == "__main__":
    main()
