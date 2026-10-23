
"""Leave-one-scenario-out validation for FINCONTAGION-X.

Tests extrapolation to a held-out stress severity on entities that
also appear in the training scenarios. This is not an independent-
entity validation or evidence of real-world predictive performance.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_scenario_dataset.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_unseen_scenario_validation.csv"
)


def main():
    print("=" * 72)
    print("FINCONTAGION-X: UNSEEN SCENARIO VALIDATION")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"scenario_id", "scenario_name", "entity_id", "distress_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["scenario_id"].isna().any():
        raise ValueError("scenario_id contains missing values.")

    scenario_ids = sorted(df["scenario_id"].unique())
    if len(scenario_ids) < 2:
        raise ValueError("At least two scenarios are required.")

    X, y, groups, _, preprocessor = prepare_features(df)
    results = []

    print("\nEach experiment holds out one complete scenario.")
    print("Training uses the remaining scenarios.")
    print("Existing saved model artifacts will not be modified.")

    for held_out in scenario_ids:
        test_mask = df["scenario_id"].eq(held_out)
        train_mask = ~test_mask

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        scenario_name = str(
            df.loc[test_mask, "scenario_name"].iloc[0]
        )

        train_scenarios = sorted(
            df.loc[train_mask, "scenario_name"].unique().tolist()
        )

        overlap_entities = (
            set(groups.loc[train_mask])
            & set(groups.loc[test_mask])
        )

        print("\n" + "-" * 72)
        print(f"Held-out scenario: {scenario_name} ({held_out})")
        print(f"Training scenarios: {train_scenarios}")
        print(f"Training rows: {len(X_train)}")
        print(f"Test rows: {len(X_test)}")
        print(f"Training distress cases: {int(y_train.sum())}")
        print(f"Test distress cases: {int(y_test.sum())}")
        print(f"Shared entities across partitions: {len(overlap_entities)}")

        if y_train.nunique() < 2:
            print("Skipping: training data contains only one target class.")
            continue

        for model_name, base_pipeline in build_models(preprocessor).items():
            pipeline = clone(base_pipeline)
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            both_classes = y_test.nunique() == 2

            result = {
                "held_out_scenario_id": held_out,
                "held_out_scenario_name": scenario_name,
                "model": model_name,
                "training_scenarios": "|".join(train_scenarios),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "training_entities": groups.loc[train_mask].nunique(),
                "test_entities": groups.loc[test_mask].nunique(),
                "shared_entities": len(overlap_entities),
                "training_distress_cases": int(y_train.sum()),
                "test_distress_cases": int(y_test.sum()),
                "accuracy": accuracy_score(y_test, predictions),
                "precision_distress": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall_distress": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_distress": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": (
                    roc_auc_score(y_test, probabilities)
                    if both_classes
                    else np.nan
                ),
                "average_precision": (
                    average_precision_score(y_test, probabilities)
                    if both_classes
                    else np.nan
                ),
                "evaluation_note": (
                    "Both target classes present; interpret with caution."
                    if both_classes
                    else (
                        "Only one target class in test scenario; "
                        "ROC-AUC and average precision are undefined."
                    )
                ),
            }
            results.append(result)

            print(f"\n{model_name}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Distress precision: {result['precision_distress']:.4f}")
            print(f"Distress recall: {result['recall_distress']:.4f}")
            print(f"Distress F1: {result['f1_distress']:.4f}")

            if both_classes:
                print(f"ROC-AUC: {result['roc_auc']:.4f}")
                print(
                    f"Average precision: "
                    f"{result['average_precision']:.4f}"
                )
            else:
                print(
                    "ROC-AUC / average precision: undefined "
                    "(only one test class)"
                )

    if not results:
        raise RuntimeError("No validation results were generated.")

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 72)
    print("UNSEEN SCENARIO VALIDATION SUMMARY")
    print("=" * 72)

    summary_columns = [
        "held_out_scenario_name",
        "model",
        "test_distress_cases",
        "accuracy",
        "precision_distress",
        "recall_distress",
        "f1_distress",
        "roc_auc",
        "average_precision",
    ]
    print(results_df[summary_columns].to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")

    print(
        "\nLIMITATIONS:\n"
        "- Entities are shared between training and held-out scenarios.\n"
        "- Mild has no positive labels; moderate has only one.\n"
        "- Stress levels are synthetic and discrete.\n"
        "- This experiment does not establish real-world performance."
    )


if __name__ == "__main__":
    main()
