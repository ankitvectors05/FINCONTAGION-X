
"""Recall-first threshold selection with entity-disjoint evaluation.

Threshold selection uses only an entity-grouped validation subset.
The final test entities are excluded from all model training and
threshold selection. The held-out scenario is also excluded from
development data.

All results are synthetic-data experiments.
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
    ROOT / "data" / "processed" / "ml_recall_first_validation.csv"
)

TARGET_RECALL = 0.90
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
OUTER_TEST_SIZE = 0.25


def grouped_validation_split(X, y, groups):
    """Create an entity-disjoint inner training/validation split."""

    splitter = GroupShuffleSplit(
        n_splits=200,
        test_size=0.25,
        random_state=42,
    )

    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if y_train.nunique() != 2 or y_val.nunique() != 2:
            continue

        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        if train_groups & val_groups:
            raise RuntimeError("Entity overlap in inner validation split.")

        return (
            X.iloc[train_idx],
            X.iloc[val_idx],
            y_train,
            y_val,
            train_groups,
            val_groups,
        )

    raise ValueError(
        "Could not find an entity-disjoint validation split with both "
        "classes in training and validation."
    )


def select_recall_first_threshold(y_true, probabilities):
    """Choose the highest threshold meeting the recall target.

    If no threshold reaches the target, choose the threshold with the
    highest validation recall; ties favor the higher threshold.
    The unmet target is explicitly reported.
    """

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "threshold": float(threshold),
                "recall": recall_score(
                    y_true, predictions, zero_division=0
                ),
                "precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "f1": f1_score(
                    y_true, predictions, zero_division=0
                ),
                "predicted_positive": int(predictions.sum()),
                "false_positives": int(
                    ((predictions == 1) & (y_true.to_numpy() == 0)).sum()
                ),
            }
        )

    eligible = [
        row for row in rows
        if row["recall"] >= TARGET_RECALL
    ]

    if eligible:
        selected = max(eligible, key=lambda row: row["threshold"])
        target_met = True
    else:
        best_recall = max(row["recall"] for row in rows)
        best_rows = [
            row for row in rows
            if np.isclose(row["recall"], best_recall)
        ]
        selected = max(best_rows, key=lambda row: row["threshold"])
        target_met = False

    return selected, target_met


def main():
    print("=" * 76)
    print("FINCONTAGION-X: RECALL-FIRST THRESHOLD VALIDATION")
    print("=" * 76)
    print(f"Validation recall target: {TARGET_RECALL:.0%}")

    df = pd.read_csv(DATA_PATH)
    X, y, groups, _, preprocessor = prepare_features(df)

    # Reserve test entities across ALL scenarios, so none of their
    # records can enter development or threshold-selection data.
    entity_table = pd.DataFrame(
        {"entity_id": groups.drop_duplicates().to_numpy()}
    )

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=OUTER_TEST_SIZE,
        random_state=42,
    )

    train_entity_idx, test_entity_idx = next(
        outer_splitter.split(
            entity_table,
            groups=entity_table["entity_id"],
        )
    )

    development_entities = set(
        entity_table.iloc[train_entity_idx]["entity_id"]
    )
    test_entities = set(
        entity_table.iloc[test_entity_idx]["entity_id"]
    )

    if development_entities & test_entities:
        raise RuntimeError("Outer entity leakage detected.")

    print(f"Development entities: {len(development_entities)}")
    print(f"Completely held-out test entities: {len(test_entities)}")

    results = []

    for held_out_id in sorted(df["scenario_id"].unique()):
        scenario_mask = df["scenario_id"].eq(held_out_id)

        # Test: held-out scenario AND completely unseen entities.
        test_mask = scenario_mask & groups.isin(test_entities)

        # Development: other scenarios AND development entities only.
        development_mask = (
            ~scenario_mask & groups.isin(development_entities)
        )

        X_development = X.loc[development_mask]
        y_development = y.loc[development_mask]
        groups_development = groups.loc[development_mask]

        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        scenario_name = str(
            df.loc[scenario_mask, "scenario_name"].iloc[0]
        )

        if X_test.empty or X_development.empty:
            print(f"Skipping {scenario_name}: empty partition.")
            continue

        print("\n" + "-" * 76)
        print(f"Held-out scenario: {scenario_name}")
        print(f"Development rows: {len(X_development)}")
        print(f"Test rows: {len(X_test)}")
        print(f"Test distress cases: {int(y_test.sum())}")

        # Confirm neither scenario nor entity leaks into development.
        if groups.loc[test_mask].isin(
            groups_development.unique()
        ).any():
            raise RuntimeError("Outer train/test entity overlap detected.")

        if df.loc[development_mask, "scenario_id"].eq(held_out_id).any():
            raise RuntimeError("Held-out scenario leaked into development.")

        (
            X_inner_train,
            X_val,
            y_inner_train,
            y_val,
            inner_train_entities,
            validation_entities,
        ) = grouped_validation_split(
            X_development,
            y_development,
            groups_development,
        )

        for model_name, base_pipeline in build_models(preprocessor).items():
            # 1. Fit on inner training data only.
            selection_model = clone(base_pipeline)
            selection_model.fit(X_inner_train, y_inner_train)

            # 2. Select threshold using validation data only.
            validation_probabilities = (
                selection_model.predict_proba(X_val)[:, 1]
            )

            selected, target_met = select_recall_first_threshold(
                y_val,
                validation_probabilities,
            )
            threshold = selected["threshold"]

            # 3. Refit on all permitted development data.
            final_model = clone(base_pipeline)
            final_model.fit(X_development, y_development)

            # 4. Evaluate once on unseen entities in the held-out scenario.
            test_probabilities = final_model.predict_proba(X_test)[:, 1]
            test_predictions = (
                test_probabilities >= threshold
            ).astype(int)

            both_test_classes = y_test.nunique() == 2

            result = {
                "held_out_scenario_id": held_out_id,
                "held_out_scenario_name": scenario_name,
                "model": model_name,
                "target_recall": TARGET_RECALL,
                "selected_threshold": threshold,
                "validation_target_met": target_met,
                "validation_recall": selected["recall"],
                "validation_precision": selected["precision"],
                "validation_f1": selected["f1"],
                "validation_false_positives": selected["false_positives"],
                "inner_training_entities": len(inner_train_entities),
                "validation_entities": len(validation_entities),
                "held_out_test_entities": len(test_entities),
                "test_rows": len(X_test),
                "actual_distress_cases": int(y_test.sum()),
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
                "data_type": "synthetic",
                "threshold_selection_basis": (
                    "highest_threshold_meeting_validation_recall_target"
                    if target_met
                    else "best_validation_recall_target_not_met"
                ),
            }
            results.append(result)

            print(f"\n{model_name}")
            print(f"Selected threshold: {threshold:.2f}")
            print(f"Validation recall: {selected['recall']:.4f}")
            print(f"Validation target met: {target_met}")
            print(f"Validation false positives: {selected['false_positives']}")
            print(f"Test precision: {result['test_precision']:.4f}")
            print(f"Test recall: {result['test_recall']:.4f}")
            print(f"Test F1: {result['test_f1']:.4f}")

    if not results:
        raise RuntimeError("No recall-first evaluation results were produced.")

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
        "validation_target_met",
        "validation_recall",
        "validation_false_positives",
        "actual_distress_cases",
        "predicted_distress_cases",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    print(output[columns].to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(
        "\nImportant: the 90% target applies only to validation. "
        "It is not a guarantee of recall on unseen entities or stress "
        "scenarios. All experiments use synthetic labels and data."
    )


if __name__ == "__main__":
    main()
