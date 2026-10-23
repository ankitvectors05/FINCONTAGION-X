
"""Evaluate FINCONTAGION-X models on independently generated ecosystems.

The models are trained on the seed-42 ecosystem and evaluated, without
refitting, on ecosystems generated with different seeds.

This validates generalization across our synthetic generator only.
It does not establish real-world financial predictive performance.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fincontagion.ml.scenario_dataset import build_scenario_dataset
from fincontagion.ml.train_early_warning import (
    build_models,
    prepare_features,
    split_by_entity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_SEED = 42
TEST_SEEDS = [7, 21, 99, 2026]

OUTPUT_PATH = DATA_DIR / "ml_cross_seed_validation.csv"


def _prepare_seed_dataset(seed: int) -> pd.DataFrame:
    """Generate a dataset without overwriting the main scenario dataset."""

    temporary_output = DATA_DIR / f"ml_scenario_dataset_seed_{seed}.csv"

    feature_output = DATA_DIR / f"ml_features_seed_{seed}.csv"

    dataset = build_scenario_dataset(
        seed=seed,
        output_path=temporary_output,
        feature_output_path=feature_output,
    )

    return dataset


def _evaluate(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> dict:
    """Evaluate a fitted model on an independent ecosystem."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "model": model_name,
        "train_seed": TRAIN_SEED,
        "test_seed": seed,
        "test_rows": len(y_test),
        "test_entities": X_test.shape[0] // len(
            # Four rows per entity in this project, one per scenario.
            [1, 2, 3, 4]
        ),
        "actual_distress_cases": int(y_test.sum()),
        "actual_non_distress_cases": int((y_test == 0).sum()),
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
            if y_test.nunique() == 2
            else float("nan")
        ),
        "average_precision": (
            average_precision_score(y_test, probabilities)
            if y_test.nunique() == 2
            else float("nan")
        ),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
        "test_label_classes": int(y_test.nunique()),
        "validation_basis": "independent_synthetic_seed",
    }


def main() -> pd.DataFrame:
    print("=" * 68)
    print("FINCONTAGION-X: INDEPENDENT-SEED VALIDATION")
    print("=" * 68)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate the training ecosystem.
    training_df = _prepare_seed_dataset(TRAIN_SEED)

    X, y, groups, _, preprocessor = prepare_features(training_df)

    (
        X_train,
        _,
        y_train,
        _,
        _,
        _,
    ) = split_by_entity(X, y, groups)

    # 2. Fit each model once using only the seed-42 training partition.
    # Clone the preprocessor for each pipeline to prevent shared state.
    models = build_models(preprocessor)
    fitted_models = {}

    for model_name, pipeline in models.items():
        pipeline = clone(pipeline)
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        print(f"\nFitted {model_name} on seed {TRAIN_SEED}.")
        print(f"Training rows: {len(X_train)}")

    # 3. Evaluate the frozen pipelines on separate generated ecosystems.
    results = []

    for seed in TEST_SEEDS:
        print("\n" + "-" * 68)
        print(f"Testing independent ecosystem: seed {seed}")
        print("-" * 68)

        test_df = _prepare_seed_dataset(seed)

        X_test, y_test, _, _, _ = prepare_features(test_df)

        # Align columns to the training feature schema.
        # Do not refit preprocessing on the test ecosystem.
        for model_name, model in fitted_models.items():
            expected_columns = list(
                model.named_steps["preprocessor"].feature_names_in_
            )

            missing = set(expected_columns) - set(X_test.columns)
            unexpected = set(X_test.columns) - set(expected_columns)

            if missing or unexpected:
                raise ValueError(
                    f"Feature schema mismatch for seed {seed}. "
                    f"Missing: {sorted(missing)}; "
                    f"unexpected: {sorted(unexpected)}"
                )

            X_test_aligned = X_test[expected_columns]

            result = _evaluate(
                model_name=model_name,
                model=model,
                X_test=X_test_aligned,
                y_test=y_test,
                seed=seed,
            )

            result["test_entities"] = test_df["entity_id"].nunique()
            results.append(result)

            print(f"\n{model_name}")
            for metric in [
                "accuracy",
                "precision_distress",
                "recall_distress",
                "f1_distress",
                "roc_auc",
                "average_precision",
            ]:
                print(f"{metric}: {result[metric]:.4f}")

            print(
                "Confusion matrix [[TN, FP], [FN, TP]]: "
                f"[[{result['true_negatives']}, "
                f"{result['false_positives']}], "
                f"[{result['false_negatives']}, "
                f"{result['true_positives']}]]"
            )

    # 4. Save independent-seed metrics separately.
    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 68)
    print("CROSS-SEED VALIDATION SUMMARY")
    print("=" * 68)

    print(
        results_df[
            [
                "model",
                "test_seed",
                "accuracy",
                "precision_distress",
                "recall_distress",
                "f1_distress",
                "roc_auc",
                "average_precision",
            ]
        ].to_string(index=False)
    )

    print("\nMean metrics across test seeds:")
    print(
        results_df.groupby("model")[
            [
                "accuracy",
                "precision_distress",
                "recall_distress",
                "f1_distress",
                "roc_auc",
                "average_precision",
            ]
        ].mean().to_string()
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(
        "\nNOTE: These results assess generalization across synthetic "
        "ecosystems, not real-world predictive performance."
    )

    return results_df


if __name__ == "__main__":
    main()
