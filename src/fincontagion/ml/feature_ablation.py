
"""Feature ablation for FINCONTAGION-X.

Compares the full predictor set with a reduced set that excludes
asset_price_decline and funding_withdrawal_rate.

This diagnostic experiment uses synthetic data and does not modify
the existing saved model artifacts.
"""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fincontagion.ml.train_early_warning import (
    build_models,
    prepare_features,
    split_by_entity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_scenario_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ml_feature_ablation.csv"

SCENARIO_FEATURES = [
    "asset_price_decline",
    "funding_withdrawal_rate",
]


def build_subset_preprocessor(X_train):
    """Rebuild preprocessing using only the selected input columns."""

    categorical_columns = [
        column
        for column in X_train.columns
        if (
            X_train[column].dtype == "object"
            or isinstance(X_train[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X_train[column].dtype)
        )
    ]

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def evaluate_model(model_name, feature_set, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "feature_set": feature_set,
        "predictor_count": X_test.shape[1],
        "test_rows": len(y_test),
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
            if y_test.nunique() == 2
            else float("nan")
        ),
        "average_precision": (
            average_precision_score(y_test, probabilities)
            if y_test.nunique() == 2
            else float("nan")
        ),
        "evaluation_basis": "single_entity_grouped_split_synthetic_data",
    }


def main():
    print("=" * 68)
    print("FINCONTAGION-X: FEATURE-ABLATION EXPERIMENT")
    print("=" * 68)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X, y, groups, _, _ = prepare_features(df)

    missing = set(SCENARIO_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Expected scenario features are missing: {missing}")

    X_train, X_test, y_train, y_test, _, _ = split_by_entity(X, y, groups)

    feature_sets = {
        "full": list(X.columns),
        "without_scenario_severity": [
            column
            for column in X.columns
            if column not in SCENARIO_FEATURES
        ],
    }

    print(f"\nScenario features removed in reduced model: {SCENARIO_FEATURES}")

    results = []

    for feature_set, columns in feature_sets.items():
        X_train_subset = X_train[columns]
        X_test_subset = X_test[columns]

        preprocessor = build_subset_preprocessor(X_train_subset)

        for model_name, base_pipeline in build_models(preprocessor).items():
            pipeline = clone(base_pipeline)
            pipeline.fit(X_train_subset, y_train)

            result = evaluate_model(
                model_name=model_name,
                feature_set=feature_set,
                model=pipeline,
                X_test=X_test_subset,
                y_test=y_test,
            )
            results.append(result)

            print(f"\n{model_name} | {feature_set}")
            print(f"Predictors: {len(columns)}")
            for metric in [
                "accuracy",
                "precision_distress",
                "recall_distress",
                "f1_distress",
                "roc_auc",
                "average_precision",
            ]:
                print(f"{metric}: {result[metric]:.4f}")

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 68)
    print("ABLATION SUMMARY")
    print("=" * 68)
    print(
        results_df[
            [
                "model",
                "feature_set",
                "predictor_count",
                "accuracy",
                "precision_distress",
                "recall_distress",
                "f1_distress",
                "roc_auc",
                "average_precision",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(
        "\nCaution: this is a single entity-grouped split on synthetic "
        "data. Metrics do not establish real-world predictive accuracy."
    )


if __name__ == "__main__":
    main()
