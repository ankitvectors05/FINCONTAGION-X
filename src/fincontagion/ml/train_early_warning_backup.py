
"""Train a leakage-conscious synthetic financial distress classifier.

This is a scenario-based prototype, not a real-world default predictor.
All records for an entity remain in one split.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_scenario_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "distress_label"
GROUP = "entity_id"

# These columns describe outcomes or identifiers, not pre-outcome predictors.
EXCLUDED_COLUMNS = {
    TARGET,
    GROUP,
    "scenario_id",
    "scenario_name",
    "stressed_equity",
    "liquidity_shortfall",
    "total_accounting_loss",
    "interest_rate_shock_configured",
    "interest_rate_loss_included",
    "assumption_basis",
}


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Impute missing values and encode categorical features."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def main() -> None:
    """Load data, split by entity, train models, and report evaluation."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Scenario dataset not found: {DATA_PATH}. "
            "Run scenario_dataset.py first."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = {
        TARGET,
        GROUP,
        "entity_type",
        "country",
        "asset_price_decline",
        "funding_withdrawal_rate",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    if data[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError(
            "Training requires both target classes, 0 and 1."
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X = data[feature_columns].copy()
    y = data[TARGET].astype(int)
    groups = data[GROUP]

    categorical_columns = [
        column
        for column in ["entity_type", "country"]
        if column in X.columns
    ]
    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # Try deterministic group splits until both classes exist in each split.
    # The split is by entity, never by individual scenario row.
    selected_split = None

    for random_state in range(42, 142):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        if (
            y.iloc[train_idx].nunique() == 2
            and y.iloc[test_idx].nunique() == 2
        ):
            selected_split = (train_idx, test_idx, random_state)
            break

    if selected_split is None:
        raise ValueError(
            "Could not find a group-based split containing both classes "
            "in training and testing. More diverse scenarios or entities "
            "are needed before evaluation."
        )

    train_idx, test_idx, random_state = selected_split

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_entities = set(groups.iloc[train_idx])
    test_entities = set(groups.iloc[test_idx])

    if train_entities & test_entities:
        raise RuntimeError("Entity leakage detected between train and test.")

    print("Dataset overview")
    print("----------------")
    print(f"Rows: {len(data)}")
    print(f"Unique entities: {groups.nunique()}")
    print(f"Predictor columns: {len(feature_columns)}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Testing rows: {len(test_idx)}")
    print(f"Training entities: {len(train_entities)}")
    print(f"Testing entities: {len(test_entities)}")
    print(f"Group split random state: {random_state}")
    print(f"Train class counts:\n{y_train.value_counts().sort_index()}")
    print(f"Test class counts:\n{y_test.value_counts().sort_index()}")
    print("\nPredictors:")
    print(", ".join(feature_columns))

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    predictions = pd.DataFrame(
        {
            "entity_id": groups.iloc[test_idx].to_numpy(),
            "scenario_id": data.iloc[test_idx]["scenario_id"].to_numpy(),
            "actual_distress": y_test.to_numpy(),
        }
    )

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(
                        numeric_columns,
                        categorical_columns,
                    ),
                ),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        predicted = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predicted)
        roc_auc = roc_auc_score(y_test, probabilities)
        average_precision = average_precision_score(y_test, probabilities)

        print(f"\n{'=' * 60}")
        print(model_name)
        print(f"{'=' * 60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Average precision: {average_precision:.4f}")
        print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
        print(confusion_matrix(y_test, predicted, labels=[0, 1]))
        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["Non-distressed", "Distressed"],
                zero_division=0,
            )
        )

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
            }
        )
        predictions[f"{model_name.lower().replace(' ', '_')}_probability"] = (
            probabilities
        )
        predictions[f"{model_name.lower().replace(' ', '_')}_prediction"] = (
            predicted
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "ml_model_comparison.csv"
    predictions_path = OUTPUT_DIR / "ml_early_warning_predictions.csv"

    pd.DataFrame(results).to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nSaved evaluation files:")
    print(results_path)
    print(predictions_path)
    print(
        "\nInterpretation: metrics measure performance on held-out entities "
        "under this synthetic scenario design only. They do not establish "
        "real-world predictive performance."
    )


if __name__ == "__main__":
    main()
