from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Project paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "data" / "models"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"
FOREST_MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"

MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

DATASET_PATH = DATA_DIR / "ml_scenario_dataset.csv"
COMPARISON_OUTPUT_PATH = DATA_DIR / "ml_model_comparison.csv"
PREDICTIONS_OUTPUT_PATH = DATA_DIR / "ml_early_warning_predictions.csv"
IMPORTANCE_OUTPUT_PATH = DATA_DIR / "ml_feature_importance.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Experimental thresholds for demonstration only.
# These are NOT calibrated real-world risk thresholds.
WARNING_THRESHOLDS = {
    "low": 0.20,
    "moderate": 0.40,
    "high": 0.60,
}


# ---------------------------------------------------------
# 2. Warning-band classification
# ---------------------------------------------------------

def classify_warning_band(probability: float) -> str:
    """Map a model score to an experimental warning band."""

    if probability < WARNING_THRESHOLDS["low"]:
        return "Low"

    if probability < WARNING_THRESHOLDS["moderate"]:
        return "Moderate"

    if probability < WARNING_THRESHOLDS["high"]:
        return "High"

    return "Very High"


# ---------------------------------------------------------
# 3. Load and validate dataset
# ---------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the scenario-labelled dataset and validate its structure."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run the scenario dataset generator first:\n"
            "python -m fincontagion.ml.scenario_dataset"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "entity_id",
        "scenario_id",
        "distress_label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The scenario dataset is empty.")

    if df["entity_id"].isna().any():
        raise ValueError("entity_id contains missing values.")

    if df["distress_label"].isna().any():
        raise ValueError("distress_label contains missing values.")

    labels = set(df["distress_label"].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"distress_label must contain only 0 and 1. Found: {labels}"
        )

    if len(labels) < 2:
        raise ValueError(
            "Training requires both non-distressed (0) and "
            "distressed (1) observations."
        )

    if df["entity_id"].nunique() < 2:
        raise ValueError(
            "At least two unique entities are required for a "
            "group-based train/test split."
        )

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Unique entities: {df['entity_id'].nunique()}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(df["distress_label"].value_counts().sort_index())

    print("\nScenario distribution:")
    print(df["scenario_id"].value_counts().sort_index())

    return df


# ---------------------------------------------------------
# 4. Prepare features and target
# ---------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    """
    Separate predictors, target and entity groups.

    Post-stress outcomes and identifiers are excluded from the
    predictors to reduce direct target leakage.
    """

    target_column = "distress_label"

    group_column = "entity_id"

    excluded_columns = {
        # Target and identifiers
        target_column,
        group_column,
        "scenario_id",
        "scenario_name",

        # Post-stress accounting outcomes
        "credit_loss",
        "market_loss",
        "interest_rate_loss",
        "interest_rate_gain",
        "interest_rate_valuation_change",
        "net_accounting_impact",
        "total_accounting_loss",
        "stressed_equity",

        # Post-stress liquidity outcomes
        "cash_before_stress",
        "funding_withdrawal",
        "cash_after_withdrawal",
        "liquidity_shortfall",

        # Stress-engine diagnostics, not intended as predictors
        "network_lending_exposure",
        "network_lending_exposure_x",
        "network_lending_exposure_y",
        "outgoing_exposure_count",
        "interest_rate_loss_included",
        "interest_rate_shock_configured",
        "interest_rate_assumption_basis",
        "assumption_basis",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No predictor columns remain after exclusions.")

    X = df[feature_columns].copy()
    y = df[target_column].astype(int).copy()
    groups = df[group_column].copy()

    # Remove columns that contain no usable information.
    unusable_columns = [
        column
        for column in X.columns
        if X[column].isna().all()
    ]

    if unusable_columns:
        print(
            "\nDropping entirely missing predictor columns:",
            unusable_columns,
        )

        X = X.drop(columns=unusable_columns)

    if X.shape[1] == 0:
        raise ValueError("No usable predictors remain.")

    # Treat entity type and country as categorical features.
    categorical_columns = [
        column
        for column in X.columns
        if (
            X[column].dtype == "object"
            or isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[column].dtype)
        )
    ]

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # Fail clearly instead of silently coercing unexpected data.
    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="raise")

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

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    print(f"\nPredictor columns: {X.shape[1]}")
    print(f"Numeric predictors: {len(numeric_columns)}")
    print(f"Categorical predictors: {len(categorical_columns)}")

    return X, y, groups, feature_columns, preprocessor


# ---------------------------------------------------------
# 5. Split data by entity
# ---------------------------------------------------------

def split_by_entity(X, y, groups):
    """
    Split by entity rather than by individual scenario row.

    This prevents scenarios belonging to the same entity from
    appearing in both training and test sets.
    """

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    for train_indices, test_indices in splitter.split(
        X,
        y,
        groups=groups,
    ):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        if y_train.nunique() == 2 and y_test.nunique() == 2:
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]

            groups_train = groups.iloc[train_indices]
            groups_test = groups.iloc[test_indices]

            overlap = set(groups_train) & set(groups_test)

            if overlap:
                raise RuntimeError(
                    "Entity leakage detected between training "
                    "and test sets."
                )

            print("\nEntity-grouped split completed.")
            print(f"Training rows: {len(X_train)}")
            print(f"Test rows: {len(X_test)}")
            print(f"Training entities: {groups_train.nunique()}")
            print(f"Test entities: {groups_test.nunique()}")
            print(f"Entity overlap: {len(overlap)}")

            print("\nTraining labels:")
            print(y_train.value_counts().sort_index())

            print("\nTest labels:")
            print(y_test.value_counts().sort_index())

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                groups_train,
                groups_test,
            )
    raise ValueError(
        "Could not find a grouped train/test split containing both "
        "classes in each partition. More entities or a different "
        "dataset may be required."
    )


# ---------------------------------------------------------
# 6. Define models
# ---------------------------------------------------------

def build_models(preprocessor):
    """Create independent preprocessing and classification pipelines."""

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }

# ---------------------------------------------------------
# 6A. Save fitted model pipelines and metadata
# ---------------------------------------------------------

def save_model_artifacts(
    model_outputs,
    feature_columns,
    X_train,
    y_train,
    groups_train,
):
    """Save fitted pipelines and metadata about their training data."""

    import numpy as np

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_paths = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "Random Forest": FOREST_MODEL_PATH,
    }

    dataset_hash = hashlib.sha256(
        DATASET_PATH.read_bytes()
    ).hexdigest()

    saved_models = {}

    for model_name, output in model_outputs.items():
        pipeline = output["pipeline"]
        model_path = model_paths[model_name]

        joblib.dump(pipeline, model_path)

        saved_models[model_name] = {
            "file": model_path.name,
            "classifier": type(
                pipeline.named_steps["classifier"]
            ).__name__,
            "training_rows": int(len(X_train)),
            "training_entities": int(groups_train.nunique()),
            "feature_columns": list(feature_columns),
        }

        print(f"\nSaved {model_name} pipeline:\n{model_path}")

    metadata = {
        "project": "FINCONTAGION-X",
        "artifact_version": "1.0.0",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset_file": DATASET_PATH.name,
        "dataset_sha256": dataset_hash,
        "dataset_type": "synthetic scenario-generated data",
        "target_column": "distress_label",
        "target_definition": (
            "Synthetic scenario distress label; not a real-world "
            "observed future distress event."
        ),
        "training_rows": int(len(X_train)),
        "training_entities": int(groups_train.nunique()),
        "feature_columns": list(feature_columns),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "warning_thresholds": WARNING_THRESHOLDS,
        "warning_thresholds_calibrated": False,
        "real_world_performance_established": False,
        "models": saved_models,
    }

    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print(f"\nModel metadata saved to:\n{MODEL_METADATA_PATH}")

    return metadata

# ---------------------------------------------------------
# 7. Evaluate models
# ---------------------------------------------------------

def evaluate_model(
    model_name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train a pipeline, calculate evaluation metrics and report results."""

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_distress": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall_distress": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_distress": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")

    for metric_name, value in metrics.items():
        if metric_name != "model":
            print(f"{metric_name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Distress", "Distress"],
            zero_division=0,
        )
    )

    return metrics, probabilities, predictions


# ---------------------------------------------------------
# 8. Calculate permutation feature importance
# ---------------------------------------------------------

def calculate_feature_importance(
    pipeline,
    X_test,
    y_test,
    feature_columns,
):
    """
    Measure how much test-set average precision changes when each
    original input feature is shuffled.

    This works with the complete pipeline, so the importance
    results correspond to original input columns rather than
    one-hot-encoded or imputed columns.
    """

    print("\nCalculating permutation feature importance...")

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1),
    )

    importance_df["importance_method"] = (
        "permutation_importance_average_precision"
    )

    IMPORTANCE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        IMPORTANCE_OUTPUT_PATH,
        index=False,
    )

    print("\nTop 10 features by permutation importance:")
    print(importance_df.head(10).to_string(index=False))

    print(f"\nFeature importance saved to:\n{IMPORTANCE_OUTPUT_PATH}")

    return importance_df


# ---------------------------------------------------------
# 9. Build test-set prediction output
# ---------------------------------------------------------

def build_prediction_output(
    X_test,
    y_test,
    groups_test,
    logistic_probabilities,
    logistic_predictions,
    forest_probabilities,
    forest_predictions,
):
    """Create an entity-level test-set prediction report."""

    predictions_df = pd.DataFrame(
        {
            "entity_id": groups_test.to_numpy(),
            "actual_distress": y_test.to_numpy(),
            "logistic_regression_probability": logistic_probabilities,
            "logistic_regression_prediction": logistic_predictions,
            "random_forest_probability": forest_probabilities,
            "random_forest_prediction": forest_predictions,
        },
        index=X_test.index,
    )

    predictions_df["random_forest_warning_band"] = (
        predictions_df["random_forest_probability"].apply(
            classify_warning_band
        )
    )

    predictions_df["warning_threshold_basis"] = (
        "experimental_thresholds_not_real_world_calibrated"
    )

    # Preserve the original dataset's scenario identifiers.
    original_df = pd.read_csv(DATASET_PATH)

    predictions_df["scenario_id"] = (
        original_df.loc[X_test.index, "scenario_id"].to_numpy()
    )

    # Keep the output easy to inspect.
    output_columns = [
        "entity_id",
        "scenario_id",
        "actual_distress",
        "logistic_regression_probability",
        "logistic_regression_prediction",
        "random_forest_probability",
        "random_forest_prediction",
        "random_forest_warning_band",
        "warning_threshold_basis",
    ]

    predictions_df = predictions_df[output_columns]

    PREDICTIONS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    print(f"\nPredictions saved to:\n{PREDICTIONS_OUTPUT_PATH}")

    print("\nRandom Forest warning-band counts:")
    print(
        predictions_df["random_forest_warning_band"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High", "Very High"],
            fill_value=0,
        )
    )

    print("\nSample predictions:")
    print(predictions_df.head(10).to_string(index=False))

    return predictions_df


# ---------------------------------------------------------
# 10. Main execution
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("FINCONTAGION-X: ML EARLY-WARNING MODEL")
    print("=" * 60)

    df = load_dataset()

    (
        X,
        y,
        groups,
        feature_columns,
        preprocessor,
    ) = prepare_features(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    ) = split_by_entity(X, y, groups)

    models = build_models(preprocessor)

    comparison_results = {}
    model_outputs = {}

    for model_name, pipeline in models.items():
        metrics, probabilities, predictions = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        comparison_results[model_name] = metrics

        model_outputs[model_name] = {
            "pipeline": pipeline,
            "probabilities": probabilities,
            "predictions": predictions,
        }

    comparison_df = pd.DataFrame(
        comparison_results.values()
    ).sort_values(
        by="average_precision",
        ascending=False,
    )

    save_model_artifacts(
        model_outputs=model_outputs,
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    print(f"\nModel comparison saved to:\n{COMPARISON_OUTPUT_PATH}")

    forest_output = model_outputs["Random Forest"]

    calculate_feature_importance(
        pipeline=forest_output["pipeline"],
        X_test=X_test,
        y_test=y_test,
        feature_columns=feature_columns,
    )

    build_prediction_output(
        X_test=X_test,
        y_test=y_test,
        groups_test=groups_test,
        logistic_probabilities=model_outputs[
            "Logistic Regression"
        ]["probabilities"],
        logistic_predictions=model_outputs[
            "Logistic Regression"
        ]["predictions"],
        forest_probabilities=forest_output["probabilities"],
        forest_predictions=forest_output["predictions"],
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    print("\nTraining and evaluation completed.")
    print("\nGenerated files:")
    print(f"- {COMPARISON_OUTPUT_PATH}")
    print(f"- {PREDICTIONS_OUTPUT_PATH}")
    print(f"- {IMPORTANCE_OUTPUT_PATH}")
    print(f"- {LOGISTIC_MODEL_PATH}")
    print(f"- {FOREST_MODEL_PATH}")
    print(f"- {MODEL_METADATA_PATH}")

    print(
        "\nIMPORTANT: The dataset and labels are synthetic. "
        "Warning bands are experimental, and the reported metrics "
        "do not establish real-world predictive performance."
    )


if __name__ == "__main__":
    main()