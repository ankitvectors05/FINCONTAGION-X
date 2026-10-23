"""Feature engineering for FINCONTAGION-X.

Builds one validated ML feature row per financial entity from entity,
financial-position, and exposure datasets.

Concentration is undefined when an entity has no recorded counterparties.
Separate relationship indicators preserve that distinction for downstream
ML preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_DIR / "ml_features.csv"


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    filename: str,
) -> None:
    """Raise a clear error when a required column is missing."""
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {sorted(missing)}"
        )


def _validate_unique_ids(
    dataframe: pd.DataFrame,
    column: str,
    filename: str,
) -> None:
    """Ensure identifiers exist and are unique."""
    if dataframe[column].isna().any():
        raise ValueError(
            f"{filename} contains missing values in {column!r}."
        )

    if dataframe[column].duplicated().any():
        duplicates = (
            dataframe.loc[dataframe[column].duplicated(), column]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"{filename} contains duplicate {column} values: {duplicates}"
        )


def _numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    filename: str,
) -> pd.DataFrame:
    """Convert required numeric columns and reject invalid values."""
    result = dataframe.copy()

    for column in columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if result[column].isna().any():
            raise ValueError(
                f"{filename} contains missing or non-numeric values "
                f"in {column!r}."
            )

        values = result[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{filename} contains infinite values in {column!r}."
            )

        if (result[column] < 0).any():
            raise ValueError(
                f"{filename} contains negative values in {column!r}."
            )

    return result


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Calculate a ratio; undefined zero-denominator cases remain NaN."""
    return numerator.div(denominator.where(denominator != 0))


def _validate_concentration(
    features: pd.DataFrame,
    count_column: str,
    concentration_column: str,
) -> None:
    """Validate concentration semantics against counterparty counts."""
    counts = features[count_column]
    concentration = features[concentration_column]

    if (counts % 1 != 0).any():
        raise ValueError(
            f"{count_column} must contain whole-number counterparty counts."
        )

    no_counterparties = counts == 0
    has_counterparties = counts > 0

    if concentration.loc[no_counterparties].notna().any():
        raise ValueError(
            f"{concentration_column} must be NaN when "
            f"{count_column} is zero."
        )

    if concentration.loc[has_counterparties].isna().any():
        raise ValueError(
            f"{concentration_column} is missing despite recorded "
            f"counterparties in {count_column}."
        )

    defined_values = concentration.loc[has_counterparties]

    if (
        (defined_values < -1e-12)
        | (defined_values > 1 + 1e-12)
    ).any():
        raise ValueError(
            f"{concentration_column} must lie between 0 and 1."
        )


def build_feature_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
    *,
    entities: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    exposures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a validated entity-level feature dataset.

    Two modes are supported:

    1. CSV mode:
       Read entities.csv, financial_positions.csv and exposures.csv
       from data_dir.

    2. In-memory mode:
       Supply entities, positions and exposures as dataframes. This
       ensures feature rows are calculated from the exact ecosystem
       used to generate scenario outcomes.

    If in-memory inputs are supplied, all three must be supplied.
    Set output_path=None to avoid writing a CSV.
    """

    data_dir = Path(data_dir)

    supplied = [
        entities is not None,
        positions is not None,
        exposures is not None,
    ]

    if any(supplied) and not all(supplied):
        raise ValueError(
            "In-memory mode requires entities, positions and exposures "
            "to be supplied together."
        )

    if not any(supplied):
        entities_path = data_dir / "entities.csv"
        positions_path = data_dir / "financial_positions.csv"
        exposures_path = data_dir / "exposures.csv"

        for path in (
            entities_path,
            positions_path,
            exposures_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required dataset not found: {path}"
                )

        entities = pd.read_csv(entities_path)
        positions = pd.read_csv(positions_path)
        exposures = pd.read_csv(exposures_path)
    else:
        entities = entities.copy()
        positions = positions.copy()
        exposures = exposures.copy()

    # 1. Validate required columns.
    _require_columns(
        entities,
        {"entity_id", "entity_type", "country"},
        "entities",
    )

    _require_columns(
        positions,
        {
            "entity_id",
            "cash_and_liquid_assets",
            "total_assets",
            "total_liabilities",
            "equity_capital",
            "short_term_liabilities",
        },
        "financial positions",
    )

    _require_columns(
        exposures,
        {
            "exposure_id",
            "creditor_id",
            "debtor_id",
            "outstanding_amount",
        },
        "exposures",
    )

    # 2. Validate identifiers.
    _validate_unique_ids(entities, "entity_id", "entities")
    _validate_unique_ids(positions, "entity_id", "financial positions")
    _validate_unique_ids(exposures, "exposure_id", "exposures")

    if entities["entity_type"].isna().any():
        raise ValueError("Entities contain missing entity types.")

    if entities["country"].isna().any():
        raise ValueError("Entities contain missing countries.")

    # 3. Ensure each entity has exactly one financial position.
    entity_ids = set(entities["entity_id"])
    position_ids = set(positions["entity_id"])

    missing_positions = entity_ids - position_ids
    unknown_positions = position_ids - entity_ids

    if missing_positions or unknown_positions:
        raise ValueError(
            "Entity/position coverage mismatch. "
            f"Entities without positions: {len(missing_positions)}; "
            f"positions with unknown entities: {len(unknown_positions)}."
        )

    # 4. Validate financial amounts and accounting identity.
    position_amount_columns = [
        "cash_and_liquid_assets",
        "total_assets",
        "total_liabilities",
        "equity_capital",
        "short_term_liabilities",
    ]

    positions = _numeric_columns(
        positions,
        position_amount_columns,
        "financial positions",
    )

    tolerance = 0.01

    if (
        positions["cash_and_liquid_assets"]
        > positions["total_assets"] + tolerance
    ).any():
        raise ValueError("Liquid assets exceed total assets.")

    if (
        positions["short_term_liabilities"]
        > positions["total_liabilities"] + tolerance
    ).any():
        raise ValueError("Short-term liabilities exceed total liabilities.")

    balance_difference = (
        positions["total_assets"]
        - positions["total_liabilities"]
        - positions["equity_capital"]
    )

    if not np.isclose(
        balance_difference.to_numpy(dtype=float),
        0.0,
        atol=tolerance,
        rtol=0.0,
    ).all():
        raise ValueError(
            "Accounting identity failed: assets must equal "
            "liabilities plus equity."
        )

    # 5. Validate network exposures.
    exposures = _numeric_columns(
        exposures,
        ["outstanding_amount"],
        "exposures",
    )

    if exposures[["creditor_id", "debtor_id"]].isna().any().any():
        raise ValueError("Exposure data contains missing entity references.")

    if (
        exposures["creditor_id"] == exposures["debtor_id"]
    ).any():
        raise ValueError(
            "An entity cannot be its own exposure counterparty."
        )

    exposure_entity_ids = (
        set(exposures["creditor_id"])
        | set(exposures["debtor_id"])
    )

    unknown_exposure_entities = exposure_entity_ids - entity_ids

    if unknown_exposure_entities:
        raise ValueError(
            "Exposures reference entities absent from entities: "
            f"{sorted(unknown_exposure_entities)[:5]}"
        )

    # 6. Aggregate contracts by creditor-debtor pair.
    pair_exposures = (
        exposures.groupby(
            ["creditor_id", "debtor_id"],
            as_index=False,
        )["outstanding_amount"]
        .sum()
    )

    lending = (
        pair_exposures.groupby("creditor_id")["outstanding_amount"]
        .agg(["sum", "count", "max"])
        .rename(
            columns={
                "sum": "network_lending_exposure",
                "count": "lending_counterparty_count",
                "max": "largest_lending_counterparty_exposure",
            }
        )
    )

    borrowing = (
        pair_exposures.groupby("debtor_id")["outstanding_amount"]
        .agg(["sum", "count", "max"])
        .rename(
            columns={
                "sum": "network_borrowing_exposure",
                "count": "borrowing_counterparty_count",
                "max": "largest_borrowing_counterparty_exposure",
            }
        )
    )

    # 7. Merge entity metadata and financial positions.
    features = entities[
        ["entity_id", "entity_type", "country"]
    ].merge(
        positions[
            [
                "entity_id",
                "cash_and_liquid_assets",
                "total_assets",
                "total_liabilities",
                "equity_capital",
                "short_term_liabilities",
            ]
        ],
        on="entity_id",
        how="inner",
        validate="one_to_one",
    )

    # 8. Financial ratios.
    features["equity_to_assets"] = _safe_ratio(
        features["equity_capital"],
        features["total_assets"],
    )

    features["liabilities_to_assets"] = _safe_ratio(
        features["total_liabilities"],
        features["total_assets"],
    )

    features["liquid_assets_to_short_term_liabilities"] = _safe_ratio(
        features["cash_and_liquid_assets"],
        features["short_term_liabilities"],
    )

    features["cash_to_assets"] = _safe_ratio(
        features["cash_and_liquid_assets"],
        features["total_assets"],
    )

    # 9. Network exposure features.
    features = features.merge(
        lending,
        left_on="entity_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    features = features.merge(
        borrowing,
        left_on="entity_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    network_columns = [
        "network_lending_exposure",
        "lending_counterparty_count",
        "largest_lending_counterparty_exposure",
        "network_borrowing_exposure",
        "borrowing_counterparty_count",
        "largest_borrowing_counterparty_exposure",
    ]

    # Zero means no exposure recorded in this dataset.
    features[network_columns] = features[network_columns].fillna(0)

    # 10. Concentration and relationship indicators.
    features["lending_concentration"] = _safe_ratio(
        features["largest_lending_counterparty_exposure"],
        features["network_lending_exposure"],
    )

    features["borrowing_concentration"] = _safe_ratio(
        features["largest_borrowing_counterparty_exposure"],
        features["network_borrowing_exposure"],
    )

    features["has_lending_relationships"] = (
        features["lending_counterparty_count"] > 0
    ).astype("int8")

    features["has_borrowing_relationships"] = (
        features["borrowing_counterparty_count"] > 0
    ).astype("int8")

    _validate_concentration(
        features,
        "lending_counterparty_count",
        "lending_concentration",
    )

    _validate_concentration(
        features,
        "borrowing_counterparty_count",
        "borrowing_concentration",
    )

    # 11. Final structural and numerical checks.
    if len(features) != len(entities):
        raise ValueError(
            "Feature dataset row count does not match entities."
        )

    _validate_unique_ids(features, "entity_id", "ML feature dataset")

    numeric_feature_columns = features.select_dtypes(
        include=["number"]
    ).columns

    numeric_values = features[numeric_feature_columns].to_numpy(
        dtype=float
    )

    # Undefined ratios may remain NaN; infinities are never valid.
    if np.isinf(numeric_values).any():
        raise ValueError(
            "Feature dataset contains infinite numeric values."
        )

    # Only write output when a path is explicitly available.
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)

        print(f"ML features saved to: {output_path}")

    print("ML feature engineering completed successfully.")
    print(f"Entities represented: {len(features)}")
    print(f"Feature columns: {len(features.columns)}")

    return features


if __name__ == "__main__":
    build_feature_dataset()