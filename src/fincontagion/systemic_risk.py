
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROCESSED_DIR = Path("data/processed")

# Illustrative weights; these are not statistically calibrated.
NETWORK_WEIGHT = 0.40
FINANCIAL_STRESS_WEIGHT = 0.40
CONTAGION_WEIGHT = 0.20


def _to_dataframe(data: Any, name: str) -> pd.DataFrame:
    """Convert supported input data into a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, list):
        return pd.DataFrame(data)

    raise TypeError(
        f"{name} must be a pandas DataFrame or a list of dictionaries; "
        f"received {type(data).__name__}."
    )


def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first candidate column present in a DataFrame."""
    return next(
        (column for column in candidates if column in dataframe.columns),
        None,
    )


def _numeric_series(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Read the first available numeric column, filling missing values."""
    column = _first_existing_column(dataframe, candidates)

    if column is None:
        return pd.Series(0.0, index=dataframe.index, dtype=float)

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).fillna(0.0)


def _normalise_score(series: pd.Series) -> pd.Series:
    """Min-max normalise non-negative scores to the range 0–100."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    values = values.clip(lower=0.0)

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(0.0, index=series.index, dtype=float)

    return ((values - minimum) / (maximum - minimum)) * 100.0


def calculate_systemic_risk_assessment(
    systemic_importance: pd.DataFrame | list[dict],
    integrated_stress: pd.DataFrame | list[dict],
    contagion_results: pd.DataFrame | list[dict],
    reconciliation: pd.DataFrame | list[dict],
) -> pd.DataFrame:
    """
    Calculate an illustrative entity-level monitoring-priority score.

    The result is not a calibrated probability of default or a regulatory
    systemic-risk measure. Input columns are detected from supported
    alternatives to accommodate the existing project outputs.
    """
    importance_df = _to_dataframe(
        systemic_importance,
        "systemic_importance",
    )
    stress_df = _to_dataframe(
        integrated_stress,
        "integrated_stress",
    )
    contagion_df = _to_dataframe(
        contagion_results,
        "contagion_results",
    )
    reconciliation_df = _to_dataframe(
        reconciliation,
        "reconciliation",
    )

    if "entity_id" not in importance_df.columns:
        raise ValueError(
            "Systemic-importance input must contain an 'entity_id' column."
        )

    # Use the network-importance dataset as the entity universe.
    result = importance_df.copy()

    # Network component: exposure and connectivity measures.
    network_exposure = _numeric_series(
        result,
        [
            "total_connected_exposure",
            "total_exposure",
            "gross_exposure",
        ],
    )
    counterparties = _numeric_series(
        result,
        [
            "unique_counterparties",
            "counterparty_count",
            "number_of_counterparties",
        ],
    )
    concentration = _numeric_series(
        result,
        [
            "largest_counterparty_concentration",
            "max_counterparty_concentration",
        ],
    )

    network_raw = (
        _normalise_score(network_exposure) * 0.60
        + _normalise_score(counterparties) * 0.25
        + _normalise_score(concentration) * 0.15
    )

    # Integrated stress component. Prefer absolute loss/shortfall measures.
    stress_columns = {
        "credit_loss": [
            "credit_loss",
            "expected_credit_loss",
            "total_expected_credit_loss",
        ],
        "market_loss": [
            "market_loss",
            "proxy_market_losses",
            "market_losses",
        ],
        "funding_withdrawal": [
            "funding_withdrawal",
            "funding_withdrawals",
        ],
        "liquidity_shortfall": [
            "liquidity_shortfall",
        ],
    }

    stress_available = any(
        _first_existing_column(stress_df, candidates) is not None
        for candidates in stress_columns.values()
    )

    if "scenario_id" in stress_df.columns:
        scenario_ids = stress_df["scenario_id"].dropna().unique()
        if len(scenario_ids) > 1:
            raise ValueError(
                "Systemic-risk scoring requires stress results for exactly "
                "one scenario. Filter stress_df by scenario_id before "
                "calling calculate_systemic_risk_assessment()."
            )

    if "entity_id" in stress_df.columns:
        stress_entity_column = "entity_id"
    else:
        stress_entity_column = None

    if stress_entity_column is not None:
        stress_work = stress_df.copy()

        for output_column, candidates in stress_columns.items():
            stress_work[output_column] = _numeric_series(
                stress_work,
                candidates,
            )

        stress_aggregated = (
            stress_work.groupby("entity_id", as_index=False)[
                list(stress_columns)
            ].sum()
        )

        result = result.merge(
            stress_aggregated,
            on="entity_id",
            how="left",
            suffixes=("", "_stress"),
        )

        for column in stress_columns:
            stress_name = (
                f"{column}_stress"
                if f"{column}_stress" in result.columns
                else column
            )

            if stress_name in result.columns:
                result[column] = pd.to_numeric(
                    result[stress_name],
                    errors="coerce",
                ).fillna(0.0)
            else:
                result[column] = 0.0

    else:
        for column in stress_columns:
            if column not in result.columns:
                result[column] = 0.0

    financial_raw = (
        _normalise_score(result["credit_loss"]) * 0.40
        + _normalise_score(result["market_loss"]) * 0.25
        + _normalise_score(result["funding_withdrawal"]) * 0.20
        + _normalise_score(result["liquidity_shortfall"]) * 0.15
    )

    if not stress_available:
        financial_raw = pd.Series(
            0.0,
            index=result.index,
            dtype=float,
        )

    # Contagion component: aggregate losses and default evidence by entity.
    if not contagion_df.empty and "entity_id" in contagion_df.columns:
        contagion_work = contagion_df.copy()

        loss_column = _first_existing_column(
            contagion_work,
            [
                "round_loss",
                "contagion_loss",
                "cumulative_contagion_loss",
            ],
        )

        if loss_column is None:
            contagion_work["_loss"] = 0.0
        else:
            contagion_work["_loss"] = pd.to_numeric(
                contagion_work[loss_column],
                errors="coerce",
            ).fillna(0.0)

        if "newly_defaulted" in contagion_work.columns:
            contagion_work["_default_flag"] = (
                contagion_work["newly_defaulted"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .astype(int)
            )
        else:
            contagion_work["_default_flag"] = 0

        loss_aggregation = (
            "max"
            if loss_column == "cumulative_contagion_loss"
            else "sum"
        )

        contagion_aggregated = (
            contagion_work.groupby("entity_id", as_index=False)
            .agg(
                entity_contagion_loss=("_loss", loss_aggregation),
                entity_default_evidence=("_default_flag", "max"),
            )
        )
        
        result = result.merge(
            contagion_aggregated,
            on="entity_id",
            how="left",
        )

        result["entity_contagion_loss"] = (
            pd.to_numeric(
                result["entity_contagion_loss"],
                errors="coerce",
            ).fillna(0.0)
        )
        result["entity_default_evidence"] = (
            pd.to_numeric(
                result["entity_default_evidence"],
                errors="coerce",
            ).fillna(0.0)
        )

        contagion_raw = (
            _normalise_score(result["entity_contagion_loss"]) * 0.80
            + _normalise_score(result["entity_default_evidence"]) * 0.20
        )
    else:
        result["entity_contagion_loss"] = 0.0
        result["entity_default_evidence"] = 0.0
        contagion_raw = pd.Series(
            0.0,
            index=result.index,
            dtype=float,
        )

    # Incorporate reconciliation discrepancies as an additional monitoring
    # driver. They do not alter the three configured component weights.
    if (
        not reconciliation_df.empty
        and "entity_id" in reconciliation_df.columns
    ):
        reconciliation_work = reconciliation_df.copy()

        discrepancy_column = _first_existing_column(
            reconciliation_work,
            [
                "reconciliation_difference",
                "difference",
                "unreconciled_amount",
                "exposure_difference",
            ],
        )

        if discrepancy_column is not None:
            reconciliation_work["_discrepancy"] = pd.to_numeric(
                reconciliation_work[discrepancy_column],
                errors="coerce",
            ).fillna(0.0).abs()

            discrepancy_by_entity = (
                reconciliation_work.groupby("entity_id")["_discrepancy"]
                .sum()
            )

            result["reconciliation_discrepancy"] = (
                result["entity_id"]
                .map(discrepancy_by_entity)
                .fillna(0.0)
            )
        else:
            result["reconciliation_discrepancy"] = 0.0
    else:
        result["reconciliation_discrepancy"] = 0.0

    # Combine the three components into a monitoring-priority index.
    result["network_score"] = network_raw
    result["financial_stress_score"] = financial_raw
    result["contagion_score"] = contagion_raw

    result["systemic_risk_score"] = (
        result["network_score"] * NETWORK_WEIGHT
        + result["financial_stress_score"] * FINANCIAL_STRESS_WEIGHT
        + result["contagion_score"] * CONTAGION_WEIGHT
    ).round(2)

    result["risk_tier"] = pd.cut(
        result["systemic_risk_score"],
        bins=[-0.01, 25, 50, 75, 100],
        labels=["Low", "Moderate", "High", "Very High"],
        include_lowest=True,
    ).astype(str)

    def identify_drivers(row: pd.Series) -> str:
        components = {
            "network exposure": row["network_score"],
            "financial stress": row["financial_stress_score"],
            "contagion": row["contagion_score"],
        }

        ordered = sorted(
            components.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if not any(score > 0 for _, score in ordered):
            return "No elevated component score"

        return ", ".join(
            name
            for name, score in ordered[:2]
            if score > 0
        )

    result["primary_risk_drivers"] = result.apply(
        identify_drivers,
        axis=1,
    )

    result = result.sort_values(
        by=["systemic_risk_score", "entity_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    result.insert(0, "systemic_risk_rank", range(1, len(result) + 1))

    return result


def main() -> None:
    """Rebuild the assessment from the previously exported project datasets."""
    required_files = {
        "systemic_importance": PROCESSED_DIR / "systemic_importance.csv",
        "integrated_stress": PROCESSED_DIR / "integrated_stress_results.csv",
        "contagion_results": PROCESSED_DIR / "contagion_results.csv",
        "reconciliation": PROCESSED_DIR / "entity_reconciliation.csv",
    }

    missing_files = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Cannot build systemic-risk assessment. Missing input files: "
            + ", ".join(missing_files)
        )

    assessment = calculate_systemic_risk_assessment(
        systemic_importance=pd.read_csv(
            required_files["systemic_importance"]
        ),
        integrated_stress=pd.read_csv(
            required_files["integrated_stress"]
        ),
        contagion_results=pd.read_csv(
            required_files["contagion_results"]
        ),
        reconciliation=pd.read_csv(
            required_files["reconciliation"]
        ),
    )

    output_path = PROCESSED_DIR / "systemic_risk_assessment.csv"
    assessment.to_csv(output_path, index=False)

    print(f"Systemic-risk assessment saved to: {output_path}")
    print(f"Entities assessed: {len(assessment)}")
    print("\nTop 10 entities by monitoring-priority score:")

    display_columns = [
        column
        for column in [
            "systemic_risk_rank",
            "entity_id",
            "entity_type",
            "systemic_risk_score",
            "risk_tier",
            "primary_risk_drivers",
        ]
        if column in assessment.columns
    ]

    print(assessment[display_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
