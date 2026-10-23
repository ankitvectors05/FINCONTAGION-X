"""Build scenario-labelled data for the FINCONTAGION-X ML prototype.

Labels describe outcomes under illustrative synthetic stress scenarios.
They are not observed defaults or calibrated real-world probabilities.
"""

from dataclasses import asdict, is_dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from fincontagion.data.generator import (
    DatasetConfig,
    generate_entities,
    generate_instruments_and_exposures,
)
from fincontagion.data.position_generator import generate_financial_positions
from fincontagion.integrated_stress import calculate_integrated_stress
from fincontagion.ml.features import build_feature_dataset
from fincontagion.scenarios import StressScenario, STRESS_SCENARIOS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "ml_scenario_dataset.csv"
FEATURE_OUTPUT_PATH = DATA_DIR / "ml_features.csv"



def _records_to_dataframe(records: list, name: str) -> pd.DataFrame:
    """Convert dictionaries or domain objects into a DataFrame."""
    if not records:
        raise ValueError(f"{name} cannot be empty.")

    converted_records = []

    for record in records:
        if isinstance(record, dict):
            converted_records.append(record)
        elif is_dataclass(record) and not isinstance(record, type):
            converted_records.append(asdict(record))
        elif hasattr(record, "model_dump"):
            converted_records.append(record.model_dump())
        elif hasattr(record, "__dict__"):
            converted_records.append(vars(record))
        else:
            raise TypeError(
                f"Unsupported record type in {name}: "
                f"{type(record).__name__}"
            )

    dataframe = pd.DataFrame(converted_records)

    if dataframe.empty:
        raise ValueError(f"Conversion produced an empty DataFrame for {name}.")

    return dataframe


def build_scenario_dataset(
    seed: int = 42,
    output_path: Path = OUTPUT_PATH,
    feature_output_path: Path = FEATURE_OUTPUT_PATH,
) -> pd.DataFrame:
    """Create one row per entity per stress scenario.

    Parameters
    ----------
    seed:
        Random seed used to generate the synthetic financial ecosystem.
    output_path:
        Destination CSV for the entity-scenario dataset.
    feature_output_path:
        Destination CSV for the entity-level feature dataset. This can be
        set separately during cross-seed validation to avoid overwriting
        the baseline feature artifact.
    """
    output_path = Path(output_path)
    feature_output_path = Path(feature_output_path)

    # 1. Generate a single reproducible synthetic ecosystem.
    config = DatasetConfig(seed=seed)
    entities = generate_entities(config)

    instruments, exposures = generate_instruments_and_exposures(
        entities=entities,
        config=config,
    )

    

    positions = generate_financial_positions(
        entities=entities,
        exposures=exposures,
        seed=seed,
    )

    # 2. Convert domain objects for feature engineering only.
    # Keep the original lists for the stress engine.
    entities_df = _records_to_dataframe(entities, "entities")
    positions_df = _records_to_dataframe(positions, "positions")
    exposures_df = _records_to_dataframe(exposures, "exposures")

    # Ensure enum values are represented consistently as strings.
    if "entity_type" in entities_df.columns:
        entities_df["entity_type"] = entities_df["entity_type"].map(
            lambda value: value.value if hasattr(value, "value") else value
        )

    features = build_feature_dataset(
        entities=entities_df,
        positions=positions_df,
        exposures=exposures_df,
        output_path=feature_output_path,
    )

    if features["entity_id"].duplicated().any():
        raise ValueError("Features contain duplicate entity IDs.")

    expected_entity_ids = set(entities_df["entity_id"])
    feature_entity_ids = set(features["entity_id"])

    if feature_entity_ids != expected_entity_ids:
        raise ValueError(
            "Feature/entity ID mismatch. "
            f"Missing features: "
            f"{len(expected_entity_ids - feature_entity_ids)}; "
            f"unexpected features: "
            f"{len(feature_entity_ids - expected_entity_ids)}."
        )

    # 3. Calculate outcomes for each scenario.
    scenario_frames = []

    for scenario in STRESS_SCENARIOS:
        results = calculate_integrated_stress(
            entities=entities,
            positions=positions,
            exposures=exposures,
            instruments=instruments,
            scenario=scenario,
        )

        outcomes = pd.DataFrame(results)

        required = {
            "entity_id",
            "scenario_id",
            "credit_loss",
            "market_loss",
            "interest_rate_loss",
            "interest_rate_gain",
            "interest_rate_valuation_change",
            "net_accounting_impact",
            "total_accounting_loss",
            "stressed_equity",
            "cash_before_stress",
            "funding_withdrawal",
            "cash_after_withdrawal",
            "liquidity_shortfall",
            "network_lending_exposure",
            "outgoing_exposure_count",
            "interest_rate_loss_included",
            "interest_rate_assumption_basis",
            "assumption_basis",
        }

        missing = required - set(outcomes.columns)
        if missing:
            raise ValueError(
                f"Stress results for {scenario.scenario_id} are missing "
                f"required columns: {sorted(missing)}"
            )

        if outcomes["entity_id"].duplicated().any():
            raise ValueError(
                f"Duplicate entity outcomes in {scenario.scenario_id}."
            )

        outcome_entity_ids = set(outcomes["entity_id"])

        if outcome_entity_ids != expected_entity_ids:
            raise ValueError(
                f"Entity coverage mismatch in {scenario.scenario_id}. "
                f"Missing outcomes: "
                f"{len(expected_entity_ids - outcome_entity_ids)}; "
                f"unexpected outcomes: "
                f"{len(outcome_entity_ids - expected_entity_ids)}."
            )

        stressed_equity = pd.to_numeric(
            outcomes["stressed_equity"], errors="raise"
        )
        liquidity_shortfall = pd.to_numeric(
            outcomes["liquidity_shortfall"], errors="raise"
        )

        # Synthetic label rule: modelled equity depletion or liquidity gap.
        outcomes["distress_label"] = (
            (stressed_equity <= 0) | (liquidity_shortfall > 0)
        ).astype("int8")

        outcomes["scenario_name"] = scenario.name
        outcomes["asset_price_decline"] = float(scenario.asset_price_decline)
        outcomes["funding_withdrawal_rate"] = float(
            scenario.funding_withdrawal_rate
        )
        outcomes["interest_rate_shock_configured"] = float(
            scenario.interest_rate_shock
        )

        scenario_frames.append(
            outcomes[
                [
                    "entity_id",
                    "scenario_id",
                    "scenario_name",

                    # Post-stress outcomes: retain for auditing and analysis.
                    "credit_loss",
                    "market_loss",
                    "interest_rate_loss",
                    "interest_rate_gain",
                    "interest_rate_valuation_change",
                    "net_accounting_impact",
                    "total_accounting_loss",
                    "stressed_equity",

                    # Liquidity and network outcomes.
                    "cash_before_stress",
                    "funding_withdrawal",
                    "cash_after_withdrawal",
                    "liquidity_shortfall",
                    "outgoing_exposure_count",

                    # Target and scenario configuration.
                    "distress_label",
                    "asset_price_decline",
                    "funding_withdrawal_rate",
                    "interest_rate_shock_configured",
                    "interest_rate_loss_included",
                    "interest_rate_assumption_basis",
                    "assumption_basis",
                ]
            ]
        )

    # 4. Join outcomes to features from the same ecosystem.
    all_outcomes = pd.concat(scenario_frames, ignore_index=True)

    dataset = all_outcomes.merge(
        features,
        on="entity_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = dataset["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} scenario rows have no matching features."
        )

    dataset = dataset.drop(columns="_merge")

    if dataset["entity_type"].isna().any():
        raise ValueError("Some scenario outcomes have missing entity types.")

    if dataset.duplicated(["entity_id", "scenario_id"]).any():
        raise ValueError("Duplicate entity/scenario combinations detected.")

    expected_rows = len(entities) * len(STRESS_SCENARIOS)

    if len(dataset) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows "
            f"({len(entities)} entities x {len(STRESS_SCENARIOS)} scenarios), "
            f"but generated {len(dataset)}."
        )

    if dataset["distress_label"].nunique() < 2:
        print(
            "\nWARNING: Only one distress-label class was generated. "
            "Do not train a classifier yet; inspect the scenario outcomes "
            "and documented label rule."
        )

    # 5. Save the datasets.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print("\nScenario dataset generated successfully.")
    print(f"Seed: {seed}")
    print(f"Rows: {len(dataset)}")
    print(f"Unique entities: {dataset['entity_id'].nunique()}")
    print(f"Scenarios: {dataset['scenario_id'].nunique()}")
    print(f"Expected rows: {expected_rows}")
    print(f"Feature file: {feature_output_path}")
    print(f"Scenario dataset: {output_path}")

    print("\nDistress-label distribution:")
    print(dataset["distress_label"].value_counts().sort_index().to_string())

    print("\nDistress rate by scenario:")
    print(
        dataset.groupby("scenario_id")["distress_label"]
        .agg(["count", "sum", "mean"])
        .to_string()
    )

    print("\nInterest-rate modelling status:")
    print(
        dataset.groupby("scenario_id")["interest_rate_loss_included"]
        .first()
        .to_string()
    )

    print("\nAssumption basis:")
    print(dataset["assumption_basis"].value_counts().to_string())

    return dataset


if __name__ == "__main__":
    build_scenario_dataset()
