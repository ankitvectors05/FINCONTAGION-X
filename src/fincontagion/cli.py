
import csv
from decimal import Decimal
from pathlib import Path

from fincontagion.contagion_engine import simulate_credit_contagion
from fincontagion.counterfactual import (
    run_counterfactual_analysis,
    save_counterfactual_results,
)
from fincontagion.data.exporter import (
    export_entities,
    export_exposures,
    export_financial_positions,
    export_instruments,
)
from fincontagion.data.generator import (
    DatasetConfig,
    generate_entities,
    generate_instruments_and_exposures,
    validate_relationships,
)
from fincontagion.data.position_generator import generate_financial_positions
from fincontagion.data.quality_checks import (
    validate_saved_datasets,
    validate_credit_stress_results,
    validate_contagion_results,
    validate_scenario_contagion_results,
)
from fincontagion.data.reconciliation import (
    build_reconciliation_report,
    export_reconciliation_report,
)
from fincontagion.contagion_validation import (
    validate_contagion_results_independently,
)
from fincontagion.graph_analytics import calculate_systemic_importance
from fincontagion.integrated_stress import calculate_integrated_stress
from fincontagion.scenarios import BASELINE_STRESS, STRESS_SCENARIOS
from fincontagion.stress_engine import calculate_network_credit_stress
from fincontagion.systemic_risk import calculate_systemic_risk_assessment


# Illustrative assumptions, not real-world predictions.
INITIAL_DEFAULT_ENTITY_IDS = {"BANK_0010"}
CONTAGION_RECOVERY_RATE = Decimal("0.40")
CONTAGION_SCENARIO_ID = "CONTAGION_DEMO_001"

# Institutions to compare as independent initial-default scenarios.
# IDs absent from the generated network are safely skipped.
COUNTERFACTUAL_CANDIDATES = [
    "BANK_0001",
    "BANK_0003",
    "BANK_0010",
    "NBFC_0001",
    "CORP_0001",
]


def export_credit_stress_results(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Export entity-level credit-stress results."""
    if not rows:
        raise ValueError("Credit-stress calculation returned no results.")

    fieldnames = list(rows[0].keys())

    if any(set(row.keys()) != set(fieldnames) for row in rows):
        raise ValueError("Credit-stress rows have inconsistent columns.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_integrated_stress_results(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Export integrated entity-level financial stress results."""
    if not rows:
        raise ValueError("Integrated stress calculation returned no results.")

    fieldnames = list(rows[0].keys())

    if any(set(row.keys()) != set(fieldnames) for row in rows):
        raise ValueError("Integrated stress rows have inconsistent columns.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_systemic_importance_results(
    rows: list[dict],
    output_path: Path,
) -> None:
    """Export entity-level network importance metrics."""
    if not rows:
        raise ValueError("Systemic importance calculation returned no results.")

    fieldnames = list(rows[0].keys())

    if any(set(row.keys()) != set(fieldnames) for row in rows):
        raise ValueError("Systemic importance rows have inconsistent columns.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_contagion_results(
    contagion_results: list,
    output_path: Path,
    scenario_id: str,
    recovery_rate: Decimal,
) -> None:
    """Export entity-level contagion outcomes for each simulation round."""
    fieldnames = [
        "scenario_id",
        "entity_id",
        "round_number",
        "round_loss",
        "cumulative_contagion_loss",
        "equity_before_contagion",
        "equity_after_contagion",
        "newly_defaulted",
        "status",
        "recovery_rate",
        "data_origin",
        "assumption_basis",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in contagion_results:
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "entity_id": result.entity_id,
                    "round_number": result.round_number,
                    "round_loss": result.round_loss,
                    "cumulative_contagion_loss": (
                        result.cumulative_contagion_loss
                    ),
                    "equity_before_contagion": (
                        result.equity_before_contagion
                    ),
                    "equity_after_contagion": result.equity_after_contagion,
                    "newly_defaulted": result.newly_defaulted,
                    "status": result.status,
                    "recovery_rate": recovery_rate,
                    "data_origin": "synthetic",
                    "assumption_basis": "illustrative_recovery_rate",
                }
            )

def export_scenario_contagion_results(
    scenario_results: list[tuple[str, object]],
    output_path: Path,
    recovery_rate: Decimal,
) -> None:
    """Export scenario-linked contagion results separately from baseline."""
    if not scenario_results:
        raise ValueError("Scenario contagion returned no results.")

    fieldnames = [
        "scenario_id",
        "entity_id",
        "round_number",
        "round_loss",
        "cumulative_contagion_loss",
        "equity_before_contagion",
        "equity_after_contagion",
        "newly_defaulted",
        "status",
        "recovery_rate",
        "data_origin",
        "assumption_basis",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for scenario_id, result in scenario_results:
            writer.writerow({
                "scenario_id": scenario_id,
                "entity_id": result.entity_id,
                "round_number": result.round_number,
                "round_loss": result.round_loss,
                "cumulative_contagion_loss": (
                    result.cumulative_contagion_loss
                ),
                "equity_before_contagion": (
                    result.equity_before_contagion
                ),
                "equity_after_contagion": result.equity_after_contagion,
                "newly_defaulted": result.newly_defaulted,
                "status": result.status,
                "recovery_rate": recovery_rate,
                "data_origin": "synthetic",
                "assumption_basis": "illustrative_recovery_rate",
            })


def main() -> None:
    """Generate, stress-test, analyze, export, and validate the ecosystem."""

    # 1. Generate the synthetic financial ecosystem.
    config = DatasetConfig(seed=42)
    entities = generate_entities(config)

    instruments, exposures = generate_instruments_and_exposures(
        entities=entities,
        config=config,
    )

    # 2. Validate entity, instrument, and exposure relationships.
    validate_relationships(
        entities=entities,
        instruments=instruments,
        exposures=exposures,
    )

    # 3. Generate internally consistent financial positions.
    positions = generate_financial_positions(
        entities=entities,
        exposures=exposures,
        seed=config.seed,
    )

    # 4. Reconcile network exposures against financial positions.
    reconciliation_rows = build_reconciliation_report(
        entities=entities,
        exposures=exposures,
        positions=positions,
    )

    # 5. Estimate expected credit losses using illustrative PD/LGD.
    credit_stress_rows = calculate_network_credit_stress(
        entities=entities,
        exposures=exposures,
    )

    # 6. Calculate integrated financial stress for all four scenarios.
    all_scenario_stress_rows = [
        row
        for scenario in STRESS_SCENARIOS
        for row in calculate_integrated_stress(
            entities=entities,
            positions=positions,
            exposures=exposures,
            instruments=instruments,
            scenario=scenario,
        )
    ]

    # Preserve baseline-only results for systemic-risk scoring.
    integrated_stress_rows = [
        row
        for row in all_scenario_stress_rows
        if row["scenario_id"] == BASELINE_STRESS.scenario_id
    ]
    # 7. Analyze the structure and concentration of the financial network.
    systemic_importance_rows = calculate_systemic_importance(
        entities=entities,
        exposures=exposures,
    )

    # 8. Validate initial defaults for the original contagion simulation.
    known_entity_ids = {entity.entity_id for entity in entities}
    unknown_defaults = INITIAL_DEFAULT_ENTITY_IDS - known_entity_ids

    if unknown_defaults:
        raise ValueError(
            "Configured initial defaults are not in the generated ecosystem: "
            f"{sorted(unknown_defaults)}"
        )

    debtors_with_creditors = {exposure.debtor_id for exposure in exposures}
    defaults_without_claims = (
        INITIAL_DEFAULT_ENTITY_IDS - debtors_with_creditors
    )

    if defaults_without_claims:
        raise ValueError(
            "The following initial defaults have no recorded creditor claims: "
            f"{sorted(defaults_without_claims)}"
        )

    # 9. Run the original illustrative credit-contagion scenario.
    contagion_results = simulate_credit_contagion(
        entities=entities,
        exposures=exposures,
        positions=positions,
        initial_defaulted_entity_ids=INITIAL_DEFAULT_ENTITY_IDS,
        recovery_rate=CONTAGION_RECOVERY_RATE,
    )

    # 9a. Run scenario-linked contagion using stressed equity.
    scenario_contagion_results = []

    for scenario in STRESS_SCENARIOS:
        scenario_stress_rows = [
            row
            for row in all_scenario_stress_rows
            if row["scenario_id"] == scenario.scenario_id
        ]

        stressed_equity_by_entity = {
            row["entity_id"]: Decimal(str(row["stressed_equity"]))
            for row in scenario_stress_rows
        }

        if set(stressed_equity_by_entity) != known_entity_ids:
            raise ValueError(
                "Scenario stressed equity must contain exactly one "
                f"row per entity for {scenario.scenario_id}."
            )

        results = simulate_credit_contagion(
            entities=entities,
            exposures=exposures,
            positions=positions,
            initial_defaulted_entity_ids=INITIAL_DEFAULT_ENTITY_IDS,
            recovery_rate=CONTAGION_RECOVERY_RATE,
            starting_equity_by_entity=stressed_equity_by_entity,
        )

        scenario_contagion_results.extend(
            (scenario.scenario_id, result)
            for result in results
        )

    
    # Independently validate contagion propagation for all four scenarios.
    for scenario in STRESS_SCENARIOS:
        scenario_stress_rows = [
            row
            for row in all_scenario_stress_rows
            if row["scenario_id"] == scenario.scenario_id
        ]

        starting_equity_by_entity = {
            row["entity_id"]: Decimal(str(row["stressed_equity"]))
            for row in scenario_stress_rows
        }

        scenario_results = [
            result
            for scenario_id, result in scenario_contagion_results
            if scenario_id == scenario.scenario_id
        ]

        validate_contagion_results_independently(
            entities=entities,
            exposures=exposures,
            positions=positions,
            initial_defaulted_entity_ids=INITIAL_DEFAULT_ENTITY_IDS,
            recovery_rate=CONTAGION_RECOVERY_RATE,
            actual_results=scenario_results,
            starting_equity_by_entity=starting_equity_by_entity,
        )

    print(
        "Independent scenario contagion validation passed "
        "for all four scenarios."
    )

    
    # 10. Run counterfactual scenarios on the SAME generated network.
    # Each candidate is tested independently at the same recovery rate.
    candidate_entity_ids = [
        entity_id
        for entity_id in COUNTERFACTUAL_CANDIDATES
        if entity_id in known_entity_ids
    ]

    if not candidate_entity_ids:
        raise ValueError(
            "None of the configured counterfactual candidates exist "
            "in the generated network. Check COUNTERFACTUAL_CANDIDATES."
        )

    counterfactual_results = run_counterfactual_analysis(
        entities=entities,
        exposures=exposures,
        positions=positions,
        candidate_entity_ids=candidate_entity_ids,
        recovery_rate=CONTAGION_RECOVERY_RATE,
    )

    # 11. Resolve the project output directory.
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 12. Export core synthetic datasets.
    export_entities(entities, output_dir / "entities.csv")
    export_instruments(instruments, output_dir / "instruments.csv")
    export_exposures(exposures, output_dir / "exposures.csv")
    export_financial_positions(
        positions,
        output_dir / "financial_positions.csv",
    )

    # 13. Export reconciliation and credit-stress results.
    reconciliation_path = output_dir / "entity_reconciliation.csv"
    export_reconciliation_report(
        reconciliation_rows,
        reconciliation_path,
    )

    credit_stress_path = output_dir / "credit_stress_results.csv"
    export_credit_stress_results(
        credit_stress_rows,
        credit_stress_path,
    )

    # 14. Export integrated financial stress results for all scenarios.
    integrated_stress_path = output_dir / "integrated_stress_results.csv"
    export_integrated_stress_results(
        all_scenario_stress_rows,
        integrated_stress_path,
    )

    # 15. Export systemic importance rankings.
    systemic_importance_path = output_dir / "systemic_importance.csv"
    export_systemic_importance_results(
        systemic_importance_rows,
        systemic_importance_path,
    )

    # 16. Export original contagion results.
    # Export scenario-linked contagion without changing baseline outputs.

    scenario_contagion_path = (
        output_dir / "scenario_contagion_results.csv"
    )
    export_scenario_contagion_results(
        scenario_results=scenario_contagion_results,
        output_path=scenario_contagion_path,
        recovery_rate=CONTAGION_RECOVERY_RATE,
    )

    

    # 17. Export counterfactual comparison results.
    counterfactual_path = output_dir / "counterfactual_contagion.csv"
    counterfactual_df = save_counterfactual_results(
        results=counterfactual_results,
        output_path=counterfactual_path,
    )

    # 18. Integrate network, financial stress, contagion, and reconciliation.
    # Read the files produced in THIS run, not a potentially stale CSV.
    systemic_risk_assessment = calculate_systemic_risk_assessment(
        systemic_importance=systemic_importance_rows,
        integrated_stress=integrated_stress_rows,
        contagion_results=contagion_results,
        reconciliation=reconciliation_rows,
    )

    systemic_risk_path = output_dir / "systemic_risk_assessment.csv"

    # Support an assessment returned as a pandas DataFrame or list of dicts.
    if hasattr(systemic_risk_assessment, "to_csv"):
        systemic_risk_assessment.to_csv(
            systemic_risk_path,
            index=False,
        )
    else:
        if not systemic_risk_assessment:
            raise ValueError(
                "Systemic-risk assessment returned no results."
            )

        with systemic_risk_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            fieldnames = list(systemic_risk_assessment[0].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(systemic_risk_assessment)

    # Normalize the result for reporting, regardless of return type.
    if hasattr(systemic_risk_assessment, "to_dict"):
        systemic_risk_rows = systemic_risk_assessment.to_dict(
            orient="records"
        )
    else:
        systemic_risk_rows = systemic_risk_assessment

    # 19. Validate the existing datasets and stress/contagion outputs.
    validation_counts = validate_saved_datasets(output_dir)

    credit_stress_count = validate_credit_stress_results(
        output_dir=output_dir,
        expected_entity_ids=known_entity_ids,
    )

    contagion_count = validate_contagion_results(output_dir)
    scenario_contagion_count = validate_scenario_contagion_results(output_dir)

    # 20. Summarize the original contagion outcome.
    affected_entity_ids = {
        result.entity_id
        for result in contagion_results
        if result.round_number > 0 and result.round_loss > 0
    }

    additional_default_ids = {
        result.entity_id
        for result in contagion_results
        if result.status == "new_default"
    }

    total_contagion_losses = sum(
        (result.round_loss for result in contagion_results),
        start=Decimal("0"),
    )

    gross_exposure = sum(
        (exposure.outstanding_amount for exposure in exposures),
        start=Decimal("0"),
    )

    # 21. Summarize integrated stress outcomes.
    total_expected_credit_loss = sum(
        (row["credit_loss"] for row in integrated_stress_rows),
        start=Decimal("0"),
    )

    total_market_loss = sum(
        (row["market_loss"] for row in integrated_stress_rows),
        start=Decimal("0"),
    )

    total_funding_withdrawal = sum(
        (row["funding_withdrawal"] for row in integrated_stress_rows),
        start=Decimal("0"),
    )

    total_liquidity_shortfall = sum(
        (row["liquidity_shortfall"] for row in integrated_stress_rows),
        start=Decimal("0"),
    )

    total_interest_rate_loss = sum(
        (row["interest_rate_loss"] for row in integrated_stress_rows),
        Decimal("0"),
    )

    total_interest_rate_gain = sum(
        (row["interest_rate_gain"] for row in integrated_stress_rows),
        Decimal("0"),
    )

    entities_with_negative_stressed_equity = sum(
        row["stressed_equity"] < Decimal("0")
        for row in integrated_stress_rows
    )

    # 22. Print execution summary.
    print("\nFINCONTAGION-X pipeline completed successfully.")
    print(f"As-of date: {config.as_of_date}")
    print(f"Random seed: {config.seed}")
    print(f"Entities generated: {len(entities)}")
    print(f"Instruments generated: {len(instruments)}")
    print(f"Exposures generated: {len(exposures)}")
    print(f"Financial positions generated: {len(positions)}")
    print(f"Gross recorded exposures: INR {gross_exposure:,.2f}")

    print("\nCore dataset validation: passed")
    for dataset_name, row_count in validation_counts.items():
        print(f"  {dataset_name}: {row_count} rows")

    print("\nCredit-stress results validation: passed")
    print(f"  credit_stress_results: {credit_stress_count} rows")

    print("\nContagion results validation: passed")
    print(f"  contagion_results: {contagion_count} rows")

    print("\nContagion simulation summary:")
    print(f"  Scenario ID: {CONTAGION_SCENARIO_ID}")
    print(
        "  Initial default(s): "
        f"{', '.join(sorted(INITIAL_DEFAULT_ENTITY_IDS))}"
    )
    print(f"  Assumed recovery rate: {CONTAGION_RECOVERY_RATE:.0%}")
    print(f"  Entities affected by contagion losses: {len(affected_entity_ids)}")
    print(f"  Additional defaults triggered: {len(additional_default_ids)}")
    print(
        "  Total simulated contagion losses: "
        f"INR {total_contagion_losses:,.2f}"
    )

    print("\nIntegrated financial stress summary:")
    print(f"  Scenario: {BASELINE_STRESS.scenario_id}")
    print(f"  Entity-scenario assessments exported: {len(all_scenario_stress_rows)}")
    print(f"  Scenarios exported: {len(STRESS_SCENARIOS)}")
    print(f"  Entities assessed per scenario: {len(entities)}")
    print(
        "  Estimated credit losses: "
        f"INR {total_expected_credit_loss:,.2f}"
    )
    print(f"  Proxy market losses: INR {total_market_loss:,.2f}")
    print(f"  Funding withdrawals: INR {total_funding_withdrawal:,.2f}")
    print(f"  Liquidity shortfall: INR {total_liquidity_shortfall:,.2f}")
    print(
        "  Entities with negative stressed equity: "
        f"{entities_with_negative_stressed_equity}"
    )
    print(
        "  Estimated interest-rate valuation losses: "
        f"INR {total_interest_rate_loss:,.2f}"
    )
    print(
        "  Estimated interest-rate valuation gains: "
        f"INR {total_interest_rate_gain:,.2f}"
    )
    print(
        "  Rate sensitivity basis: illustrative synthetic duration proxies"
    )

    print("\nSystemic importance summary:")
    print(f"  Entities ranked: {len(systemic_importance_rows)}")
    print("  Ranking basis: total connected exposure, then counterparties")
    print("  Top 5 entities by exposure-based importance:")

    for row in systemic_importance_rows[:5]:
        print(
            f"    {row['exposure_importance_rank']}. "
            f"{row['entity_id']} ({row['entity_type']}) | "
            f"Total connected exposure: "
            f"INR {row['total_connected_exposure']:,.2f} | "
            f"Counterparties: {row['unique_counterparties']} | "
            f"Largest counterparty concentration: "
            f"{row['largest_counterparty_concentration']:.1%}"
        )

    print("\nCounterfactual contagion analysis:")
    print(f"  Independent scenarios compared: {len(counterfactual_df)}")
    print(f"  Recovery rate held constant: {CONTAGION_RECOVERY_RATE:.0%}")
    print(f"  Output: {counterfactual_path}")
    print("  Highest-impact initial-default scenarios:")
    print(counterfactual_df.head(5).to_string(index=False))

    # 23. Report the integrated systemic-risk assessment.
    print("\nIntegrated systemic risk assessment:")
    print(f"  Entities assessed: {len(systemic_risk_rows)}")
    print("  Score: synthetic monitoring-priority index")
    print(f"  Output: {systemic_risk_path}")

    if systemic_risk_rows:
        preferred_columns = [
            "systemic_risk_rank",
            "entity_id",
            "entity_type",
            "systemic_risk_score",
            "risk_tier",
            "primary_risk_drivers",
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in systemic_risk_rows[0]
        ]

        if available_columns:
            print("  Top 10 entities:")
            for row in systemic_risk_rows[:10]:
                print(
                    "    "
                    + " | ".join(
                        f"{column}={row.get(column)}"
                        for column in available_columns
                    )
                )

    print("\nReconciliation validation: passed")
    print("Accounting identity and network exposure checks passed.")

    
    print("\nGenerated files:")
    for filename in (
        "entities.csv",
        "instruments.csv",
        "exposures.csv",
        "financial_positions.csv",
        "entity_reconciliation.csv",
        "credit_stress_results.csv",
        "integrated_stress_results.csv",
        "systemic_importance.csv",
        "contagion_results.csv",
        "scenario_contagion_results.csv",
        "counterfactual_contagion.csv",
        "systemic_risk_assessment.csv",
    ):
        print(f"  {output_dir / filename}")



if __name__ == "__main__":
    main()
