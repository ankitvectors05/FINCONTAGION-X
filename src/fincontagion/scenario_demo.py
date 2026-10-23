
from decimal import Decimal

from fincontagion.data.generator import (
    DatasetConfig,
    generate_entities,
    generate_instruments_and_exposures,
    validate_relationships,
)
from fincontagion.data.position_generator import generate_financial_positions
from fincontagion.scenario_runner import (
    ContagionScenario,
    run_contagion_scenarios,
)
from fincontagion.scenario_validation import validate_scenario_result
from fincontagion.contagion_validation import (
    validate_contagion_results_independently,
)


def main() -> None:
    """Compare and independently validate recovery-rate scenarios."""

    # 1. Generate one deterministic synthetic network.
    config = DatasetConfig(seed=42)
    entities = generate_entities(config)

    instruments, exposures = generate_instruments_and_exposures(
        entities=entities,
        config=config,
    )

    validate_relationships(
        entities=entities,
        instruments=instruments,
        exposures=exposures,
    )

    positions = generate_financial_positions(
        entities=entities,
        exposures=exposures,
        seed=config.seed,
    )

    # 2. Confirm the initial default exists.
    known_entity_ids = {entity.entity_id for entity in entities}
    initial_default_id = "BANK_0010"

    if initial_default_id not in known_entity_ids:
        raise ValueError(
            f"Initial default entity {initial_default_id} does not exist."
        )

    initial_defaults = frozenset({initial_default_id})

    # 3. Configure scenarios; only recovery rate changes.
    scenarios = [
        ContagionScenario(
            scenario_id="CONTAGION_BASELINE",
            initial_defaulted_entity_ids=initial_defaults,
            recovery_rate=Decimal("0.40"),
        ),
        ContagionScenario(
            scenario_id="CONTAGION_LOW_RECOVERY",
            initial_defaulted_entity_ids=initial_defaults,
            recovery_rate=Decimal("0.20"),
        ),
        ContagionScenario(
            scenario_id="CONTAGION_HIGH_RECOVERY",
            initial_defaulted_entity_ids=initial_defaults,
            recovery_rate=Decimal("0.60"),
        ),
    ]

    # 4. Run every scenario on the same network.
    scenario_results = run_contagion_scenarios(
        entities=entities,
        exposures=exposures,
        positions=positions,
        scenarios=scenarios,
    )

    # 5. Run both structural and independent validation.
    for scenario_result in scenario_results:
        validate_scenario_result(
            result=scenario_result,
            entities=entities,
        )

        validate_contagion_results_independently(
            entities=entities,
            exposures=exposures,
            positions=positions,
            initial_defaulted_entity_ids=set(
                scenario_result.initial_defaulted_entity_ids
            ),
            recovery_rate=scenario_result.recovery_rate,
            actual_results=list(scenario_result.contagion_results),
        )

        print(
            f"Scenario {scenario_result.scenario_id}: "
            "structural and independent validation passed."
        )

    # 6. Report comparable results.
    print("\nFINCONTAGION-X: Recovery Sensitivity Analysis")
    print(f"Entities: {len(entities)}")
    print(f"Exposures: {len(exposures)}")
    print(f"Initial default: {initial_default_id}")
    print("-" * 92)

    print(
        f"{'Scenario ID':<30}"
        f"{'Recovery':>12}"
        f"{'Affected':>12}"
        f"{'New defaults':>15}"
        f"{'Total losses (INR)':>23}"
    )

    print("-" * 92)

    for scenario_result in scenario_results:
        rows = scenario_result.contagion_results

        affected_entities = {
            row.entity_id
            for row in rows
            if row.round_number > 0 and row.round_loss > 0
        }

        additional_defaults = {
            row.entity_id
            for row in rows
            if row.round_number > 0 and row.newly_defaulted
        }

        total_losses = sum(
            (row.round_loss for row in rows),
            start=Decimal("0"),
        )

        print(
            f"{scenario_result.scenario_id:<30}"
            f"{scenario_result.recovery_rate * 100:>11.0f}%"
            f"{len(affected_entities):>12}"
            f"{len(additional_defaults):>15}"
            f"{total_losses:>23,.2f}"
        )

    print("-" * 92)
    print(
        "Results are deterministic outcomes under the specified "
        "synthetic network and recovery assumptions."
    )
    print(
        "Validation checks mathematical consistency against the documented "
        "model; it does not establish real-world predictive accuracy."
    )


if __name__ == "__main__":
    main()
