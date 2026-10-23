
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from fincontagion.contagion_engine import (
    ContagionRoundResult,
    simulate_credit_contagion,
)
from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition


@dataclass(frozen=True, slots=True)
class ContagionScenario:
    """Configuration for one credit-contagion simulation."""

    scenario_id: str
    initial_defaulted_entity_ids: frozenset[str]
    recovery_rate: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string.")

        if not isinstance(self.initial_defaulted_entity_ids, frozenset):
            raise TypeError(
                "initial_defaulted_entity_ids must be a frozenset."
            )

        if not self.initial_defaulted_entity_ids:
            raise ValueError(
                "At least one initial default must be specified."
            )

        if any(
            not isinstance(entity_id, str) or not entity_id.strip()
            for entity_id in self.initial_defaulted_entity_ids
        ):
            raise ValueError(
                "Initial default entity IDs must be non-empty strings."
            )

        if not isinstance(self.recovery_rate, Decimal):
            raise TypeError("recovery_rate must be a Decimal.")

        if (
            not self.recovery_rate.is_finite()
            or not Decimal("0") <= self.recovery_rate <= Decimal("1")
        ):
            raise ValueError(
                "recovery_rate must be finite and between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    """Results and assumptions associated with one scenario."""

    scenario_id: str
    initial_defaulted_entity_ids: tuple[str, ...]
    recovery_rate: Decimal
    contagion_results: tuple[ContagionRoundResult, ...]


def run_contagion_scenario(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    positions: list[FinancialPosition],
    scenario: ContagionScenario,
) -> ScenarioRunResult:
    """Run one configured scenario on an existing financial network."""

    results = simulate_credit_contagion(
        entities=entities,
        exposures=exposures,
        positions=positions,
        initial_defaulted_entity_ids=set(
            scenario.initial_defaulted_entity_ids
        ),
        recovery_rate=scenario.recovery_rate,
    )

    return ScenarioRunResult(
        scenario_id=scenario.scenario_id,
        initial_defaulted_entity_ids=tuple(
            sorted(scenario.initial_defaulted_entity_ids)
        ),
        recovery_rate=scenario.recovery_rate,
        contagion_results=tuple(results),
    )


def run_contagion_scenarios(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    positions: list[FinancialPosition],
    scenarios: Iterable[ContagionScenario],
) -> list[ScenarioRunResult]:
    """Run several scenarios against the same financial network."""

    scenario_list = list(scenarios)

    if not scenario_list:
        raise ValueError("At least one scenario must be provided.")

    scenario_ids = [scenario.scenario_id for scenario in scenario_list]

    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario IDs must be unique within a batch.")

    return [
        run_contagion_scenario(
            entities=entities,
            exposures=exposures,
            positions=positions,
            scenario=scenario,
        )
        for scenario in scenario_list
    ]
