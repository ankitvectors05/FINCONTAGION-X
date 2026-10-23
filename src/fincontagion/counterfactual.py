
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from fincontagion.contagion_engine import simulate_credit_contagion
from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """Summary of one independent initial-default scenario."""

    initial_default_entity_id: str
    recovery_rate: Decimal
    total_contagion_losses: Decimal
    affected_entities: int
    additional_defaults: int
    max_propagation_round: int


def run_counterfactual_analysis(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    positions: list[FinancialPosition],
    candidate_entity_ids: list[str],
    recovery_rate: Decimal = Decimal("0.40"),
) -> list[CounterfactualResult]:
    """
    Compare separate initial-default scenarios on the same network.

    Each candidate is simulated independently. Losses and defaults from
    one simulation do not carry over into another.

    Total contagion losses represent losses recorded by creditors during
    propagation rounds. They are not necessarily total economic losses.
    """

    if not candidate_entity_ids:
        raise ValueError("Provide at least one candidate entity ID.")

    if len(candidate_entity_ids) != len(set(candidate_entity_ids)):
        raise ValueError("Candidate entity IDs must be unique.")

    known_entity_ids = {entity.entity_id for entity in entities}
    unknown_ids = set(candidate_entity_ids) - known_entity_ids

    if unknown_ids:
        raise ValueError(
            f"Unknown candidate entity IDs: {sorted(unknown_ids)}"
        )

    results: list[CounterfactualResult] = []

    for candidate_id in candidate_entity_ids:
        simulation = simulate_credit_contagion(
            entities=entities,
            exposures=exposures,
            positions=positions,
            initial_defaulted_entity_ids={candidate_id},
            recovery_rate=recovery_rate,
        )

        propagated = [
            row for row in simulation
            if row.round_number > 0
        ]

        total_losses = sum(
            (row.round_loss for row in propagated),
            start=Decimal("0"),
        )

        affected_entity_ids = {
            row.entity_id
            for row in propagated
            if row.round_loss > Decimal("0")
        }

        additional_default_ids = {
            row.entity_id
            for row in propagated
            if row.newly_defaulted
            and row.entity_id != candidate_id
        }

        max_round = max(
            (row.round_number for row in propagated),
            default=0,
        )

        results.append(
            CounterfactualResult(
                initial_default_entity_id=candidate_id,
                recovery_rate=recovery_rate,
                total_contagion_losses=total_losses,
                affected_entities=len(affected_entity_ids),
                additional_defaults=len(additional_default_ids),
                max_propagation_round=max_round,
            )
        )

    # Rank primarily by additional defaults, then losses, then reach.
    return sorted(
        results,
        key=lambda result: (
            result.additional_defaults,
            result.total_contagion_losses,
            result.affected_entities,
        ),
        reverse=True,
    )


def save_counterfactual_results(
    results: list[CounterfactualResult],
    output_path: str | Path,
) -> pd.DataFrame:
    """Save counterfactual scenario summaries to CSV."""

    if not results:
        raise ValueError("Cannot save an empty counterfactual result list.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "systemic_impact_rank": rank,
            "initial_default_entity_id": result.initial_default_entity_id,
            "recovery_rate": float(result.recovery_rate),
            "total_contagion_losses": float(
                result.total_contagion_losses
            ),
            "affected_entities": result.affected_entities,
            "additional_defaults": result.additional_defaults,
            "max_propagation_round": result.max_propagation_round,
        }
        for rank, result in enumerate(results, start=1)
    ]

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(path, index=False)
    return dataframe
