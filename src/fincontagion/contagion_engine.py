
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition


@dataclass(frozen=True, slots=True)
class ContagionRoundResult:
    """Outcome for one entity in one round of contagion simulation."""

    entity_id: str
    round_number: int
    round_loss: Decimal
    cumulative_contagion_loss: Decimal
    equity_before_contagion: Decimal
    equity_after_contagion: Decimal
    newly_defaulted: bool
    status: str


def simulate_credit_contagion(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    positions: list[FinancialPosition],
    initial_defaulted_entity_ids: set[str],
    recovery_rate: Decimal = Decimal("0.40"),
    starting_equity_by_entity: dict[str, Decimal] | None = None,
) -> list[ContagionRoundResult]:
    """
    Run a simplified, deterministic credit-contagion simulation.

    A defaulted debtor causes each creditor to lose:
        exposure amount * (1 - recovery rate)

    Losses reduce creditor equity. A creditor with positive starting
    equity becomes a new default when cumulative contagion losses reach
    or exceed that starting equity.

    Zero-equity entities that have not been explicitly defaulted are
    marked capital_impaired, not automatically defaulted. They do not
    initiate contagion propagation unless explicitly seeded as defaults.

    Each exposure generates a loss at most once.

    Limitations:
    - No collateral, netting, guarantees, or recovery delays.
    - No liquidity feedback or regulatory/legal default rules.
    - No negative starting equity, because FinancialPosition rejects it.
    - Simplified recovery and default mechanics; not a predictive model.
    """

    if not isinstance(recovery_rate, Decimal):
        raise TypeError("recovery_rate must be a Decimal.")

    if (
        not recovery_rate.is_finite()
        or not Decimal("0") <= recovery_rate <= Decimal("1")
    ):
        raise ValueError("recovery_rate must be finite and between 0 and 1.")

    entity_by_id = {entity.entity_id: entity for entity in entities}

    if len(entity_by_id) != len(entities):
        raise ValueError("Duplicate entity IDs found.")

    if not entity_by_id:
        raise ValueError("At least one entity must be provided.")

    position_by_id = {position.entity_id: position for position in positions}

    if len(position_by_id) != len(positions):
        raise ValueError("Duplicate financial position entity IDs found.")

    if set(position_by_id) != set(entity_by_id):
        raise ValueError(
            "Financial positions must contain exactly one row per entity."
        )

    if not isinstance(initial_defaulted_entity_ids, set):
        raise TypeError("initial_defaulted_entity_ids must be a set.")

    unknown_initial_defaults = (
        initial_defaulted_entity_ids - set(entity_by_id)
    )

    if unknown_initial_defaults:
        raise ValueError(
            "Unknown initial default entity IDs: "
            f"{sorted(unknown_initial_defaults)}"
        )

    for exposure in exposures:
        if exposure.creditor_id not in entity_by_id:
            raise ValueError(f"Unknown creditor: {exposure.creditor_id}")

        if exposure.debtor_id not in entity_by_id:
            raise ValueError(f"Unknown debtor: {exposure.debtor_id}")

        if exposure.creditor_id == exposure.debtor_id:
            raise ValueError(
                f"Self-exposure found: {exposure.exposure_id}"
            )

    if starting_equity_by_entity is None:
        starting_equity = {
            entity_id: position.equity_capital
            for entity_id, position in position_by_id.items()
        }
    else:
        if not isinstance(starting_equity_by_entity, dict):
            raise TypeError(
                "starting_equity_by_entity must be a dictionary."
            )

        if set(starting_equity_by_entity) != set(entity_by_id):
            raise ValueError(
                "Starting-equity mapping must contain exactly one "
                "value for every entity."
            )

        starting_equity = {}

        for entity_id, equity in starting_equity_by_entity.items():
            if not isinstance(equity, Decimal):
                raise TypeError(
                    f"Starting equity for {entity_id} must be a Decimal."
                )

            if not equity.is_finite():
                raise ValueError(
                    f"Starting equity for {entity_id} must be finite."
                )

            starting_equity[entity_id] = equity

    exposures_by_debtor: dict[str, list[FinancialExposure]] = defaultdict(list)
    exposure_ids: set[str] = set()

    for exposure in exposures:
        if exposure.exposure_id in exposure_ids:
            raise ValueError(f"Duplicate exposure ID: {exposure.exposure_id}")

        exposure_ids.add(exposure.exposure_id)
        exposures_by_debtor[exposure.debtor_id].append(exposure)

    defaulted = set(initial_defaulted_entity_ids)
    cumulative_loss = {
        entity_id: Decimal("0")
        for entity_id in entity_by_id
    }
    processed_exposure_ids: set[str] = set()
    results: list[ContagionRoundResult] = []

    # Record the initial state. Only explicitly seeded entities are defaults.
    for entity in entities:
        entity_id = entity.entity_id
        equity = starting_equity[entity_id]
        is_initial_default = entity_id in defaulted

        if is_initial_default:
            status = "initial_default"
        elif equity <= Decimal("0"):
            status = "stress_impaired"
        else:
            status = "active"

        results.append(
            ContagionRoundResult(
                entity_id=entity_id,
                round_number=0,
                round_loss=Decimal("0"),
                cumulative_contagion_loss=Decimal("0"),
                equity_before_contagion=equity,
                equity_after_contagion=equity,
                newly_defaulted=is_initial_default,
                status=status,
            )
        )

    current_default_wave = set(initial_defaulted_entity_ids)
    round_number = 1

    while current_default_wave:
        round_loss_by_creditor: dict[str, Decimal] = defaultdict(
            lambda: Decimal("0")
        )

        # Only defaults in the current wave trigger creditor losses.
        for defaulted_debtor_id in current_default_wave:
            for exposure in exposures_by_debtor[defaulted_debtor_id]:
                if exposure.exposure_id in processed_exposure_ids:
                    continue

                processed_exposure_ids.add(exposure.exposure_id)

                loss = exposure.outstanding_amount * (
                    Decimal("1") - recovery_rate
                )
                round_loss_by_creditor[exposure.creditor_id] += loss

        next_default_wave: set[str] = set()

        for entity in entities:
            entity_id = entity.entity_id
            round_loss = round_loss_by_creditor[entity_id]

            if round_loss == 0:
                continue

            equity_before = (
                starting_equity[entity_id] - cumulative_loss[entity_id]
            )

            cumulative_loss[entity_id] += round_loss

            equity_after = (
                starting_equity[entity_id] - cumulative_loss[entity_id]
            )

            newly_defaulted = (
                entity_id not in defaulted
                and starting_equity[entity_id] > Decimal("0")
                and cumulative_loss[entity_id] >= starting_equity[entity_id]
            )

            if newly_defaulted:
                defaulted.add(entity_id)
                next_default_wave.add(entity_id)

            if newly_defaulted:
                status = "new_default"
            elif entity_id in defaulted:
                status = "already_defaulted"
            elif starting_equity[entity_id] == Decimal("0"):
                status = "capital_impaired"
            else:
                status = "affected"

            results.append(
                ContagionRoundResult(
                    entity_id=entity_id,
                    round_number=round_number,
                    round_loss=round_loss,
                    cumulative_contagion_loss=cumulative_loss[entity_id],
                    equity_before_contagion=equity_before,
                    equity_after_contagion=equity_after,
                    newly_defaulted=newly_defaulted,
                    status=status,
                )
            )

        current_default_wave = next_default_wave
        round_number += 1

    return results
