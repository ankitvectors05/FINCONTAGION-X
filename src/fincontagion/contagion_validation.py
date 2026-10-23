
from collections import defaultdict
from decimal import Decimal

from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition
from fincontagion.contagion_engine import (
    ContagionRoundResult,
    simulate_credit_contagion,
)



def validate_contagion_results_independently(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    positions: list[FinancialPosition],
    initial_defaulted_entity_ids: set[str],
    recovery_rate: Decimal,
    actual_results: list[ContagionRoundResult],
    starting_equity_by_entity: dict[str, Decimal] | None = None,
) -> None:

    """
    Independently recompute contagion waves and compare with engine output.

    This checks mathematical consistency under the documented simplified
    model. It does not establish real-world predictive accuracy.
    """

    if not isinstance(recovery_rate, Decimal):
        raise TypeError("Recovery rate must be a Decimal.")

    if (
        not recovery_rate.is_finite()
        or not Decimal("0") <= recovery_rate <= Decimal("1")
    ):
        raise ValueError("Recovery rate must be finite and between 0 and 1.")

    if not isinstance(initial_defaulted_entity_ids, set):
        raise TypeError("Initial default IDs must be provided as a set.")

    entity_ids = [entity.entity_id for entity in entities]
    entity_id_set = set(entity_ids)

    if len(entity_ids) != len(entity_id_set):
        raise ValueError("Duplicate entity IDs found.")

    if not entity_id_set:
        raise ValueError("At least one entity must be provided.")

    if not initial_defaulted_entity_ids:
        raise ValueError("At least one initial default is required.")

    if not initial_defaulted_entity_ids.issubset(entity_id_set):
        raise ValueError("An initial default is not present in the network.")

    position_by_id = {position.entity_id: position for position in positions}

    if len(position_by_id) != len(positions):
        raise ValueError("Duplicate financial position entity IDs found.")

    if set(position_by_id) != entity_id_set:
        raise ValueError(
            "Financial positions must contain exactly one row per entity."
        )

    
    if starting_equity_by_entity is None:
        starting_equity = {
            entity_id: position_by_id[entity_id].equity_capital
            for entity_id in entity_id_set
        }
    else:
        if set(starting_equity_by_entity) != entity_id_set:
            raise ValueError(
                "Starting-equity mapping must contain exactly one value per entity."
            )

        starting_equity = starting_equity_by_entity.copy()

        for entity_id, equity in starting_equity.items():
            if not isinstance(equity, Decimal) or not equity.is_finite():
                raise ValueError(
                    f"Invalid starting equity for entity {entity_id}."
                )


    exposures_by_debtor: dict[str, list[FinancialExposure]] = defaultdict(list)
    exposure_ids: set[str] = set()

    for exposure in exposures:
        if exposure.exposure_id in exposure_ids:
            raise ValueError(f"Duplicate exposure ID: {exposure.exposure_id}")
        exposure_ids.add(exposure.exposure_id)

        if exposure.creditor_id not in entity_id_set:
            raise ValueError(f"Unknown creditor: {exposure.creditor_id}")

        if exposure.debtor_id not in entity_id_set:
            raise ValueError(f"Unknown debtor: {exposure.debtor_id}")

        if exposure.creditor_id == exposure.debtor_id:
            raise ValueError(f"Self-exposure found: {exposure.exposure_id}")

        if (
            not exposure.outstanding_amount.is_finite()
            or exposure.outstanding_amount <= Decimal("0")
        ):
            raise ValueError(
                f"Invalid outstanding amount: {exposure.exposure_id}"
            )

        exposures_by_debtor[exposure.debtor_id].append(exposure)

    # Build an independent expected result using the documented rules.
    expected_rows: list[ContagionRoundResult] = []
    cumulative_losses = {
        entity_id: Decimal("0") for entity_id in entity_id_set
    }
    defaulted = set(initial_defaulted_entity_ids)
    processed_exposures: set[str] = set()

    for entity_id in entity_ids:
        initial_default = entity_id in initial_defaulted_entity_ids
        equity = starting_equity[entity_id]

        if initial_default:
            status = "initial_default"
        elif equity <= Decimal("0"):
            status = "stress_impaired"
        else:
            status = "active"

        expected_rows.append(
            ContagionRoundResult(
                entity_id=entity_id,
                round_number=0,
                round_loss=Decimal("0"),
                cumulative_contagion_loss=Decimal("0"),
                equity_before_contagion=equity,
                equity_after_contagion=equity,
                newly_defaulted=initial_default,
                status=status,
            )
        )
    current_wave = set(initial_defaulted_entity_ids)
    round_number = 1

    while current_wave:
        losses_by_creditor: dict[str, Decimal] = defaultdict(
            lambda: Decimal("0")
        )

        for debtor_id in current_wave:
            for exposure in exposures_by_debtor[debtor_id]:
                if exposure.exposure_id in processed_exposures:
                    continue

                processed_exposures.add(exposure.exposure_id)

                losses_by_creditor[exposure.creditor_id] += (
                    exposure.outstanding_amount
                    * (Decimal("1") - recovery_rate)
                )

        next_wave: set[str] = set()

        for entity_id in entity_ids:
            round_loss = losses_by_creditor[entity_id]

            if round_loss == Decimal("0"):
                continue

            equity_before = (
                starting_equity[entity_id] - cumulative_losses[entity_id]
            )

            cumulative_losses[entity_id] += round_loss

            equity_after = (
                starting_equity[entity_id] - cumulative_losses[entity_id]
            )

            newly_defaulted = (
                entity_id not in defaulted
                and starting_equity[entity_id] > Decimal("0")
                and cumulative_losses[entity_id] >= starting_equity[entity_id]
            )

            if newly_defaulted:
                defaulted.add(entity_id)
                next_wave.add(entity_id)

            if newly_defaulted:
                status = "new_default"
            elif entity_id in defaulted:
                status = "already_defaulted"
            else:
                status = "affected"

            expected_rows.append(
                ContagionRoundResult(
                    entity_id=entity_id,
                    round_number=round_number,
                    round_loss=round_loss,
                    cumulative_contagion_loss=cumulative_losses[entity_id],
                    equity_before_contagion=equity_before,
                    equity_after_contagion=equity_after,
                    newly_defaulted=newly_defaulted,
                    status=status,
                )
            )

        current_wave = next_wave
        round_number += 1

    # Compare every result field, not just aggregate losses.
    if len(actual_results) != len(expected_rows):
        raise ValueError(
            "Independent validation failed: result row count differs. "
            f"Expected {len(expected_rows)}, got {len(actual_results)}."
        )

    fields_to_compare = (
        "entity_id",
        "round_number",
        "round_loss",
        "cumulative_contagion_loss",
        "equity_before_contagion",
        "equity_after_contagion",
        "newly_defaulted",
        "status",
    )

    for index, (actual, expected) in enumerate(
        zip(actual_results, expected_rows)
    ):
        for field_name in fields_to_compare:
            actual_value = getattr(actual, field_name)
            expected_value = getattr(expected, field_name)

            if actual_value != expected_value:
                raise ValueError(
                    "Independent validation failed at row "
                    f"{index}, field '{field_name}': "
                    f"expected {expected_value!r}, got {actual_value!r}."
                )

    # Explicit accounting checks for every recorded row.
    for row in actual_results:
        if row.equity_after_contagion != (
            row.equity_before_contagion - row.round_loss
        ):
            raise ValueError(
                "Equity change does not match the recorded round loss "
                f"for entity {row.entity_id}, round {row.round_number}."
            )

    actual_total_loss = sum(
        (row.round_loss for row in actual_results),
        start=Decimal("0"),
    )
    expected_total_loss = sum(
        (row.round_loss for row in expected_rows),
        start=Decimal("0"),
    )

    if actual_total_loss != expected_total_loss:
        raise ValueError("Independent total-loss comparison failed.")
