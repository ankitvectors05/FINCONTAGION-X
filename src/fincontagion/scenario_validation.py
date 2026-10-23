
"""Independent validation of contagion scenario results."""

from decimal import Decimal

from fincontagion.scenario_runner import ScenarioRunResult


VALID_STATUSES = {
    "initial_default",
    "active",
    "capital_impaired",
    "affected",
    "new_default",
    "already_defaulted",
}


def _is_finite_decimal(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite()


def validate_scenario_result(
    result: ScenarioRunResult,
    entities: list,
) -> None:
    """Validate scenario output independently of the contagion engine."""

    prefix = f"{result.scenario_id}:"

    if not isinstance(result.scenario_id, str) or not result.scenario_id.strip():
        raise ValueError("Scenario ID must be a non-empty string.")

    if not isinstance(result.recovery_rate, Decimal):
        raise ValueError(f"{prefix} recovery rate must be a Decimal.")

    if (
        not result.recovery_rate.is_finite()
        or not Decimal("0") <= result.recovery_rate <= Decimal("1")
    ):
        raise ValueError(f"{prefix} invalid recovery rate.")

    if not entities:
        raise ValueError(f"{prefix} entities cannot be empty.")

    entity_ids = [entity.entity_id for entity in entities]

    if any(
        not isinstance(entity_id, str) or not entity_id.strip()
        for entity_id in entity_ids
    ):
        raise ValueError(f"{prefix} entity IDs must be non-empty strings.")

    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError(f"{prefix} duplicate entity IDs.")

    known_entities = set(entity_ids)
    initial_defaults = set(result.initial_defaulted_entity_ids)

    if not initial_defaults:
        raise ValueError(f"{prefix} initial defaults cannot be empty.")

    if not initial_defaults.issubset(known_entities):
        raise ValueError(f"{prefix} unknown initial default entity.")

    rows = list(result.contagion_results)

    if not rows:
        raise ValueError(f"{prefix} scenario results cannot be empty.")

    rows_by_entity: dict[str, list] = {}
    rows_by_round: dict[int, list] = {}
    seen_pairs: set[tuple[str, int]] = set()

    for row in rows:
        if row.entity_id not in known_entities:
            raise ValueError(f"{prefix} unknown entity {row.entity_id}.")

        if (
            isinstance(row.round_number, bool)
            or not isinstance(row.round_number, int)
            or row.round_number < 0
        ):
            raise ValueError(f"{prefix} invalid round number.")

        pair = (row.entity_id, row.round_number)

        if pair in seen_pairs:
            raise ValueError(
                f"{prefix} duplicate entity/round: {pair}."
            )

        seen_pairs.add(pair)

        numeric_fields = {
            "round_loss": row.round_loss,
            "cumulative_contagion_loss": row.cumulative_contagion_loss,
            "equity_before_contagion": row.equity_before_contagion,
            "equity_after_contagion": row.equity_after_contagion,
        }

        for field_name, value in numeric_fields.items():
            if not _is_finite_decimal(value):
                raise ValueError(
                    f"{prefix} {field_name} must be a finite Decimal."
                )

        if row.round_loss < 0:
            raise ValueError(f"{prefix} round loss cannot be negative.")

        if row.cumulative_contagion_loss < 0:
            raise ValueError(
                f"{prefix} cumulative loss cannot be negative."
            )

        if not isinstance(row.newly_defaulted, bool):
            raise ValueError(
                f"{prefix} newly_defaulted must be a boolean."
            )

        if row.status not in VALID_STATUSES:
            raise ValueError(
                f"{prefix} invalid status {row.status!r}."
            )

        if (
            row.equity_after_contagion
            != row.equity_before_contagion - row.round_loss
        ):
            raise ValueError(
                f"{prefix} equity change does not match round loss "
                f"for {row.entity_id}, round {row.round_number}."
            )

        rows_by_entity.setdefault(row.entity_id, []).append(row)
        rows_by_round.setdefault(row.round_number, []).append(row)

    # Round zero must contain every entity exactly once.
    round_zero = rows_by_round.get(0, [])

    if len(round_zero) != len(known_entities):
        raise ValueError(
            f"{prefix} round zero must contain every entity exactly once."
        )

    round_zero_by_entity = {row.entity_id: row for row in round_zero}

    if set(round_zero_by_entity) != known_entities:
        raise ValueError(
            f"{prefix} round-zero entity coverage is incomplete."
        )

    for entity_id, row in round_zero_by_entity.items():
        if row.round_loss != Decimal("0"):
            raise ValueError(
                f"{prefix} round-zero loss must be zero for {entity_id}."
            )

        if row.cumulative_contagion_loss != Decimal("0"):
            raise ValueError(
                f"{prefix} round-zero cumulative loss must be zero "
                f"for {entity_id}."
            )

        if row.equity_after_contagion != row.equity_before_contagion:
            raise ValueError(
                f"{prefix} round-zero equity must remain unchanged "
                f"for {entity_id}."
            )

        if entity_id in initial_defaults:
            expected_status = "initial_default"

            if not row.newly_defaulted:
                raise ValueError(
                    f"{prefix} initial default must be flagged in round zero."
                )
        else:
            if row.newly_defaulted:
                raise ValueError(
                    f"{prefix} non-initial entity cannot be newly defaulted "
                    "in round zero."
                )

            if row.equity_before_contagion == Decimal("0"):
                expected_status = "capital_impaired"
            else:
                expected_status = "active"

        if row.status != expected_status:
            raise ValueError(
                f"{prefix} incorrect round-zero status for {entity_id}: "
                f"expected {expected_status!r}, got {row.status!r}."
            )

    # Entity records are sparse after round zero: no-loss rounds are omitted.
    observed_rounds = sorted(rows_by_round)

    if not observed_rounds or observed_rounds[0] != 0:
        raise ValueError(f"{prefix} simulation must begin at round zero.")

    if observed_rounds != list(range(observed_rounds[-1] + 1)):
        raise ValueError(f"{prefix} simulation rounds must be contiguous.")

    for entity_id, entity_rows in rows_by_entity.items():
        entity_rows.sort(key=lambda row: row.round_number)
        previous = entity_rows[0]

        if previous.round_number != 0:
            raise ValueError(
                f"{prefix} missing round-zero record for {entity_id}."
            )

        # Initial defaults are already defaulted before contagion begins.
        previously_defaulted = entity_id in initial_defaults

        for row in entity_rows[1:]:
            if row.round_number <= previous.round_number:
                raise ValueError(
                    f"{prefix} rounds are not strictly increasing "
                    f"for {entity_id}."
                )

            expected_cumulative = (
                previous.cumulative_contagion_loss + row.round_loss
            )

            if row.cumulative_contagion_loss != expected_cumulative:
                raise ValueError(
                    f"{prefix} cumulative-loss arithmetic is inconsistent "
                    f"for {entity_id}, round {row.round_number}."
                )

            if (
                row.equity_before_contagion
                != previous.equity_after_contagion
            ):
                raise ValueError(
                    f"{prefix} equity continuity is inconsistent "
                    f"for {entity_id}, round {row.round_number}."
                )

            if row.status == "initial_default":
                raise ValueError(
                    f"{prefix} initial_default status is only valid "
                    f"in round zero."
                )

            if row.newly_defaulted and row.status != "new_default":
                raise ValueError(
                    f"{prefix} newly defaulted entity must have "
                    f"status 'new_default'."
                )

            if row.status == "new_default" and not row.newly_defaulted:
                raise ValueError(
                    f"{prefix} new_default status requires "
                    f"newly_defaulted=True."
                )

            if row.newly_defaulted:
                if previously_defaulted:
                    raise ValueError(
                        f"{prefix} an already-defaulted entity cannot "
                        f"default again: {entity_id}."
                    )

                if row.equity_before_contagion <= Decimal("0"):
                    raise ValueError(
                        f"{prefix} newly defaulted entity must have "
                        f"positive equity before the loss: {entity_id}."
                    )

                if row.equity_after_contagion > Decimal("0"):
                    raise ValueError(
                        f"{prefix} new default must exhaust equity: "
                        f"{entity_id}."
                    )

                previously_defaulted = True

            elif row.status == "already_defaulted":
                if not previously_defaulted:
                    raise ValueError(
                        f"{prefix} already_defaulted status used before "
                        f"the entity defaulted: {entity_id}."
                    )

            previous = row

    total_loss = sum(
        (row.round_loss for row in rows),
        start=Decimal("0"),
    )

    if not total_loss.is_finite():
        raise ValueError(f"{prefix} total simulated loss is not finite.")
