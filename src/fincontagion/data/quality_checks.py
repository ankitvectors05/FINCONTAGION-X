
import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fincontagion.stress_assumptions import get_credit_assumption


REQUIRED_COLUMNS = {
    "entities.csv": {
        "entity_id",
        "name",
        "entity_type",
        "country",
        "base_currency",
        "data_origin",
    },
    "instruments.csv": {
        "instrument_id",
        "instrument_type",
        "issuer_id",
        "currency",
        "issue_date",
        "maturity_date",
        "rate_type",
        "annual_interest_rate",
        "data_origin",
    },
    "exposures.csv": {
        "exposure_id",
        "instrument_id",
        "creditor_id",
        "debtor_id",
        "outstanding_amount",
        "currency",
        "as_of_date",
        "data_origin",
    },
    "financial_positions.csv": {
        "entity_id",
        "as_of_date",
        "cash_and_liquid_assets",
        "total_assets",
        "total_liabilities",
        "equity_capital",
        "short_term_liabilities",
        "data_origin",
    },
    "entity_reconciliation.csv": {
        "entity_id",
        "entity_type",
        "network_lending_assets",
        "network_borrowing_liabilities",
        "residual_assets",
        "residual_liabilities",
        "cash_and_liquid_assets",
        "total_assets",
        "total_liabilities",
        "equity_capital",
        "accounting_identity",
        "data_origin",
    },
    "contagion_results.csv": {
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
    },
    "credit_stress_results.csv": {
        "entity_id",
        "entity_type",
        "outgoing_exposure_count",
        "network_lending_exposure",
        "estimated_credit_loss",
        "data_origin",
        "assumption_basis",
    },
}


def _read_csv(
    file_path: Path,
    required_columns: set[str],
) -> list[dict[str, str]]:
    """Read a CSV and verify its required columns."""

    if not file_path.exists():
        raise FileNotFoundError(f"Required dataset not found: {file_path}")

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {file_path.name}")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset is empty: {file_path.name}")

    return rows


def _decimal(value: str, field_name: str) -> Decimal:
    """Convert a CSV value to a finite Decimal."""

    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"Invalid decimal value for {field_name}: {value!r}"
        ) from exc

    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite.")

    return result


def _require_unique(
    rows: list[dict[str, str]],
    key: str,
    dataset_name: str,
) -> set[str]:
    """Require non-empty, unique values for a column."""

    values = [row[key] for row in rows]

    if any(not value.strip() for value in values):
        raise ValueError(
            f"{dataset_name} contains an empty {key}."
        )

    if len(values) != len(set(values)):
        raise ValueError(
            f"{dataset_name} contains duplicate {key} values."
        )

    return set(values)


def _validate_synthetic_origin(
    rows: list[dict[str, str]],
    dataset_name: str,
) -> None:
    """Ensure the synthetic data provenance is explicit."""

    for row in rows:
        if row["data_origin"] != "synthetic":
            raise ValueError(
                f"{dataset_name} contains an unexpected data origin."
            )


def _validate_positions(
    entities: list[dict[str, str]],
    exposures: list[dict[str, str]],
    positions: list[dict[str, str]],
) -> None:
    """Validate position coverage, amounts, dates and accounting identity."""

    entity_ids = {row["entity_id"] for row in entities}
    position_ids = _require_unique(
        positions,
        "entity_id",
        "financial_positions.csv",
    )

    if position_ids != entity_ids:
        raise ValueError(
            "Financial positions must contain exactly one row per entity."
        )

    exposure_dates = {row["as_of_date"] for row in exposures}
    position_dates = {row["as_of_date"] for row in positions}

    if len(exposure_dates) != 1:
        raise ValueError(
            "Exposures must share one as-of date for this dataset version."
        )

    if position_dates != exposure_dates:
        raise ValueError(
            "Position dates do not match the exposure as-of date."
        )

    for row in positions:
        entity_id = row["entity_id"]

        cash = _decimal(row["cash_and_liquid_assets"], "cash_and_liquid_assets")
        assets = _decimal(row["total_assets"], "total_assets")
        liabilities = _decimal(row["total_liabilities"], "total_liabilities")
        equity = _decimal(row["equity_capital"], "equity_capital")
        short_term = _decimal(
            row["short_term_liabilities"],
            "short_term_liabilities",
        )

        amounts = {
            "cash_and_liquid_assets": cash,
            "total_assets": assets,
            "total_liabilities": liabilities,
            "equity_capital": equity,
            "short_term_liabilities": short_term,
        }

        for name, amount in amounts.items():
            if amount < 0:
                raise ValueError(
                    f"{name} cannot be negative for {entity_id}."
                )

        if cash > assets:
            raise ValueError(
                f"Cash exceeds total assets for {entity_id}."
            )

        if short_term > liabilities:
            raise ValueError(
                f"Short-term liabilities exceed total liabilities for {entity_id}."
            )

        if assets != liabilities + equity:
            raise ValueError(
                f"Accounting identity failed for {entity_id}."
            )

        _validate_synthetic_origin([row], "financial_positions.csv")


def _validate_relationships(
    entities: list[dict[str, str]],
    instruments: list[dict[str, str]],
    exposures: list[dict[str, str]],
) -> None:
    """Validate references and basic financial relationship rules."""

    entity_ids = _require_unique(entities, "entity_id", "entities.csv")
    instrument_ids = _require_unique(
        instruments,
        "instrument_id",
        "instruments.csv",
    )
    _require_unique(exposures, "exposure_id", "exposures.csv")

    for row in entities:
        currency = row["base_currency"]

        if len(currency) != 3 or not currency.isalpha() or not currency.isupper():
            raise ValueError(
                f"Invalid base currency for {row['entity_id']}."
            )

    instrument_by_id = {
        row["instrument_id"]: row for row in instruments
    }

    for row in instruments:
        if row["issuer_id"] not in entity_ids:
            raise ValueError(
                f"Unknown issuer {row['issuer_id']} in instruments.csv."
            )

        currency = row["currency"]

        if len(currency) != 3 or not currency.isalpha() or not currency.isupper():
            raise ValueError(
                f"Invalid instrument currency for {row['instrument_id']}."
            )

        issue_date = row["issue_date"]
        maturity_date = row["maturity_date"]

        if maturity_date and maturity_date < issue_date:
            raise ValueError(
                f"Maturity precedes issue date for {row['instrument_id']}."
            )

        rate_type = row["rate_type"]
        interest_text = row["annual_interest_rate"]

        if rate_type == "not_applicable" and interest_text:
            raise ValueError(
                f"Non-rate-bearing instrument {row['instrument_id']} "
                "has an interest rate."
            )

        if rate_type != "not_applicable":
            if not interest_text:
                raise ValueError(
                    f"Rate-bearing instrument {row['instrument_id']} "
                    "has no interest rate."
                )

            interest_rate = _decimal(
                interest_text,
                "annual_interest_rate",
            )

            if interest_rate < 0:
                raise ValueError(
                    f"Negative interest rate in {row['instrument_id']}."
                )

    total_lending = Decimal("0")
    total_borrowing = Decimal("0")
    exposure_dates: set[str] = set()

    for row in exposures:
        creditor_id = row["creditor_id"]
        debtor_id = row["debtor_id"]
        instrument_id = row["instrument_id"]

        if creditor_id not in entity_ids:
            raise ValueError(f"Unknown creditor: {creditor_id}")

        if debtor_id not in entity_ids:
            raise ValueError(f"Unknown debtor: {debtor_id}")

        if creditor_id == debtor_id:
            raise ValueError(
                f"Self-exposure found: {creditor_id}."
            )

        if instrument_id not in instrument_ids:
            raise ValueError(f"Unknown instrument: {instrument_id}")

        instrument = instrument_by_id[instrument_id]

        if instrument["issuer_id"] != debtor_id:
            raise ValueError(
                f"Issuer/debtor mismatch for exposure {row['exposure_id']}."
            )

        amount = _decimal(
            row["outstanding_amount"],
            "outstanding_amount",
        )

        if amount <= 0:
            raise ValueError(
                f"Exposure amount must be positive: {row['exposure_id']}."
            )

        currency = row["currency"]

        if len(currency) != 3 or not currency.isalpha() or not currency.isupper():
            raise ValueError(
                f"Invalid currency in exposure {row['exposure_id']}."
            )

        if currency != instrument["currency"]:
            raise ValueError(
                f"Exposure/instrument currency mismatch for {row['exposure_id']}."
            )

        exposure_dates.add(row["as_of_date"])
        total_lending += amount
        total_borrowing += amount

    if len(exposure_dates) != 1:
        raise ValueError(
            "Exposures must share one as-of date for this dataset version."
        )

    if total_lending != total_borrowing:
        raise ValueError(
            "System lending and borrowing totals do not reconcile."
        )

    _validate_synthetic_origin(entities, "entities.csv")
    _validate_synthetic_origin(instruments, "instruments.csv")
    _validate_synthetic_origin(exposures, "exposures.csv")


def _validate_reconciliation(
    entities: list[dict[str, str]],
    exposures: list[dict[str, str]],
    positions: list[dict[str, str]],
    reconciliation: list[dict[str, str]],
) -> None:
    """Recompute network totals and accounting checks independently."""

    entity_ids = {row["entity_id"] for row in entities}
    reconciliation_ids = _require_unique(
        reconciliation,
        "entity_id",
        "entity_reconciliation.csv",
    )

    if reconciliation_ids != entity_ids:
        raise ValueError(
            "Reconciliation report must contain exactly one row per entity."
        )

    lending: dict[str, Decimal] = defaultdict_decimal()
    borrowing: dict[str, Decimal] = defaultdict_decimal()

    for row in exposures:
        amount = _decimal(row["outstanding_amount"], "outstanding_amount")
        lending[row["creditor_id"]] += amount
        borrowing[row["debtor_id"]] += amount

    position_by_id = {row["entity_id"]: row for row in positions}
    reconciliation_by_id = {
        row["entity_id"]: row for row in reconciliation
    }

    for entity_id in entity_ids:
        position = position_by_id[entity_id]
        report = reconciliation_by_id[entity_id]

        assets = _decimal(position["total_assets"], "total_assets")
        liabilities = _decimal(
            position["total_liabilities"],
            "total_liabilities",
        )
        equity = _decimal(position["equity_capital"], "equity_capital")

        reported_lending = _decimal(
            report["network_lending_assets"],
            "network_lending_assets",
        )
        reported_borrowing = _decimal(
            report["network_borrowing_liabilities"],
            "network_borrowing_liabilities",
        )
        residual_assets = _decimal(
            report["residual_assets"],
            "residual_assets",
        )
        residual_liabilities = _decimal(
            report["residual_liabilities"],
            "residual_liabilities",
        )

        if reported_lending != lending[entity_id]:
            raise ValueError(
                f"Reconciliation lending mismatch for {entity_id}."
            )

        if reported_borrowing != borrowing[entity_id]:
            raise ValueError(
                f"Reconciliation borrowing mismatch for {entity_id}."
            )

        if assets - reported_lending != residual_assets:
            raise ValueError(
                f"Residual asset mismatch for {entity_id}."
            )

        if liabilities - reported_borrowing != residual_liabilities:
            raise ValueError(
                f"Residual liability mismatch for {entity_id}."
            )

        if assets != liabilities + equity:
            raise ValueError(
                f"Reconciliation accounting identity failed for {entity_id}."
            )

        if report["accounting_identity"] != "passed":
            raise ValueError(
                f"Accounting identity status failed for {entity_id}."
            )

        if report["data_origin"] != "synthetic":
            raise ValueError(
                f"Unexpected reconciliation data origin for {entity_id}."
            )


def defaultdict_decimal() -> dict[str, Decimal]:
    """Return a dictionary with a zero Decimal for unseen keys."""

    from collections import defaultdict

    return defaultdict(lambda: Decimal("0"))


def validate_credit_stress_results(
    output_dir: Path,
    expected_entity_ids: set[str] | None = None,
) -> int:
    """Validate saved credit-stress results against independently recomputed values."""
    entities = _read_csv(
        output_dir / "entities.csv",
        REQUIRED_COLUMNS["entities.csv"],
    )
    exposures = _read_csv(
        output_dir / "exposures.csv",
        REQUIRED_COLUMNS["exposures.csv"],
    )
    rows = _read_csv(
        output_dir / "credit_stress_results.csv",
        REQUIRED_COLUMNS["credit_stress_results.csv"],
    )

    entity_ids = _require_unique(entities, "entity_id", "entities.csv")
    result_ids = _require_unique(
        rows, "entity_id", "credit_stress_results.csv"
    )

    if expected_entity_ids is not None and result_ids != expected_entity_ids:
        raise ValueError(
            "Credit-stress result coverage does not match the entity dataset."
        )
    if result_ids != entity_ids:
        raise ValueError(
            "Credit-stress results must contain exactly one row per entity."
        )

    entity_by_id = {row["entity_id"]: row for row in entities}
    expected_count: dict[str, int] = defaultdict(int)
    expected_lending: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    expected_loss: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

    # Independently rebuild each creditor's exposures and expected credit loss.
    for exposure in exposures:
        creditor_id = exposure["creditor_id"]
        debtor_id = exposure["debtor_id"]

        if creditor_id not in entity_by_id:
            raise ValueError(
                f"Unknown creditor {creditor_id} while validating credit stress."
            )
        if debtor_id not in entity_by_id:
            raise ValueError(
                f"Unknown debtor {debtor_id} while validating credit stress."
            )

        amount = _decimal(exposure["outstanding_amount"], "outstanding_amount")
        if amount <= 0:
            raise ValueError(
                f"Exposure amount must be positive: {exposure['exposure_id']}."
            )

        debtor_type = entity_by_id[debtor_id]["entity_type"]
        assumption = get_credit_assumption(debtor_type)
        loss = (
            amount
            * assumption.probability_of_default
            * assumption.loss_given_default
        )

        expected_count[creditor_id] += 1
        expected_lending[creditor_id] += amount
        expected_loss[creditor_id] += loss

    result_by_id = {row["entity_id"]: row for row in rows}

    for entity_id, row in result_by_id.items():
        if row["data_origin"] != "synthetic":
            raise ValueError(
                f"Unexpected credit-stress data origin for {entity_id}."
            )
        if row["assumption_basis"] != "illustrative_debtor_type_pd_lgd":
            raise ValueError(f"Unexpected assumption basis for {entity_id}.")
        if row["entity_type"] != entity_by_id[entity_id]["entity_type"]:
            raise ValueError(
                f"Entity type mismatch in credit-stress results for {entity_id}."
            )

        try:
            actual_count = int(row["outgoing_exposure_count"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid exposure count for {entity_id}."
            ) from exc

        if actual_count < 0:
            raise ValueError(f"Negative exposure count for {entity_id}.")

        actual_lending = _decimal(
            row["network_lending_exposure"], "network_lending_exposure"
        )
        actual_loss = _decimal(
            row["estimated_credit_loss"], "estimated_credit_loss"
        )

        if actual_lending < 0:
            raise ValueError(f"Negative lending exposure for {entity_id}.")
        if actual_loss < 0:
            raise ValueError(f"Negative credit loss for {entity_id}.")

        if actual_count != expected_count[entity_id]:
            raise ValueError(
                f"Credit-stress exposure count mismatch for {entity_id}: "
                f"saved={actual_count}, recomputed={expected_count[entity_id]}."
            )
        if actual_lending != expected_lending[entity_id]:
            raise ValueError(
                f"Credit-stress lending mismatch for {entity_id}: "
                f"saved={actual_lending}, recomputed={expected_lending[entity_id]}."
            )
        if actual_loss != expected_loss[entity_id]:
            raise ValueError(
                f"Credit-stress loss mismatch for {entity_id}: "
                f"saved={actual_loss}, recomputed={expected_loss[entity_id]}."
            )

    return len(rows)


def validate_saved_datasets(output_dir: Path) -> dict[str, int]:
    """Validate the generated CSV datasets and return their row counts."""

    datasets = {
        filename: _read_csv(
            output_dir / filename,
            REQUIRED_COLUMNS[filename],
        )
        for filename in (
            "entities.csv",
            "instruments.csv",
            "exposures.csv",
            "financial_positions.csv",
            "entity_reconciliation.csv",
        )
    }

    entities = datasets["entities.csv"]
    instruments = datasets["instruments.csv"]
    exposures = datasets["exposures.csv"]
    positions = datasets["financial_positions.csv"]
    reconciliation = datasets["entity_reconciliation.csv"]

    _validate_relationships(entities, instruments, exposures)
    _validate_positions(entities, exposures, positions)
    _validate_reconciliation(
        entities,
        exposures,
        positions,
        reconciliation,
    )

    return {
        "entities": len(entities),
        "instruments": len(instruments),
        "exposures": len(exposures),
        "financial_positions": len(positions),
        "entity_reconciliation": len(reconciliation),
    }



def validate_contagion_results(output_dir: Path) -> int:
    """Independently validate saved contagion outcomes against source CSVs.

    Rebuilds each default wave from the saved exposures and checks losses,
    cumulative losses, equity changes, initial states, and default transitions.
    Returns the number of rows in contagion_results.csv.
    """
    entities = _read_csv(
        output_dir / "entities.csv",
        REQUIRED_COLUMNS["entities.csv"],
    )
    exposures = _read_csv(
        output_dir / "exposures.csv",
        REQUIRED_COLUMNS["exposures.csv"],
    )
    positions = _read_csv(
        output_dir / "financial_positions.csv",
        REQUIRED_COLUMNS["financial_positions.csv"],
    )
    rows = _read_csv(
        output_dir / "contagion_results.csv",
        REQUIRED_COLUMNS["contagion_results.csv"],
    )

    entity_by_id = {row["entity_id"]: row for row in entities}
    if len(entity_by_id) != len(entities):
        raise ValueError("Duplicate entity IDs in entities.csv.")

    position_by_id = {row["entity_id"]: row for row in positions}
    if set(position_by_id) != set(entity_by_id):
        raise ValueError(
            "Financial positions must contain exactly one row per entity "
            "for contagion validation."
        )
    starting_equity = {
        entity_id: _decimal(row["equity_capital"], "equity_capital")
        for entity_id, row in position_by_id.items()
    }

    if not rows:
        raise ValueError("contagion_results.csv is empty.")

    scenario_ids = {row["scenario_id"] for row in rows}
    recovery_rates = {
        _decimal(row["recovery_rate"], "recovery_rate") for row in rows
    }
    assumption_bases = {row["assumption_basis"] for row in rows}
    if len(scenario_ids) != 1 or not next(iter(scenario_ids)).strip():
        raise ValueError("Contagion rows must have one non-empty scenario_id.")
    if len(recovery_rates) != 1:
        raise ValueError("Contagion rows must use one recovery rate.")
    recovery_rate = next(iter(recovery_rates))
    if not Decimal("0") <= recovery_rate <= Decimal("1"):
        raise ValueError("Contagion recovery_rate must be between 0 and 1.")
    if assumption_bases != {"illustrative_recovery_rate"}:
        raise ValueError("Unexpected contagion assumption_basis.")

    result_by_key: dict[tuple[str, int], dict[str, str]] = {}
    rows_by_round: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        entity_id = row["entity_id"]
        if entity_id not in entity_by_id:
            raise ValueError(f"Unknown entity in contagion results: {entity_id}.")
        if row["data_origin"] != "synthetic":
            raise ValueError(f"Unexpected contagion data origin for {entity_id}.")
        try:
            round_number = int(row["round_number"])
        except ValueError as exc:
            raise ValueError(f"Invalid round_number for {entity_id}.") from exc
        if round_number < 0:
            raise ValueError("Contagion round_number cannot be negative.")
        key = (entity_id, round_number)
        if key in result_by_key:
            raise ValueError(
                f"Duplicate contagion result for entity/round: {key}."
            )
        result_by_key[key] = row
        rows_by_round[round_number][entity_id] = row

        round_loss = _decimal(row["round_loss"], "round_loss")
        cumulative_loss = _decimal(
            row["cumulative_contagion_loss"], "cumulative_contagion_loss"
        )
        equity_before = _decimal(
            row["equity_before_contagion"], "equity_before_contagion"
        )
        equity_after = _decimal(
            row["equity_after_contagion"], "equity_after_contagion"
        )
        if round_loss < 0 or cumulative_loss < 0:
            raise ValueError(f"Negative contagion loss for {entity_id}.")
        if row["newly_defaulted"] not in {"True", "False"}:
            raise ValueError(f"Invalid newly_defaulted flag for {entity_id}.")
        if equity_before - round_loss != equity_after:
            raise ValueError(
                f"Equity change does not match round loss for {entity_id}, "
                f"round {round_number}."
            )

    round_zero = rows_by_round.get(0, {})
    if set(round_zero) != set(entity_by_id):
        raise ValueError(
            "Round 0 must contain exactly one initial-state row per entity."
        )

    initial_defaults: set[str] = set()
    defaulted: set[str] = set()
    cumulative_loss_by_entity = {entity_id: Decimal("0") for entity_id in entity_by_id}
    for entity_id, row in round_zero.items():
        equity = starting_equity[entity_id]
        if (
            _decimal(row["round_loss"], "round_loss") != 0
            or _decimal(row["cumulative_contagion_loss"], "cumulative_contagion_loss") != 0
            or _decimal(row["equity_before_contagion"], "equity_before_contagion") != equity
            or _decimal(row["equity_after_contagion"], "equity_after_contagion") != equity
        ):
            raise ValueError(f"Invalid initial-state values for {entity_id}.")
        if row["status"] == "initial_default":
            if row["newly_defaulted"] != "True":
                raise ValueError(f"Initial default flag inconsistent for {entity_id}.")
            initial_defaults.add(entity_id)
            defaulted.add(entity_id)
        elif row["status"] == "active":
            if row["newly_defaulted"] != "False":
                raise ValueError(f"Active initial state flagged default for {entity_id}.")
        else:
            raise ValueError(f"Invalid round-0 status for {entity_id}: {row['status']}.")

    exposures_by_debtor: dict[str, list[dict[str, str]]] = defaultdict(list)
    exposure_ids: set[str] = set()
    for exposure in exposures:
        exposure_id = exposure["exposure_id"]
        if exposure_id in exposure_ids:
            raise ValueError(f"Duplicate exposure ID during contagion validation: {exposure_id}.")
        exposure_ids.add(exposure_id)
        creditor_id = exposure["creditor_id"]
        debtor_id = exposure["debtor_id"]
        if creditor_id not in entity_by_id or debtor_id not in entity_by_id:
            raise ValueError(f"Unknown counterparty in exposure {exposure_id}.")
        amount = _decimal(exposure["outstanding_amount"], "outstanding_amount")
        if amount <= 0:
            raise ValueError(f"Non-positive exposure amount: {exposure_id}.")
        exposures_by_debtor[debtor_id].append(exposure)

    current_wave = set(initial_defaults)
    processed_exposures: set[str] = set()
    max_round = max(rows_by_round)
    expected_rounds = set(range(max_round + 1))
    if set(rows_by_round) != expected_rounds:
        raise ValueError("Contagion round numbers must be contiguous from zero.")

    for round_number in range(1, max_round + 1):
        expected_loss_by_creditor: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        for debtor_id in current_wave:
            for exposure in exposures_by_debtor[debtor_id]:
                exposure_id = exposure["exposure_id"]
                if exposure_id in processed_exposures:
                    continue
                processed_exposures.add(exposure_id)
                amount = _decimal(exposure["outstanding_amount"], "outstanding_amount")
                expected_loss_by_creditor[exposure["creditor_id"]] += amount * (Decimal("1") - recovery_rate)

        expected_affected = {
            entity_id: loss
            for entity_id, loss in expected_loss_by_creditor.items()
            if loss > 0
        }
        actual_round = rows_by_round.get(round_number, {})
        if set(actual_round) != set(expected_affected):
            raise ValueError(
                f"Affected-entity coverage mismatch in contagion round {round_number}: "
                f"expected={sorted(expected_affected)}, actual={sorted(actual_round)}."
            )

        next_wave: set[str] = set()
        for entity_id, expected_round_loss in expected_affected.items():
            row = actual_round[entity_id]
            actual_round_loss = _decimal(row["round_loss"], "round_loss")
            if actual_round_loss != expected_round_loss:
                raise ValueError(
                    f"Contagion loss mismatch for {entity_id}, round {round_number}: "
                    f"saved={actual_round_loss}, recomputed={expected_round_loss}."
                )

            expected_cumulative = cumulative_loss_by_entity[entity_id] + expected_round_loss
            actual_cumulative = _decimal(
                row["cumulative_contagion_loss"], "cumulative_contagion_loss"
            )
            if actual_cumulative != expected_cumulative:
                raise ValueError(
                    f"Cumulative contagion loss mismatch for {entity_id}, "
                    f"round {round_number}."
                )
            expected_equity_before = starting_equity[entity_id] - cumulative_loss_by_entity[entity_id]
            expected_equity_after = starting_equity[entity_id] - expected_cumulative
            if _decimal(row["equity_before_contagion"], "equity_before_contagion") != expected_equity_before:
                raise ValueError(f"Equity-before mismatch for {entity_id}, round {round_number}.")
            if _decimal(row["equity_after_contagion"], "equity_after_contagion") != expected_equity_after:
                raise ValueError(f"Equity-after mismatch for {entity_id}, round {round_number}.")

            newly_defaulted = (
                entity_id not in defaulted
                and expected_cumulative >= starting_equity[entity_id]
            )
            if (row["newly_defaulted"] == "True") != newly_defaulted:
                raise ValueError(
                    f"Default-transition mismatch for {entity_id}, round {round_number}."
                )
            expected_status = "new_default" if newly_defaulted else (
                "already_defaulted" if entity_id in defaulted else "affected"
            )
            if row["status"] != expected_status:
                raise ValueError(
                    f"Status mismatch for {entity_id}, round {round_number}: "
                    f"saved={row['status']}, expected={expected_status}."
                )

            cumulative_loss_by_entity[entity_id] = expected_cumulative
            if newly_defaulted:
                defaulted.add(entity_id)
                next_wave.add(entity_id)

        current_wave = next_wave

    return len(rows)


def validate_scenario_contagion_results(output_dir: Path) -> int:
    """Validate saved scenario-linked contagion results against stressed equity."""
    import csv
    from collections import defaultdict
    from decimal import Decimal
    from pathlib import Path

    output_dir = Path(output_dir)
    results_path = output_dir / "scenario_contagion_results.csv"
    stress_path = output_dir / "integrated_stress_results.csv"

    expected_scenarios = {
        "SCENARIO_MILD",
        "SCENARIO_MODERATE",
        "SCENARIO_SEVERE",
        "SCENARIO_EXTREME",
    }

    if not results_path.exists():
        raise FileNotFoundError(
            f"Scenario contagion results not found: {results_path}"
        )
    if not stress_path.exists():
        raise FileNotFoundError(
            f"Integrated stress results not found: {stress_path}"
        )

    with results_path.open(newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))

    with stress_path.open(newline="", encoding="utf-8") as file:
        stress_rows = list(csv.DictReader(file))

    if not results:
        raise ValueError("Scenario contagion results are empty.")

    scenario_ids = {row["scenario_id"] for row in results}
    if scenario_ids != expected_scenarios:
        raise ValueError(
            f"Unexpected scenario IDs. Expected {sorted(expected_scenarios)}, "
            f"found {sorted(scenario_ids)}."
        )

    # Each scenario must have exactly one stressed-equity value per entity.
    stressed_equity = {}
    for row in stress_rows:
        key = (row["scenario_id"], row["entity_id"])
        if key in stressed_equity:
            raise ValueError(f"Duplicate integrated-stress entity row: {key}")
        stressed_equity[key] = Decimal(row["stressed_equity"])

    results_by_scenario = defaultdict(list)
    for row in results:
        results_by_scenario[row["scenario_id"]].append(row)

    validated_rows = 0

    for scenario_id in sorted(expected_scenarios):
        rows = results_by_scenario[scenario_id]
        entities = {row["entity_id"] for row in rows}
        round_zero = [
            row for row in rows if int(row["round_number"]) == 0
        ]

        if len(entities) != 80:
            raise ValueError(
                f"{scenario_id}: expected 80 unique entities, found "
                f"{len(entities)}."
            )

        round_zero_by_entity = {}
        for row in round_zero:
            entity_id = row["entity_id"]
            if entity_id in round_zero_by_entity:
                raise ValueError(
                    f"{scenario_id}: duplicate round-zero row for {entity_id}."
                )
            round_zero_by_entity[entity_id] = row

        if set(round_zero_by_entity) != entities:
            raise ValueError(
                f"{scenario_id}: round zero must contain every entity exactly once."
            )

        stress_entities = {
            entity_id
            for sid, entity_id in stressed_equity
            if sid == scenario_id
        }
        if stress_entities != entities:
            raise ValueError(
                f"{scenario_id}: stress and contagion entity coverage differ."
            )

        # Verify round-zero equity is the scenario's stressed starting equity.
        for entity_id, row in round_zero_by_entity.items():
            expected_equity = stressed_equity[(scenario_id, entity_id)]
            actual_equity = Decimal(row["equity_before_contagion"])
            if actual_equity != expected_equity:
                raise ValueError(
                    f"{scenario_id}/{entity_id}: round-zero equity "
                    f"{actual_equity} does not match stressed equity "
                    f"{expected_equity}."
                )

            if Decimal(row["round_loss"]) != Decimal("0"):
                raise ValueError(
                    f"{scenario_id}/{entity_id}: round-zero loss must be zero."
                )
            if Decimal(row["cumulative_contagion_loss"]) != Decimal("0"):
                raise ValueError(
                    f"{scenario_id}/{entity_id}: round-zero cumulative "
                    "contagion loss must be zero."
                )
            if Decimal(row["equity_after_contagion"]) != expected_equity:
                raise ValueError(
                    f"{scenario_id}/{entity_id}: round-zero ending equity "
                    "must equal starting stressed equity."
                )

        # Validate arithmetic and row identity in every round.
        seen = set()
        for row in rows:
            entity_id = row["entity_id"]
            round_number = int(row["round_number"])
            key = (entity_id, round_number)

            if key in seen:
                raise ValueError(
                    f"{scenario_id}: duplicate entity/round row {key}."
                )
            seen.add(key)

            if round_number < 0:
                raise ValueError(f"{scenario_id}: negative round number.")

            loss = Decimal(row["round_loss"])
            cumulative = Decimal(row["cumulative_contagion_loss"])
            before = Decimal(row["equity_before_contagion"])
            after = Decimal(row["equity_after_contagion"])

            if min(loss, cumulative) < 0:
                raise ValueError(
                    f"{scenario_id}/{entity_id}/round {round_number}: "
                    "loss values cannot be negative."
                )
            if after != before - loss:
                raise ValueError(
                    f"{scenario_id}/{entity_id}/round {round_number}: "
                    "equity arithmetic does not reconcile."
                )

        # Cumulative losses must not decrease across recorded rounds.
        for entity_id in entities:
            entity_rows = sorted(
                (row for row in rows if row["entity_id"] == entity_id),
                key=lambda row: int(row["round_number"]),
            )
            previous_cumulative = Decimal("0")
            for row in entity_rows:
                cumulative = Decimal(row["cumulative_contagion_loss"])
                if cumulative < previous_cumulative:
                    raise ValueError(
                        f"{scenario_id}/{entity_id}: cumulative contagion "
                        "loss decreased between rounds."
                    )
                previous_cumulative = cumulative

        validated_rows += len(rows)

    print(
        "Scenario contagion validation passed: "
        f"{validated_rows} rows across {len(expected_scenarios)} scenarios."
    )
    return validated_rows

