
import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition


def build_reconciliation_report(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    positions: list[FinancialPosition],
) -> list[dict[str, str]]:
    """Reconcile network lending and borrowing with entity balance sheets."""

    entity_ids = {entity.entity_id for entity in entities}
    position_by_entity = {position.entity_id: position for position in positions}

    if len(position_by_entity) != len(positions):
        raise ValueError("Duplicate entity IDs found in financial positions.")

    if set(position_by_entity) != entity_ids:
        raise ValueError(
            "Financial positions must contain exactly one row per entity."
        )

    lending = defaultdict(lambda: Decimal("0"))
    borrowing = defaultdict(lambda: Decimal("0"))

    for exposure in exposures:
        if exposure.creditor_id not in entity_ids:
            raise ValueError(f"Unknown creditor: {exposure.creditor_id}")

        if exposure.debtor_id not in entity_ids:
            raise ValueError(f"Unknown debtor: {exposure.debtor_id}")

        lending[exposure.creditor_id] += exposure.outstanding_amount
        borrowing[exposure.debtor_id] += exposure.outstanding_amount

    report: list[dict[str, str]] = []

    for entity in entities:
        position = position_by_entity[entity.entity_id]

        network_lending = lending[entity.entity_id]
        network_borrowing = borrowing[entity.entity_id]

        residual_assets = position.total_assets - network_lending
        residual_liabilities = position.total_liabilities - network_borrowing

        if residual_assets < 0:
            raise ValueError(
                f"{entity.entity_id}: network lending exceeds total assets."
            )

        if residual_liabilities < 0:
            raise ValueError(
                f"{entity.entity_id}: network borrowing exceeds total liabilities."
            )

        if position.total_assets != (
            position.total_liabilities + position.equity_capital
        ):
            raise ValueError(
                f"{entity.entity_id}: accounting identity failed."
            )

        report.append(
            {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "network_lending_assets": str(network_lending),
                "network_borrowing_liabilities": str(network_borrowing),
                "residual_assets": str(residual_assets),
                "residual_liabilities": str(residual_liabilities),
                "cash_and_liquid_assets": str(
                    position.cash_and_liquid_assets
                ),
                "total_assets": str(position.total_assets),
                "total_liabilities": str(position.total_liabilities),
                "equity_capital": str(position.equity_capital),
                "accounting_identity": "passed",
                "data_origin": "synthetic",
            }
        )

    total_lending = sum(lending.values(), Decimal("0"))
    total_borrowing = sum(borrowing.values(), Decimal("0"))

    if total_lending != total_borrowing:
        raise ValueError(
            "Network reconciliation failed: total lending does not "
            "equal total borrowing."
        )

    return report


def export_reconciliation_report(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Save the reconciliation report as a CSV file."""

    if not rows:
        raise ValueError("Cannot export an empty reconciliation report.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
