
import csv
from pathlib import Path

from fincontagion.positions import FinancialPosition
from fincontagion.domain import FinancialEntity
from fincontagion.instruments import (
    FinancialExposure,
    FinancialInstrument,
)


def export_entities(
    entities: list[FinancialEntity],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "entity_id",
                "name",
                "entity_type",
                "country",
                "base_currency",
                "data_origin",
            ],
        )
        writer.writeheader()

        for entity in entities:
            writer.writerow(
                {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "country": entity.country,
                    "base_currency": entity.base_currency,
                    "data_origin": "synthetic",
                }
            )


def export_instruments(
    instruments: list[FinancialInstrument],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "instrument_id",
                "instrument_type",
                "issuer_id",
                "currency",
                "issue_date",
                "maturity_date",
                "rate_type",
                "annual_interest_rate",
                "data_origin",
            ],
        )
        writer.writeheader()

        for instrument in instruments:
            writer.writerow(
                {
                    "instrument_id": instrument.instrument_id,
                    "instrument_type": instrument.instrument_type.value,
                    "issuer_id": instrument.issuer_id,
                    "currency": instrument.currency,
                    "issue_date": instrument.issue_date.isoformat(),
                    "maturity_date": (
                        instrument.maturity_date.isoformat()
                        if instrument.maturity_date is not None
                        else ""
                    ),
                    "rate_type": instrument.rate_type.value,
                    "annual_interest_rate": (
                        str(instrument.annual_interest_rate)
                        if instrument.annual_interest_rate is not None
                        else ""
                    ),
                    "data_origin": "synthetic",
                }
            )


def export_exposures(
    exposures: list[FinancialExposure],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "exposure_id",
                "instrument_id",
                "creditor_id",
                "debtor_id",
                "outstanding_amount",
                "currency",
                "as_of_date",
                "data_origin",
            ],
        )
        writer.writeheader()

        for exposure in exposures:
            writer.writerow(
                {
                    "exposure_id": exposure.exposure_id,
                    "instrument_id": exposure.instrument_id,
                    "creditor_id": exposure.creditor_id,
                    "debtor_id": exposure.debtor_id,
                    "outstanding_amount": str(
                        exposure.outstanding_amount
                    ),
                    "currency": exposure.currency,
                    "as_of_date": exposure.as_of_date.isoformat(),
                    "data_origin": "synthetic",
                }
            )



def export_financial_positions(
    positions: list[FinancialPosition],
    output_path: Path,
) -> None:
    """Export synthetic financial positions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "entity_id",
                "as_of_date",
                "cash_and_liquid_assets",
                "total_assets",
                "total_liabilities",
                "equity_capital",
                "short_term_liabilities",
                "data_origin",
            ],
        )
        writer.writeheader()

        for position in positions:
            writer.writerow(
                {
                    "entity_id": position.entity_id,
                    "as_of_date": position.as_of_date.isoformat(),
                    "cash_and_liquid_assets": str(
                        position.cash_and_liquid_assets
                    ),
                    "total_assets": str(position.total_assets),
                    "total_liabilities": str(
                        position.total_liabilities
                    ),
                    "equity_capital": str(position.equity_capital),
                    "short_term_liabilities": str(
                        position.short_term_liabilities
                    ),
                    "data_origin": "synthetic",
                }
            )
