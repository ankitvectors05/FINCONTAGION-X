
from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class FinancialPosition:
    """A synthetic entity's financial position on a specific date."""

    entity_id: str
    as_of_date: date
    cash_and_liquid_assets: Decimal
    total_assets: Decimal
    total_liabilities: Decimal
    equity_capital: Decimal
    short_term_liabilities: Decimal

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty")

        monetary_fields = (
            "cash_and_liquid_assets",
            "total_assets",
            "total_liabilities",
            "equity_capital",
            "short_term_liabilities",
        )

        for field_name in monetary_fields:
            amount = getattr(self, field_name)

            if not amount.is_finite():
                raise ValueError(f"{field_name} must be finite")

            if amount < Decimal("0"):
                raise ValueError(f"{field_name} cannot be negative")

        if self.cash_and_liquid_assets > self.total_assets:
            raise ValueError(
                "Cash and liquid assets cannot exceed total assets"
            )

        if self.short_term_liabilities > self.total_liabilities:
            raise ValueError(
                "Short-term liabilities cannot exceed total liabilities"
            )

        if self.total_assets != (
            self.total_liabilities + self.equity_capital
        ):
            raise ValueError(
                "Accounting identity violated: "
                "assets must equal liabilities plus equity"
            )
