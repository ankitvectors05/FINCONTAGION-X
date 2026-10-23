
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum


class InstrumentType(str, Enum):
    LOAN = "loan"
    BOND = "bond"
    EQUITY = "equity"
    INTERBANK_FUNDING = "interbank_funding"
    TRADE_CREDIT = "trade_credit"


class RateType(str, Enum):
    FIXED = "fixed"
    FLOATING = "floating"
    NOT_APPLICABLE = "not_applicable"


RATE_BEARING_INSTRUMENTS = frozenset(
    {
        InstrumentType.LOAN,
        InstrumentType.BOND,
        InstrumentType.INTERBANK_FUNDING,
    }
)

NON_RATE_BEARING_INSTRUMENTS = frozenset(
    {
        InstrumentType.EQUITY,
        InstrumentType.TRADE_CREDIT,
    }
)


@dataclass(frozen=True, slots=True)
class FinancialInstrument:
    instrument_id: str
    instrument_type: InstrumentType
    issuer_id: str
    currency: str
    issue_date: date
    maturity_date: date | None = None
    rate_type: RateType = RateType.NOT_APPLICABLE
    annual_interest_rate: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise ValueError("instrument_id cannot be empty")

        if not isinstance(self.issuer_id, str) or not self.issuer_id.strip():
            raise ValueError("issuer_id cannot be empty")

        if not isinstance(self.instrument_type, InstrumentType):
            raise ValueError("instrument_type must be a valid InstrumentType")

        if not isinstance(self.rate_type, RateType):
            raise ValueError("rate_type must be a valid RateType")

        if not isinstance(self.issue_date, date):
            raise ValueError("issue_date must be a date")

        if (
            self.maturity_date is not None
            and not isinstance(self.maturity_date, date)
        ):
            raise ValueError("maturity_date must be a date or None")

        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isalpha()
            or not self.currency.isupper()
        ):
            raise ValueError(
                "currency must be a three-letter uppercase code"
            )

        if (
            self.maturity_date is not None
            and self.maturity_date < self.issue_date
        ):
            raise ValueError(
                "maturity_date cannot precede issue_date"
            )

        if self.annual_interest_rate is not None:
            if not isinstance(self.annual_interest_rate, Decimal):
                raise ValueError(
                    "annual_interest_rate must be a Decimal or None"
                )

            if (
                not self.annual_interest_rate.is_finite()
                or self.annual_interest_rate < Decimal("0")
            ):
                raise ValueError(
                    "annual_interest_rate must be finite and non-negative"
                )

        if self.instrument_type in RATE_BEARING_INSTRUMENTS:
            if self.rate_type not in (
                RateType.FIXED,
                RateType.FLOATING,
            ):
                raise ValueError(
                    f"{self.instrument_type.value} requires a fixed "
                    "or floating rate type"
                )

            if self.annual_interest_rate is None:
                raise ValueError(
                    f"{self.instrument_type.value} requires "
                    "an annual interest rate"
                )

        elif self.instrument_type in NON_RATE_BEARING_INSTRUMENTS:
            if self.rate_type != RateType.NOT_APPLICABLE:
                raise ValueError(
                    f"{self.instrument_type.value} must use "
                    "RateType.NOT_APPLICABLE in this model"
                )

            if self.annual_interest_rate is not None:
                raise ValueError(
                    f"{self.instrument_type.value} cannot have "
                    "an annual interest rate in this model"
                )


@dataclass(frozen=True, slots=True)
class FinancialExposure:
    exposure_id: str
    instrument_id: str
    creditor_id: str
    debtor_id: str
    outstanding_amount: Decimal
    currency: str
    as_of_date: date

    def __post_init__(self) -> None:
        for field_name in (
            "exposure_id",
            "instrument_id",
            "creditor_id",
            "debtor_id",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")

        if self.creditor_id == self.debtor_id:
            raise ValueError(
                "Creditor and debtor must be different entities"
            )

        if not isinstance(self.outstanding_amount, Decimal):
            raise ValueError(
                "outstanding_amount must be a Decimal"
            )

        if (
            not self.outstanding_amount.is_finite()
            or self.outstanding_amount <= Decimal("0")
        ):
            raise ValueError(
                "outstanding_amount must be finite and positive"
            )

        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isalpha()
            or not self.currency.isupper()
        ):
            raise ValueError(
                "currency must be a three-letter uppercase code"
            )

        if not isinstance(self.as_of_date, date):
            raise ValueError("as_of_date must be a date")
