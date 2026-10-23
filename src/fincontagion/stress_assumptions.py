
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class CreditStressAssumption:
    """Illustrative credit-risk assumptions for a borrower category."""

    probability_of_default: Decimal
    loss_given_default: Decimal

    def __post_init__(self) -> None:
        if not self.probability_of_default.is_finite():
            raise ValueError("Probability of default must be finite.")

        if not self.loss_given_default.is_finite():
            raise ValueError("Loss given default must be finite.")

        if not Decimal("0") <= self.probability_of_default <= Decimal("1"):
            raise ValueError("Probability of default must be between 0 and 1.")

        if not Decimal("0") <= self.loss_given_default <= Decimal("1"):
            raise ValueError("Loss given default must be between 0 and 1.")


# Illustrative scenario assumptions, not calibrated market estimates.
CREDIT_STRESS_BY_ENTITY_TYPE = {
    "bank": CreditStressAssumption(
        probability_of_default=Decimal("0.02"),
        loss_given_default=Decimal("0.45"),
    ),
    "nbfc": CreditStressAssumption(
        probability_of_default=Decimal("0.05"),
        loss_given_default=Decimal("0.55"),
    ),
    "investment_fund": CreditStressAssumption(
        probability_of_default=Decimal("0.03"),
        loss_given_default=Decimal("0.40"),
    ),
    "corporation": CreditStressAssumption(
        probability_of_default=Decimal("0.06"),
        loss_given_default=Decimal("0.50"),
    ),
    "supplier": CreditStressAssumption(
        probability_of_default=Decimal("0.08"),
        loss_given_default=Decimal("0.60"),
    ),
}


def get_credit_assumption(entity_type: str) -> CreditStressAssumption:
    """Return the illustrative credit assumption for an entity type."""

    try:
        return CREDIT_STRESS_BY_ENTITY_TYPE[entity_type]
    except KeyError as exc:
        raise ValueError(
            f"No credit stress assumption defined for {entity_type!r}."
        ) from exc
