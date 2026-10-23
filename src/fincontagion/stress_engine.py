
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from fincontagion.domain import FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.scenarios import StressScenario
from fincontagion.stress_assumptions import get_credit_assumption


@dataclass(frozen=True, slots=True)
class EntityStressResult:
    """Direct stress estimates for one financial entity."""

    entity_id: str
    credit_loss: Decimal
    market_loss: Decimal
    liquidity_outflow: Decimal

    @property
    def total_direct_accounting_loss(self) -> Decimal:
        """Accounting losses exclude liquidity outflows."""
        return self.credit_loss + self.market_loss


def calculate_credit_loss(
    exposure_amount: Decimal,
    probability_of_default: Decimal,
    loss_given_default: Decimal,
) -> Decimal:
    """
    Calculate illustrative expected credit loss:
    ECL = EAD × PD × LGD.
    """

    for name, value in (
        ("exposure_amount", exposure_amount),
        ("probability_of_default", probability_of_default),
        ("loss_given_default", loss_given_default),
    ):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite.")

    if exposure_amount < 0:
        raise ValueError("Exposure amount cannot be negative.")

    if not Decimal("0") <= probability_of_default <= Decimal("1"):
        raise ValueError("Probability of default must be between 0 and 1.")

    if not Decimal("0") <= loss_given_default <= Decimal("1"):
        raise ValueError("Loss given default must be between 0 and 1.")

    return exposure_amount * probability_of_default * loss_given_default


def calculate_market_loss(
    eligible_asset_value: Decimal,
    scenario: StressScenario,
) -> Decimal:
    """Estimate a price decline on explicitly eligible assets only."""

    if not eligible_asset_value.is_finite():
        raise ValueError("Eligible asset value must be finite.")

    if eligible_asset_value < 0:
        raise ValueError("Eligible asset value cannot be negative.")

    return eligible_asset_value * scenario.asset_price_decline


def calculate_liquidity_outflow(
    eligible_funding_base: Decimal,
    scenario: StressScenario,
) -> Decimal:
    """Estimate cash withdrawals; this is not automatically an accounting loss."""

    if not eligible_funding_base.is_finite():
        raise ValueError("Eligible funding base must be finite.")

    if eligible_funding_base < 0:
        raise ValueError("Eligible funding base cannot be negative.")

    return eligible_funding_base * scenario.funding_withdrawal_rate


def calculate_fixed_rate_bond_market_loss(
    market_value: Decimal,
    modified_duration: Decimal,
    interest_rate_shock: Decimal,
) -> Decimal:
    """
    First-order fixed-rate bond valuation approximation.

    Estimated loss ≈ market value × modified duration × change in yield.

    A positive result represents a loss; a negative result represents
    an estimated gain when yields fall. Convexity and spread changes
    are not included.
    """

    for name, value in (
        ("market_value", market_value),
        ("modified_duration", modified_duration),
        ("interest_rate_shock", interest_rate_shock),
    ):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite.")

    if market_value < 0:
        raise ValueError("Market value cannot be negative.")

    if modified_duration < 0:
        raise ValueError("Modified duration cannot be negative.")

    return market_value * modified_duration * interest_rate_shock


def calculate_network_credit_stress(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
) -> list[dict[str, str]]:
    """
    Estimate credit losses across the financial exposure network.

    For each exposure:
    1. Identify the creditor and debtor.
    2. Select illustrative PD/LGD assumptions using the debtor's type.
    3. Calculate EAD × PD × LGD.
    4. Attribute the estimated loss to the creditor.

    Results are aggregated by creditor. These are simplified,
    assumption-driven estimates, not calibrated forecasts.
    """

    entity_by_id = {entity.entity_id: entity for entity in entities}

    if len(entity_by_id) != len(entities):
        raise ValueError("Duplicate entity IDs found.")

    exposure_by_creditor: dict[str, Decimal] = defaultdict(
        lambda: Decimal("0")
    )
    credit_loss_by_creditor: dict[str, Decimal] = defaultdict(
        lambda: Decimal("0")
    )
    exposure_count_by_creditor: dict[str, int] = defaultdict(int)

    for exposure in exposures:
        creditor = entity_by_id.get(exposure.creditor_id)
        debtor = entity_by_id.get(exposure.debtor_id)

        if creditor is None:
            raise ValueError(f"Unknown creditor: {exposure.creditor_id}")

        if debtor is None:
            raise ValueError(f"Unknown debtor: {exposure.debtor_id}")

        assumption = get_credit_assumption(debtor.entity_type.value)

        estimated_loss = calculate_credit_loss(
            exposure_amount=exposure.outstanding_amount,
            probability_of_default=assumption.probability_of_default,
            loss_given_default=assumption.loss_given_default,
        )

        exposure_by_creditor[creditor.entity_id] += (
            exposure.outstanding_amount
        )
        credit_loss_by_creditor[creditor.entity_id] += estimated_loss
        exposure_count_by_creditor[creditor.entity_id] += 1

    results: list[dict[str, str]] = []

    for entity in entities:
        results.append(
            {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type.value,
                "outgoing_exposure_count": str(
                    exposure_count_by_creditor[entity.entity_id]
                ),
                "network_lending_exposure": str(
                    exposure_by_creditor[entity.entity_id]
                ),
                "estimated_credit_loss": str(
                    credit_loss_by_creditor[entity.entity_id]
                ),
                "data_origin": "synthetic",
                "assumption_basis": "illustrative_debtor_type_pd_lgd",
            }
        )

    return results
