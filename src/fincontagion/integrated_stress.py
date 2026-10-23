
"""Integrated entity-level financial stress calculations.

All assumptions are illustrative and intended for synthetic scenarios.

Interest-rate valuation changes use a simplified duration approximation.
The duration coefficients are modelling proxies, not market-calibrated
estimates. Floating-rate instruments are excluded because repricing
frequency and basis risk are not currently modelled.
"""

from datetime import date
from decimal import Decimal

from fincontagion.domain import FinancialEntity
from fincontagion.instruments import (
    FinancialExposure,
    FinancialInstrument,
    InstrumentType,
    RateType,
)
from fincontagion.positions import FinancialPosition
from fincontagion.scenarios import StressScenario
from fincontagion.stress_assumptions import get_credit_assumption


ZERO = Decimal("0")
ONE = Decimal("1")
DAYS_PER_YEAR = Decimal("365.25")

# Illustrative duration proxies as fractions of remaining maturity.
# These are synthetic modelling assumptions, not observed durations.
DURATION_FACTORS = {
    InstrumentType.BOND: Decimal("0.70"),
    InstrumentType.LOAN: Decimal("0.50"),
    InstrumentType.INTERBANK_FUNDING: Decimal("0.25"),
}


def _duration_proxy(
    instrument: FinancialInstrument,
    as_of_date: date,
) -> Decimal:
    """Estimate modified-duration proxy for a fixed-rate instrument."""
    if instrument.rate_type != RateType.FIXED:
        return ZERO

    factor = DURATION_FACTORS.get(instrument.instrument_type, ZERO)

    if instrument.maturity_date is None or factor == ZERO:
        return ZERO

    remaining_days = max(
        0,
        (instrument.maturity_date - as_of_date).days,
    )
    remaining_years = Decimal(remaining_days) / DAYS_PER_YEAR

    return remaining_years * factor


def calculate_integrated_stress(
    entities: list[FinancialEntity],
    positions: list[FinancialPosition],
    exposures: list[FinancialExposure],
    instruments: list[FinancialInstrument],
    scenario: StressScenario,
) -> list[dict]:
    """Return one integrated stress result per entity.

    Modelling choices:
    - Expected credit loss = exposure * debtor PD * debtor LGD.
    - Interest-rate valuation change applies to fixed-rate network exposures
      using a simplified duration approximation.
    - Market loss applies to residual assets, excluding cash and network
      lending. This is a synthetic proxy, not observed holdings.
    - Funding withdrawals are liquidity events, not automatically losses.
    - Net accounting impact = credit loss + market loss + rate valuation loss
      - rate valuation gain.
    - Total accounting loss is floored at zero; a net gain can still increase
      stressed equity through the signed net accounting impact.
    """
    if not entities:
        raise ValueError("At least one entity is required.")

    entity_by_id = {entity.entity_id: entity for entity in entities}
    position_by_id = {position.entity_id: position for position in positions}
    instrument_by_id = {
        instrument.instrument_id: instrument for instrument in instruments
    }

    if len(entity_by_id) != len(entities):
        raise ValueError("Duplicate entity IDs detected.")

    if len(position_by_id) != len(positions):
        raise ValueError("Duplicate financial-position entity IDs detected.")

    if len(instrument_by_id) != len(instruments):
        raise ValueError("Duplicate instrument IDs detected.")

    if set(entity_by_id) != set(position_by_id):
        raise ValueError(
            "Entities and financial positions must contain the same entity IDs."
        )

    position_dates = {position.as_of_date for position in positions}
    if len(position_dates) != 1:
        raise ValueError(
            "All financial positions must share exactly one reporting date."
        )

    reporting_date = next(iter(position_dates))

    lending = {entity_id: ZERO for entity_id in entity_by_id}
    credit_losses = {entity_id: ZERO for entity_id in entity_by_id}
    rate_valuation_changes = {entity_id: ZERO for entity_id in entity_by_id}
    exposure_counts = {entity_id: 0 for entity_id in entity_by_id}

    for exposure in exposures:
        if exposure.creditor_id not in entity_by_id:
            raise ValueError(f"Unknown creditor: {exposure.creditor_id}")

        if exposure.debtor_id not in entity_by_id:
            raise ValueError(f"Unknown debtor: {exposure.debtor_id}")

        if exposure.creditor_id == exposure.debtor_id:
            raise ValueError(f"Self-exposure detected: {exposure.exposure_id}")

        if exposure.as_of_date != reporting_date:
            raise ValueError(
                f"Exposure {exposure.exposure_id} has a reporting date "
                "that differs from the financial positions."
            )

        amount = exposure.outstanding_amount
        if not isinstance(amount, Decimal) or not amount.is_finite():
            raise ValueError(f"Invalid exposure amount: {exposure.exposure_id}")

        if amount <= ZERO:
            raise ValueError(
                f"Exposure amount must be positive: {exposure.exposure_id}"
            )

        instrument = instrument_by_id.get(exposure.instrument_id)
        if instrument is None:
            raise ValueError(
                f"Unknown instrument: {exposure.instrument_id}"
            )

        if instrument.issuer_id != exposure.debtor_id:
            raise ValueError(
                f"Instrument issuer does not match exposure debtor: "
                f"{exposure.exposure_id}"
            )

        if instrument.currency != exposure.currency:
            raise ValueError(
                f"Instrument currency does not match exposure currency: "
                f"{exposure.exposure_id}"
            )

        creditor_id = exposure.creditor_id
        debtor = entity_by_id[exposure.debtor_id]
        assumption = get_credit_assumption(debtor.entity_type)

        lending[creditor_id] += amount
        exposure_counts[creditor_id] += 1
        credit_losses[creditor_id] += (
            amount
            * assumption.probability_of_default
            * assumption.loss_given_default
        )

        # A positive rate shock generally reduces the value of a fixed-rate
        # asset; a negative shock generally increases it.
        duration = _duration_proxy(instrument, reporting_date)

        if duration > ZERO:
            estimated_return = -duration * scenario.interest_rate_shock

            # Bound the linear approximation to a maximum 100% gain or loss.
            estimated_return = max(
                Decimal("-1"),
                min(ONE, estimated_return),
            )

            rate_valuation_changes[creditor_id] += amount * estimated_return

    results = []

    for entity in entities:
        position = position_by_id[entity.entity_id]
        entity_id = entity.entity_id

        if (
            position.total_assets
            < lending[entity_id] + position.cash_and_liquid_assets
        ):
            raise ValueError(
                "Network lending plus cash exceeds total assets for "
                f"{entity_id}."
            )

        other_assets = (
            position.total_assets
            - lending[entity_id]
            - position.cash_and_liquid_assets
        )

        market_loss = other_assets * scenario.asset_price_decline

        liquidity_outflow = (
            position.short_term_liabilities
            * scenario.funding_withdrawal_rate
        )

        # Negative valuation change = loss; positive change = gain.
        rate_change = rate_valuation_changes[entity_id]
        interest_rate_loss = max(ZERO, -rate_change)
        interest_rate_gain = max(ZERO, rate_change)

        net_accounting_impact = (
            credit_losses[entity_id]
            + market_loss
            - rate_change
        )

        total_accounting_loss = max(ZERO, net_accounting_impact)
        stressed_equity = position.equity_capital - net_accounting_impact

        cash_after_withdrawal = max(
            ZERO,
            position.cash_and_liquid_assets - liquidity_outflow,
        )
        liquidity_shortfall = max(
            ZERO,
            liquidity_outflow - position.cash_and_liquid_assets,
        )

        results.append(
            {
                "entity_id": entity_id,
                "entity_type": entity.entity_type.value,
                "as_of_date": position.as_of_date.isoformat(),
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "interest_rate_shock": scenario.interest_rate_shock,
                "asset_price_decline": scenario.asset_price_decline,
                "funding_withdrawal_rate": scenario.funding_withdrawal_rate,
                "credit_loss": credit_losses[entity_id],
                "market_loss": market_loss,
                "interest_rate_loss": interest_rate_loss,
                "interest_rate_gain": interest_rate_gain,
                "interest_rate_valuation_change": rate_change,
                "net_accounting_impact": net_accounting_impact,
                "total_accounting_loss": total_accounting_loss,
                "baseline_equity": position.equity_capital,
                "stressed_equity": stressed_equity,
                "cash_before_stress": position.cash_and_liquid_assets,
                "funding_withdrawal": liquidity_outflow,
                "cash_after_withdrawal": cash_after_withdrawal,
                "liquidity_shortfall": liquidity_shortfall,
                "network_lending_exposure": lending[entity_id],
                "outgoing_exposure_count": exposure_counts[entity_id],
                "interest_rate_loss_included": True,
                "interest_rate_assumption_basis": (
                    "synthetic_duration_proxy_for_fixed_rate_instruments"
                ),
                "assumption_basis": "illustrative_synthetic_assumptions",
            }
        )

    return results
