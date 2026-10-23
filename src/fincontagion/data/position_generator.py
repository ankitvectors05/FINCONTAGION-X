
import random
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP

from fincontagion.domain import EntityType, FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition


@dataclass(frozen=True, slots=True)
class PositionGenerationAssumptions:
    """
    Illustrative synthetic balance-sheet assumptions.

    These are development assumptions, not calibrated regulatory
    requirements or estimates of actual Indian institutions.
    """

    target_leverage_min: str
    target_leverage_max: str
    max_leverage: str
    cash_ratio_min: str
    cash_ratio_max: str
    short_term_share_min: str
    short_term_share_max: str
    base_assets_min_inr: int
    base_assets_max_inr: int


ASSUMPTIONS_BY_TYPE = {
    EntityType.BANK: PositionGenerationAssumptions(
        target_leverage_min="0.78",
        target_leverage_max="0.91",
        max_leverage="0.95",
        cash_ratio_min="0.04",
        cash_ratio_max="0.12",
        short_term_share_min="0.20",
        short_term_share_max="0.40",
        base_assets_min_inr=2_000_000_000,
        base_assets_max_inr=12_000_000_000,
    ),
    EntityType.NBFC: PositionGenerationAssumptions(
        target_leverage_min="0.65",
        target_leverage_max="0.85",
        max_leverage="0.92",
        cash_ratio_min="0.03",
        cash_ratio_max="0.10",
        short_term_share_min="0.25",
        short_term_share_max="0.50",
        base_assets_min_inr=500_000_000,
        base_assets_max_inr=4_000_000_000,
    ),
    EntityType.INVESTMENT_FUND: PositionGenerationAssumptions(
        target_leverage_min="0.10",
        target_leverage_max="0.40",
        max_leverage="0.70",
        cash_ratio_min="0.08",
        cash_ratio_max="0.25",
        short_term_share_min="0.10",
        short_term_share_max="0.35",
        base_assets_min_inr=300_000_000,
        base_assets_max_inr=3_000_000_000,
    ),
    EntityType.CORPORATION: PositionGenerationAssumptions(
        target_leverage_min="0.45",
        target_leverage_max="0.78",
        max_leverage="0.90",
        cash_ratio_min="0.03",
        cash_ratio_max="0.12",
        short_term_share_min="0.25",
        short_term_share_max="0.55",
        base_assets_min_inr=500_000_000,
        base_assets_max_inr=5_000_000_000,
    ),
    EntityType.SUPPLIER: PositionGenerationAssumptions(
        target_leverage_min="0.20",
        target_leverage_max="0.60",
        max_leverage="0.80",
        cash_ratio_min="0.05",
        cash_ratio_max="0.20",
        short_term_share_min="0.20",
        short_term_share_max="0.50",
        base_assets_min_inr=100_000_000,
        base_assets_max_inr=1_000_000_000,
    ),
}


RUPEE = Decimal("1")
ZERO = Decimal("0")
ONE = Decimal("1")


def _round_money(value: Decimal) -> Decimal:
    """Round a monetary amount to the nearest whole INR."""
    return value.quantize(RUPEE, rounding=ROUND_HALF_UP)


def _ceil_money(value: Decimal) -> Decimal:
    """Round a monetary amount up to the next whole INR if needed."""
    return value.quantize(RUPEE, rounding=ROUND_CEILING)


def _sample_decimal(
    rng: random.Random,
    lower: str,
    upper: str,
) -> Decimal:
    """Sample reproducibly between two decimal bounds without floats."""

    scale = 10_000
    lower_int = int(Decimal(lower) * scale)
    upper_int = int(Decimal(upper) * scale)

    return Decimal(rng.randint(lower_int, upper_int)) / Decimal(scale)


def _validate_assumptions(
    entity_type: EntityType,
    assumptions: PositionGenerationAssumptions,
) -> None:
    """Validate the configured ranges before using them."""

    target_min = Decimal(assumptions.target_leverage_min)
    target_max = Decimal(assumptions.target_leverage_max)
    max_leverage = Decimal(assumptions.max_leverage)
    cash_min = Decimal(assumptions.cash_ratio_min)
    cash_max = Decimal(assumptions.cash_ratio_max)
    short_min = Decimal(assumptions.short_term_share_min)
    short_max = Decimal(assumptions.short_term_share_max)

    if not (
        ZERO < target_min <= target_max <= max_leverage < ONE
    ):
        raise ValueError(
            f"Invalid leverage assumptions for {entity_type.value}"
        )

    if not (ZERO <= cash_min <= cash_max < ONE):
        raise ValueError(
            f"Invalid cash-ratio assumptions for {entity_type.value}"
        )

    if not (ZERO <= short_min <= short_max <= ONE):
        raise ValueError(
            f"Invalid short-term share assumptions for {entity_type.value}"
        )

    if (
        assumptions.base_assets_min_inr <= 0
        or assumptions.base_assets_max_inr
        < assumptions.base_assets_min_inr
    ):
        raise ValueError(
            f"Invalid asset-scale assumptions for {entity_type.value}"
        )


def generate_financial_positions(
    entities: list[FinancialEntity],
    exposures: list[FinancialExposure],
    seed: int = 42,
) -> list[FinancialPosition]:
    """
    Generate illustrative synthetic balance sheets.

    Current accounting structure:
        assets = network lending + cash + other assets
        liabilities = network borrowing + other liabilities
        assets = liabilities + equity

    All exposures are treated as debt-like claims. Equity exposures
    are not supported by this accounting method.

    Monetary totals are stored to whole INR so accounting identities
    hold exactly. Assumptions are not calibrated to real institutions.
    """

    if not entities:
        raise ValueError("At least one entity is required")

    if not exposures:
        raise ValueError(
            "Cannot infer the reporting date because exposures are empty"
        )

    entity_ids = [entity.entity_id for entity in entities]

    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("Duplicate entity IDs detected")

    reporting_dates = {exposure.as_of_date for exposure in exposures}

    if len(reporting_dates) != 1:
        raise ValueError(
            "All exposures must have the same reporting date"
        )

    as_of_date: date = next(iter(reporting_dates))
    rng = random.Random(seed)

    entity_by_id = {entity.entity_id: entity for entity in entities}

    lending_by_entity = {
        entity.entity_id: ZERO for entity in entities
    }
    borrowing_by_entity = {
        entity.entity_id: ZERO for entity in entities
    }

    exposure_ids: set[str] = set()

    for exposure in exposures:
        if exposure.exposure_id in exposure_ids:
            raise ValueError(
                f"Duplicate exposure ID: {exposure.exposure_id}"
            )
        exposure_ids.add(exposure.exposure_id)

        if exposure.creditor_id not in entity_by_id:
            raise ValueError(
                f"Unknown creditor: {exposure.creditor_id}"
            )

        if exposure.debtor_id not in entity_by_id:
            raise ValueError(
                f"Unknown debtor: {exposure.debtor_id}"
            )

        if exposure.creditor_id == exposure.debtor_id:
            raise ValueError(
                f"Self-exposure detected: {exposure.exposure_id}"
            )

        if (
            not isinstance(exposure.outstanding_amount, Decimal)
            or not exposure.outstanding_amount.is_finite()
            or exposure.outstanding_amount <= ZERO
        ):
            raise ValueError(
                f"Invalid exposure amount: {exposure.exposure_id}"
            )

        if not isinstance(exposure.as_of_date, date):
            raise ValueError(
                f"Invalid reporting date: {exposure.exposure_id}"
            )

        creditor = entity_by_id[exposure.creditor_id]
        debtor = entity_by_id[exposure.debtor_id]

        # Currency conversion is not implemented in this generator.
        # Reject mismatches rather than silently adding unlike currencies.
        if (
            exposure.currency != creditor.base_currency
            or exposure.currency != debtor.base_currency
        ):
            raise ValueError(
                f"Exposure currency {exposure.currency} does not match "
                f"both counterparties' base currencies for "
                f"{exposure.exposure_id}. Currency conversion is "
                "not implemented."
            )

        lending_by_entity[exposure.creditor_id] += (
            exposure.outstanding_amount
        )
        borrowing_by_entity[exposure.debtor_id] += (
            exposure.outstanding_amount
        )

    positions: list[FinancialPosition] = []

    for entity in entities:
        assumptions = ASSUMPTIONS_BY_TYPE.get(entity.entity_type)

        if assumptions is None:
            raise ValueError(
                f"Missing position assumptions for {entity.entity_type}"
            )

        _validate_assumptions(entity.entity_type, assumptions)

        network_lending = lending_by_entity[entity.entity_id]
        network_borrowing = borrowing_by_entity[entity.entity_id]

        target_leverage = _sample_decimal(
            rng,
            assumptions.target_leverage_min,
            assumptions.target_leverage_max,
        )
        max_leverage = Decimal(assumptions.max_leverage)

        cash_ratio = _sample_decimal(
            rng,
            assumptions.cash_ratio_min,
            assumptions.cash_ratio_max,
        )
        short_term_share = _sample_decimal(
            rng,
            assumptions.short_term_share_min,
            assumptions.short_term_share_max,
        )

        sampled_asset_scale = Decimal(
            rng.randint(
                assumptions.base_assets_min_inr,
                assumptions.base_assets_max_inr,
            )
        )

        # Assets must cover network lending and cash.
        asset_floor_from_lending = _ceil_money(
            network_lending / (ONE - cash_ratio)
        )

        # Assets must also support network borrowing without exceeding
        # the configured maximum leverage.
        asset_floor_from_borrowing = _ceil_money(
            network_borrowing / max_leverage
        )

        total_assets = max(
            sampled_asset_scale,
            asset_floor_from_lending,
            asset_floor_from_borrowing,
        )

        cash = _round_money(total_assets * cash_ratio)

        # Preserve the exact asset identity at whole-INR precision.
        attempts = 0

        while network_lending + cash > total_assets:
            total_assets += RUPEE
            cash = _round_money(total_assets * cash_ratio)
            attempts += 1

            if attempts > 10:
                raise ValueError(
                    f"Could not reconcile assets for {entity.entity_id}"
                )

        other_assets = total_assets - network_lending - cash

        target_liabilities = _round_money(
            total_assets * target_leverage
        )

        total_liabilities = max(
            network_borrowing,
            target_liabilities,
        )

        # Explicitly enforce the maximum leverage assumption.
        # If network borrowing alone breaches this limit, do not
        # silently produce a balance sheet that violates the model.
        if total_liabilities > _round_money(
            total_assets * max_leverage
        ):
            raise ValueError(
                f"Maximum leverage exceeded for {entity.entity_id}: "
                f"liabilities={total_liabilities}, "
                f"assets={total_assets}, "
                f"maximum leverage={max_leverage}. "
                "Review the synthetic network or asset assumptions."
            )

        other_liabilities = total_liabilities - network_borrowing
        equity = total_assets - total_liabilities

        if other_assets < ZERO:
            raise ValueError(
                f"Negative residual assets for {entity.entity_id}"
            )

        if other_liabilities < ZERO:
            raise ValueError(
                f"Negative residual liabilities for {entity.entity_id}"
            )

        if equity <= ZERO:
            raise ValueError(
                f"Non-positive equity generated for {entity.entity_id}: "
                f"assets={total_assets}, liabilities={total_liabilities}"
            )

        short_term_liabilities = _round_money(
            total_liabilities * short_term_share
        )

        # Verify the identities before constructing the domain object.
        if total_assets != network_lending + cash + other_assets:
            raise ValueError(
                f"Asset identity failed for {entity.entity_id}"
            )

        if total_liabilities != (
            network_borrowing + other_liabilities
        ):
            raise ValueError(
                f"Liability identity failed for {entity.entity_id}"
            )

        if total_assets != total_liabilities + equity:
            raise ValueError(
                f"Accounting identity failed for {entity.entity_id}"
            )

        if not (
            ZERO <= short_term_liabilities <= total_liabilities
        ):
            raise ValueError(
                f"Invalid short-term liabilities for {entity.entity_id}"
            )

        positions.append(
            FinancialPosition(
                entity_id=entity.entity_id,
                as_of_date=as_of_date,
                cash_and_liquid_assets=cash,
                total_assets=total_assets,
                total_liabilities=total_liabilities,
                equity_capital=equity,
                short_term_liabilities=short_term_liabilities,
            )
        )

    return positions
