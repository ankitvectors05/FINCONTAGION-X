
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import random

from fincontagion.domain import EntityType, FinancialEntity
from fincontagion.instruments import (
    FinancialExposure,
    FinancialInstrument,
    InstrumentType,
    RateType,
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    seed: int = 42
    as_of_date: date = date(2026, 1, 1)
    bank_count: int = 10
    nbfc_count: int = 8
    fund_count: int = 8
    corporation_count: int = 24
    supplier_count: int = 30

    def __post_init__(self) -> None:
        counts = (
            self.bank_count,
            self.nbfc_count,
            self.fund_count,
            self.corporation_count,
            self.supplier_count,
        )

        if any(count < 0 for count in counts):
            raise ValueError("Entity counts cannot be negative")

        if sum(counts) == 0:
            raise ValueError("At least one entity is required")


INDIAN_CITIES = (
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Pune",
    "Ahmedabad",
    "Kolkata",
    "Jaipur",
    "Lucknow",
)

ENTITY_PREFIXES = {
    EntityType.BANK: "BANK",
    EntityType.NBFC: "NBFC",
    EntityType.INVESTMENT_FUND: "FUND",
    EntityType.CORPORATION: "CORP",
    EntityType.SUPPLIER: "SUP",
}


def generate_entities(
    config: DatasetConfig,
) -> list[FinancialEntity]:
    """Generate reproducible synthetic Indian financial entities."""

    rng = random.Random(config.seed)
    entities: list[FinancialEntity] = []

    specifications = (
        (EntityType.BANK, config.bank_count, "Bank"),
        (EntityType.NBFC, config.nbfc_count, "NBFC"),
        (EntityType.INVESTMENT_FUND, config.fund_count, "Investment Fund"),
        (EntityType.CORPORATION, config.corporation_count, "Corporation"),
        (EntityType.SUPPLIER, config.supplier_count, "Supplier"),
    )

    for entity_type, count, label in specifications:
        prefix = ENTITY_PREFIXES[entity_type]

        for number in range(1, count + 1):
            city = rng.choice(INDIAN_CITIES)

            entities.append(
                FinancialEntity(
                    entity_id=f"{prefix}_{number:04d}",
                    name=f"{city} {label} {number:04d}",
                    entity_type=entity_type,
                    country="India",
                    base_currency="INR",
                )
            )

    return entities


def generate_instruments_and_exposures(
    config: DatasetConfig,
    entities: list[FinancialEntity],
) -> tuple[list[FinancialInstrument], list[FinancialExposure]]:
    """
    Generate a reproducible synthetic financial network.

    Relationship directions:
    - Loan/interbank funding/bond: creditor holds a claim on debtor.
    - Trade credit: supplier or corporation is creditor; buyer is debtor.

    The network is illustrative and is not calibrated to observed
    institution-level Indian financial relationships.
    """

    rng = random.Random(config.seed + 1)

    by_type = {
        entity_type: [
            entity
            for entity in entities
            if entity.entity_type == entity_type
        ]
        for entity_type in EntityType
    }

    banks = by_type[EntityType.BANK]
    nbfcs = by_type[EntityType.NBFC]
    funds = by_type[EntityType.INVESTMENT_FUND]
    corporations = by_type[EntityType.CORPORATION]
    suppliers = by_type[EntityType.SUPPLIER]

    lenders = banks + nbfcs
    bond_issuers = banks + nbfcs + corporations
    bond_holders = banks + funds

    instruments: list[FinancialInstrument] = []
    exposures: list[FinancialExposure] = []

    relationship_keys: set[tuple[InstrumentType, str, str]] = set()

    def add_relationship(
        instrument_type: InstrumentType,
        creditor: FinancialEntity,
        debtor: FinancialEntity,
    ) -> bool:
        """Add one valid relationship; return False for duplicates."""

        if creditor.entity_id == debtor.entity_id:
            raise ValueError("Self-exposures are not permitted")

        key = (
            instrument_type,
            creditor.entity_id,
            debtor.entity_id,
        )

        if key in relationship_keys:
            return False

        number = len(instruments) + 1
        instrument_id = f"INST_{number:06d}"
        exposure_id = f"EXP_{number:06d}"

        issue_date = config.as_of_date - timedelta(
            days=rng.randint(30, 3650)
        )

        maturity_date = config.as_of_date + timedelta(
            days=rng.randint(30, 3650)
        )

        if instrument_type in (
            InstrumentType.LOAN,
            InstrumentType.BOND,
            InstrumentType.INTERBANK_FUNDING,
        ):
            if instrument_type == InstrumentType.BOND:
                rate_type = RateType.FIXED
            else:
                rate_type = rng.choice(
                    [RateType.FIXED, RateType.FLOATING]
                )

            interest_rate = Decimal(
                str(
                    rng.choice(
                        [0.065, 0.07, 0.075, 0.08, 0.09, 0.10]
                    )
                )
            )
        else:
            rate_type = RateType.NOT_APPLICABLE
            interest_rate = None
            maturity_date = None

        instrument = FinancialInstrument(
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            issuer_id=debtor.entity_id,
            currency="INR",
            issue_date=issue_date,
            maturity_date=maturity_date,
            rate_type=rate_type,
            annual_interest_rate=interest_rate,
        )

        amount = Decimal(
            rng.randrange(5_000_000, 500_000_001, 1_000_000)
        )

        exposure = FinancialExposure(
            exposure_id=exposure_id,
            instrument_id=instrument_id,
            creditor_id=creditor.entity_id,
            debtor_id=debtor.entity_id,
            outstanding_amount=amount,
            currency="INR",
            as_of_date=config.as_of_date,
        )

        instruments.append(instrument)
        exposures.append(exposure)
        relationship_keys.add(key)

        return True

    def add_random_relationships(
        instrument_type: InstrumentType,
        creditor: FinancialEntity,
        eligible_debtors: list[FinancialEntity],
        minimum: int,
        maximum: int,
    ) -> None:
        """Choose distinct eligible counterparties for one creditor."""

        eligible = [
            debtor
            for debtor in eligible_debtors
            if debtor.entity_id != creditor.entity_id
            and (
                instrument_type,
                creditor.entity_id,
                debtor.entity_id,
            ) not in relationship_keys
        ]

        if not eligible:
            return

        desired_count = min(
            rng.randint(minimum, maximum),
            len(eligible),
        )

        for debtor in rng.sample(eligible, desired_count):
            add_relationship(instrument_type, creditor, debtor)

    # 1. Bank/NBFC loans.
    loan_borrowers = nbfcs + corporations + suppliers

    for lender in lenders:
        add_random_relationships(
            InstrumentType.LOAN,
            lender,
            loan_borrowers,
            minimum=3,
            maximum=6,
        )

    # 2. Interbank funding.
    for bank in banks:
        add_random_relationships(
            InstrumentType.INTERBANK_FUNDING,
            bank,
            [other for other in banks if other.entity_id != bank.entity_id],
            minimum=1,
            maximum=3,
        )

    # 3. Bond holdings.
    for holder in bond_holders:
        add_random_relationships(
            InstrumentType.BOND,
            holder,
            bond_issuers,
            minimum=2,
            maximum=5,
        )

    # 4. Supplier trade credit.
    for supplier in suppliers:
        add_random_relationships(
            InstrumentType.TRADE_CREDIT,
            supplier,
            corporations,
            minimum=1,
            maximum=3,
        )

    # 5. Additional corporate trade-credit relationships.
    for corporation in corporations:
        eligible_corporations = [
            other
            for other in corporations
            if other.entity_id != corporation.entity_id
        ]

        add_random_relationships(
            InstrumentType.TRADE_CREDIT,
            corporation,
            eligible_corporations,
            minimum=0,
            maximum=2,
        )

    validate_relationships(entities, instruments, exposures)

    return instruments, exposures


def validate_relationships(
    entities: list[FinancialEntity],
    instruments: list[FinancialInstrument],
    exposures: list[FinancialExposure],
) -> None:
    """Validate IDs, references, issuer alignment, currency, and dates."""

    entity_ids = [entity.entity_id for entity in entities]
    instrument_ids = [
        instrument.instrument_id for instrument in instruments
    ]
    exposure_ids = [
        exposure.exposure_id for exposure in exposures
    ]

    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("Duplicate entity IDs detected")

    if len(instrument_ids) != len(set(instrument_ids)):
        raise ValueError("Duplicate instrument IDs detected")

    if len(exposure_ids) != len(set(exposure_ids)):
        raise ValueError("Duplicate exposure IDs detected")

    entity_id_set = set(entity_ids)

    instrument_by_id = {
        instrument.instrument_id: instrument
        for instrument in instruments
    }

    for instrument in instruments:
        if instrument.issuer_id not in entity_id_set:
            raise ValueError(
                f"Unknown issuer: {instrument.issuer_id}"
            )

    for exposure in exposures:
        if exposure.instrument_id not in instrument_by_id:
            raise ValueError(
                f"Unknown instrument: {exposure.instrument_id}"
            )

        if exposure.creditor_id not in entity_id_set:
            raise ValueError(
                f"Unknown creditor: {exposure.creditor_id}"
            )

        if exposure.debtor_id not in entity_id_set:
            raise ValueError(
                f"Unknown debtor: {exposure.debtor_id}"
            )

        if exposure.creditor_id == exposure.debtor_id:
            raise ValueError(
                f"Self-exposure detected: {exposure.exposure_id}"
            )

        instrument = instrument_by_id[exposure.instrument_id]

        if instrument.issuer_id != exposure.debtor_id:
            raise ValueError(
                f"Issuer/debtor mismatch for {exposure.instrument_id}"
            )

        if instrument.currency != exposure.currency:
            raise ValueError(
                f"Currency mismatch for exposure {exposure.exposure_id}: "
                f"instrument currency is {instrument.currency}, "
                f"exposure currency is {exposure.currency}"
            )

        if exposure.as_of_date < instrument.issue_date:
            raise ValueError(
                f"Exposure predates instrument {exposure.instrument_id}"
            )

        if (
            instrument.maturity_date is not None
            and exposure.as_of_date > instrument.maturity_date
        ):
            raise ValueError(
                f"Exposure is after maturity for {exposure.instrument_id}"
            )
