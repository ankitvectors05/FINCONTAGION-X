
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class StressScenario:
    """Configurable assumptions for a financial stress scenario.

    Rates are represented as decimal fractions:
    - 0.03 means a 3-percentage-point interest-rate shock.
    - 0.10 means a 10% eligible-asset price decline.
    - 0.05 means a 5% eligible-funding withdrawal.

    Interest-rate shocks may be positive or negative. Asset-price
    declines and funding-withdrawal rates must be between zero and one.
    """

    scenario_id: str
    name: str
    interest_rate_shock: Decimal
    asset_price_decline: Decimal
    funding_withdrawal_rate: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string.")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Scenario name must be a non-empty string.")

        for field_name, value in (
            ("interest_rate_shock", self.interest_rate_shock),
            ("asset_price_decline", self.asset_price_decline),
            ("funding_withdrawal_rate", self.funding_withdrawal_rate),
        ):
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be a Decimal.")

            if not value.is_finite():
                raise ValueError(f"{field_name} must be finite.")

        if not Decimal("-1") <= self.interest_rate_shock <= Decimal("1"):
            raise ValueError(
                "Interest-rate shock must be between -1 and 1 "
                "in decimal yield-change units."
            )

        if not Decimal("0") <= self.asset_price_decline <= Decimal("1"):
            raise ValueError(
                "Asset-price decline must be between 0 and 1."
            )

        if not Decimal("0") <= self.funding_withdrawal_rate <= Decimal("1"):
            raise ValueError(
                "Funding-withdrawal rate must be between 0 and 1."
            )


BASELINE_STRESS = StressScenario(
    scenario_id="SCENARIO_MODERATE",
    name="Illustrative Moderate Financial Stress",
    interest_rate_shock=Decimal("0.03"),
    asset_price_decline=Decimal("0.10"),
    funding_withdrawal_rate=Decimal("0.05"),
)



STRESS_SCENARIOS = (
    StressScenario(
        scenario_id="SCENARIO_MILD",
        name="Mild stress",
        interest_rate_shock=Decimal("0.01"),
        asset_price_decline=Decimal("0.03"),
        funding_withdrawal_rate=Decimal("0.02"),
    ),
    StressScenario(
        scenario_id="SCENARIO_MODERATE",
        name="Moderate stress",
        interest_rate_shock=Decimal("0.03"),
        asset_price_decline=Decimal("0.10"),
        funding_withdrawal_rate=Decimal("0.05"),
    ),
    StressScenario(
        scenario_id="SCENARIO_SEVERE",
        name="Severe stress",
        interest_rate_shock=Decimal("0.05"),
        asset_price_decline=Decimal("0.25"),
        funding_withdrawal_rate=Decimal("0.15"),
    ),
    StressScenario(
        scenario_id="SCENARIO_EXTREME",
        name="Extreme stress",
        interest_rate_shock=Decimal("0.08"),
        asset_price_decline=Decimal("0.40"),
        funding_withdrawal_rate=Decimal("0.30"),
    ),
)


