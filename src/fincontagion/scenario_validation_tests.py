
"""Permanent tests for scenario-result validation.

Run from the project root:
    python -m fincontagion.scenario_validation_tests
"""

import unittest
from dataclasses import replace
from datetime import date
from decimal import Decimal

from fincontagion.data.generator import (
    DatasetConfig,
    generate_entities,
    generate_instruments_and_exposures,
)
from fincontagion.data.position_generator import generate_financial_positions
from fincontagion.domain import EntityType, FinancialEntity
from fincontagion.instruments import (
    FinancialExposure,
    InstrumentType,
    RateType,
    FinancialInstrument,
)
from fincontagion.positions import FinancialPosition
from fincontagion.scenario_runner import (
    ContagionScenario,
    run_contagion_scenario,
)
from fincontagion.scenario_validation import validate_scenario_result


class ScenarioValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = DatasetConfig(seed=42)
        cls.entities = generate_entities(config)

        cls.instruments, cls.exposures = generate_instruments_and_exposures(
            entities=cls.entities,
            config=config,
        )

        cls.positions = generate_financial_positions(
            entities=cls.entities,
            exposures=cls.exposures,
            seed=config.seed,
        )

        cls.scenario = ContagionScenario(
            scenario_id="VALIDATOR_TEST",
            initial_defaulted_entity_ids=frozenset({"BANK_0010"}),
            recovery_rate=Decimal("0.40"),
        )

        cls.valid_result = run_contagion_scenario(
            entities=cls.entities,
            exposures=cls.exposures,
            positions=cls.positions,
            scenario=cls.scenario,
        )

    def assert_invalid(self, modified_rows, message: str) -> None:
        modified_result = replace(
            self.valid_result,
            contagion_results=tuple(modified_rows),
        )

        with self.assertRaisesRegex(ValueError, message):
            validate_scenario_result(modified_result, self.entities)

    def test_valid_scenario_is_accepted(self) -> None:
        validate_scenario_result(self.valid_result, self.entities)

    def test_duplicate_entity_round_is_rejected(self) -> None:
        rows = list(self.valid_result.contagion_results)
        rows.append(rows[0])
        self.assert_invalid(rows, "duplicate entity/round")

    def test_nonzero_round_zero_loss_is_rejected(self) -> None:
        rows = list(self.valid_result.contagion_results)

        index = next(
            i
            for i, row in enumerate(rows)
            if row.round_number == 0 and row.entity_id != "BANK_0010"
        )

        original = rows[index]
        rows[index] = replace(
            original,
            round_loss=Decimal("1"),
            equity_after_contagion=(
                original.equity_after_contagion - Decimal("1")
            ),
        )

        self.assert_invalid(rows, "round-zero loss must be zero")

    def test_inconsistent_equity_change_is_rejected(self) -> None:
        rows = list(self.valid_result.contagion_results)

        index = next(
            i for i, row in enumerate(rows)
            if row.round_number > 0
        )

        rows[index] = replace(
            rows[index],
            equity_after_contagion=(
                rows[index].equity_after_contagion + Decimal("1")
            ),
        )

        self.assert_invalid(rows, "equity change does not match")

    def test_inconsistent_cumulative_loss_is_rejected(self) -> None:
        rows = list(self.valid_result.contagion_results)

        index = next(
            i for i, row in enumerate(rows)
            if row.round_number > 0
        )

        rows[index] = replace(
            rows[index],
            cumulative_contagion_loss=(
                rows[index].cumulative_contagion_loss + Decimal("1")
            ),
        )

        self.assert_invalid(rows, "cumulative-loss arithmetic")

    def test_invalid_status_is_rejected(self) -> None:
        rows = list(self.valid_result.contagion_results)
        rows[0] = replace(rows[0], status="unknown_status")
        self.assert_invalid(rows, "invalid status")

    @staticmethod
    def make_new_default_scenario():
        """Build a tiny valid network that triggers one new default."""
        as_of = date(2026, 1, 1)

        debtor = FinancialEntity(
            entity_id="TEST_DEBTOR",
            name="Test Debtor",
            entity_type=EntityType.BANK,
            country="India",
        )
        creditor = FinancialEntity(
            entity_id="TEST_CREDITOR",
            name="Test Creditor",
            entity_type=EntityType.NBFC,
            country="India",
        )
        entities = [debtor, creditor]

        instrument = FinancialInstrument(
            instrument_id="TEST_LOAN",
            instrument_type=InstrumentType.LOAN,
            issuer_id=debtor.entity_id,
            currency="INR",
            issue_date=as_of,
            rate_type=RateType.FIXED,
            annual_interest_rate=Decimal("0.05"),
        )

        exposure = FinancialExposure(
            exposure_id="TEST_EXPOSURE",
            instrument_id=instrument.instrument_id,
            creditor_id=creditor.entity_id,
            debtor_id=debtor.entity_id,
            outstanding_amount=Decimal("100"),
            currency="INR",
            as_of_date=as_of,
        )

        positions = [
            FinancialPosition(
                entity_id=debtor.entity_id,
                as_of_date=as_of,
                cash_and_liquid_assets=Decimal("100"),
                total_assets=Decimal("200"),
                total_liabilities=Decimal("100"),
                equity_capital=Decimal("100"),
                short_term_liabilities=Decimal("50"),
            ),
            FinancialPosition(
                entity_id=creditor.entity_id,
                as_of_date=as_of,
                cash_and_liquid_assets=Decimal("20"),
                total_assets=Decimal("150"),
                total_liabilities=Decimal("100"),
                equity_capital=Decimal("50"),
                short_term_liabilities=Decimal("40"),
            ),
        ]

        scenario = ContagionScenario(
            scenario_id="TARGETED_NEW_DEFAULT_TEST",
            initial_defaulted_entity_ids=frozenset({debtor.entity_id}),
            recovery_rate=Decimal("0"),
        )

        result = run_contagion_scenario(
            entities=entities,
            exposures=[exposure],
            positions=positions,
            scenario=scenario,
        )

        return entities, result

    def test_new_default_flag_and_status_must_agree(self) -> None:
        entities, result = self.make_new_default_scenario()

        new_default_rows = [
            row for row in result.contagion_results
            if row.status == "new_default"
        ]
        self.assertEqual(len(new_default_rows), 1)

        rows = list(result.contagion_results)
        index = next(
            i for i, row in enumerate(rows)
            if row.status == "new_default"
        )

        rows[index] = replace(rows[index], newly_defaulted=False)
        corrupted_result = replace(result, contagion_results=tuple(rows))

        with self.assertRaisesRegex(
            ValueError, "new_default status requires"
        ):
            validate_scenario_result(corrupted_result, entities)


if __name__ == "__main__":
    unittest.main(verbosity=2)
