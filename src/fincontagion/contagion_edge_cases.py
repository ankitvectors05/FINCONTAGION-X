
"""Permanent edge-case tests for the credit-contagion engine.

Run from the project root:
    python -m fincontagion.contagion_edge_cases
"""

import unittest
from datetime import date
from decimal import Decimal

from fincontagion.contagion_engine import simulate_credit_contagion
from fincontagion.domain import EntityType, FinancialEntity
from fincontagion.instruments import FinancialExposure
from fincontagion.positions import FinancialPosition


AS_OF_DATE = date(2026, 1, 1)


def make_entity(entity_id: str) -> FinancialEntity:
    return FinancialEntity(
        entity_id=entity_id,
        name=entity_id,
        entity_type=EntityType.BANK,
        country="India",
        base_currency="INR",
    )


def make_position(
    entity_id: str,
    equity: Decimal = Decimal("100"),
) -> FinancialPosition:
    liabilities = Decimal("900")
    return FinancialPosition(
        entity_id=entity_id,
        as_of_date=AS_OF_DATE,
        cash_and_liquid_assets=Decimal("100"),
        total_assets=liabilities + equity,
        total_liabilities=liabilities,
        equity_capital=equity,
        short_term_liabilities=Decimal("100"),
    )


def make_exposure(
    exposure_id: str,
    creditor_id: str,
    debtor_id: str,
    amount: Decimal = Decimal("100"),
) -> FinancialExposure:
    return FinancialExposure(
        exposure_id=exposure_id,
        instrument_id=f"INST_{exposure_id}",
        creditor_id=creditor_id,
        debtor_id=debtor_id,
        outstanding_amount=amount,
        currency="INR",
        as_of_date=AS_OF_DATE,
    )


class ContagionEdgeCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entities = [
            make_entity("BANK_A"),
            make_entity("BANK_B"),
            make_entity("BANK_C"),
        ]
        self.positions = [
            make_position("BANK_A"),
            make_position("BANK_B"),
            make_position("BANK_C"),
        ]

    def run_model(
        self,
        exposures: list[FinancialExposure],
        initial_defaults: set[str],
        recovery: Decimal,
    ):
        return simulate_credit_contagion(
            entities=self.entities,
            exposures=exposures,
            positions=self.positions,
            initial_defaulted_entity_ids=initial_defaults,
            recovery_rate=recovery,
        )

    def test_zero_recovery_causes_full_exposure_loss(self) -> None:
        results = self.run_model(
            [make_exposure("EXP_1", "BANK_A", "BANK_B")],
            {"BANK_B"},
            Decimal("0"),
        )

        bank_a = [
            row for row in results
            if row.entity_id == "BANK_A" and row.round_number == 1
        ]
        self.assertEqual(len(bank_a), 1)
        self.assertEqual(bank_a[0].round_loss, Decimal("100"))

    def test_full_recovery_causes_no_credit_loss(self) -> None:
        results = self.run_model(
            [make_exposure("EXP_1", "BANK_A", "BANK_B")],
            {"BANK_B"},
            Decimal("1"),
        )

        bank_a = [
            row for row in results
            if row.entity_id == "BANK_A" and row.round_number > 0
        ]
        self.assertEqual(bank_a, [])

    def test_no_exposures_record_initial_default_only(self) -> None:
        results = self.run_model([], {"BANK_B"}, Decimal("0.4"))

        round_zero = [
            row for row in results if row.round_number == 0
        ]
        self.assertEqual(len(round_zero), 3)
        self.assertEqual(
            {row.entity_id for row in round_zero if row.newly_defaulted},
            {"BANK_B"},
        )
        self.assertEqual(len(results), 3)

    def test_unknown_initial_default_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown initial default"):
            self.run_model([], {"BANK_UNKNOWN"}, Decimal("0.4"))

    def test_duplicate_exposure_id_is_rejected(self) -> None:
        exposures = [
            make_exposure("EXP_DUP", "BANK_A", "BANK_B"),
            make_exposure("EXP_DUP", "BANK_C", "BANK_B"),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate exposure ID"):
            self.run_model(exposures, {"BANK_B"}, Decimal("0.4"))

    def test_entity_defaults_when_loss_equals_equity(self) -> None:
        self.positions = [
            make_position("BANK_A", Decimal("60")),
            make_position("BANK_B"),
            make_position("BANK_C"),
        ]

        results = self.run_model(
            [make_exposure("EXP_1", "BANK_A", "BANK_B", Decimal("100"))],
            {"BANK_B"},
            Decimal("0.4"),
        )

        bank_a = [
            row for row in results
            if row.entity_id == "BANK_A" and row.round_number == 1
        ]
        self.assertEqual(len(bank_a), 1)
        self.assertEqual(bank_a[0].round_loss, Decimal("60"))
        self.assertTrue(bank_a[0].newly_defaulted)
        self.assertEqual(bank_a[0].status, "new_default")

    def test_default_propagates_through_multiple_rounds(self) -> None:
        self.positions = [
            make_position("BANK_A", Decimal("50")),
            make_position("BANK_B"),
            make_position("BANK_C"),
        ]

        exposures = [
            make_exposure("EXP_1", "BANK_A", "BANK_B", Decimal("100")),
            make_exposure("EXP_2", "BANK_C", "BANK_A", Decimal("80")),
        ]

        results = self.run_model(
            exposures,
            {"BANK_B"},
            Decimal("0.4"),
        )

        bank_a = [
            row for row in results
            if row.entity_id == "BANK_A" and row.round_number == 1
        ]
        bank_c = [
            row for row in results
            if row.entity_id == "BANK_C" and row.round_number == 2
        ]

        self.assertEqual(bank_a[0].round_loss, Decimal("60"))
        self.assertTrue(bank_a[0].newly_defaulted)
        self.assertEqual(bank_c[0].round_loss, Decimal("48"))

    def test_domain_rejects_self_exposure(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be different"):
            make_exposure("EXP_SELF", "BANK_A", "BANK_A")

    def test_domain_rejects_zero_exposure(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            make_exposure(
                "EXP_ZERO",
                "BANK_A",
                "BANK_B",
                Decimal("0"),
            )

    def test_circular_exposures_terminate(self) -> None:
        exposures = [
            make_exposure("EXP_AB", "BANK_B", "BANK_A", Decimal("80")),
            make_exposure("EXP_BA", "BANK_A", "BANK_B", Decimal("100")),
        ]

        results = self.run_model(
            exposures,
            {"BANK_A"},
            Decimal("0.4"),
        )

        losses = [
            row for row in results
            if row.round_number > 0
        ]

        self.assertEqual(
            sum((row.round_loss for row in losses), Decimal("0")),
            Decimal("48"),
        )

        bank_b = [
            row for row in results
            if row.entity_id == "BANK_B" and row.round_number == 1
        ]

        self.assertEqual(len(bank_b), 1)
        self.assertEqual(bank_b[0].round_loss, Decimal("48"))
        self.assertFalse(bank_b[0].newly_defaulted)
        self.assertTrue(
            all(row.round_number == 1 for row in losses)
        )

    def test_zero_equity_without_loss_is_not_auto_defaulted(self) -> None:
        self.positions = [
            make_position("BANK_A", Decimal("0")),
            make_position("BANK_B"),
            make_position("BANK_C"),
        ]

        results = self.run_model([], {"BANK_B"}, Decimal("0.4"))

        bank_a = [
            row for row in results
            if row.entity_id == "BANK_A" and row.round_number == 0
        ]

        self.assertEqual(len(bank_a), 1)
        self.assertEqual(bank_a[0].status, "capital_impaired")
        self.assertFalse(bank_a[0].newly_defaulted)

    def test_zero_equity_loss_does_not_trigger_new_default(self) -> None:
        self.positions = [
            make_position("BANK_A", Decimal("0")),
            make_position("BANK_B"),
            make_position("BANK_C"),
        ]

        exposures = [
            make_exposure("EXP_1", "BANK_A", "BANK_B", Decimal("100")),
        ]

        results = self.run_model(
            exposures,
            {"BANK_B"},
            Decimal("0.4"),
        )

        bank_a = [
            row for row in results
            if row.entity_id == "BANK_A" and row.round_number == 1
        ]

        self.assertEqual(len(bank_a), 1)
        self.assertEqual(bank_a[0].round_loss, Decimal("60"))
        self.assertFalse(bank_a[0].newly_defaulted)
        self.assertEqual(bank_a[0].status, "capital_impaired")

    def test_invalid_recovery_rate_is_rejected(self) -> None:
        invalid_recovery_rates = (
            Decimal("-0.01"),
            Decimal("1.01"),
            Decimal("NaN"),
            Decimal("Infinity"),
        )

        for recovery in invalid_recovery_rates:
            with self.subTest(recovery=recovery):
                with self.assertRaises(ValueError):
                    self.run_model([], {"BANK_B"}, recovery)

    def test_missing_financial_position_is_rejected(self) -> None:
        incomplete_positions = self.positions[:-1]

        with self.assertRaisesRegex(
            ValueError,
            "exactly one row per entity",
        ):
            simulate_credit_contagion(
                entities=self.entities,
                exposures=[],
                positions=incomplete_positions,
                initial_defaulted_entity_ids={"BANK_B"},
                recovery_rate=Decimal("0.4"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
