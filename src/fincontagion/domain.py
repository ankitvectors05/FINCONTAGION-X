
from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    BANK = "bank"
    NBFC = "nbfc"
    INVESTMENT_FUND = "investment_fund"
    CORPORATION = "corporation"
    SUPPLIER = "supplier"


@dataclass(frozen=True, slots=True)
class FinancialEntity:
    entity_id: str
    name: str
    entity_type: EntityType
    country: str
    base_currency: str = "INR"

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty")

        if not self.name.strip():
            raise ValueError("name cannot be empty")

        if not self.country.strip():
            raise ValueError("country cannot be empty")

        if (
            len(self.base_currency) != 3
            or not self.base_currency.isalpha()
            or not self.base_currency.isupper()
        ):
            raise ValueError(
                "base_currency must be a three-letter uppercase code"
            )
