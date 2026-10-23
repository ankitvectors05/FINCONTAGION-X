# FINCONTAGION-X — Dataset Dictionary

## 1. Purpose

Define the data contract for the financial ecosystem, stress-testing engine, contagion model, graph analytics, and machine-learning components.

All datasets must use stable identifiers, explicit units, dates, and documented provenance. Synthetic data must be labelled as synthetic.

## 2. Entities — `entities.csv`

One row per financial or corporate participant.

| Field | Meaning |
|---|---|
| entity_id | Unique, stable participant ID |
| name | Participant name |
| entity_type | bank, nbfc, investment_fund, corporation, supplier |
| country | Country of registration or operation |
| base_currency | Three-letter accounting currency code |
| data_origin | synthetic or public |
| as_of_date | Date the record represents |

**Key:** `entity_id`

## 3. Financial Instruments — `instruments.csv`

One row per financial contract or security.

| Field | Meaning |
|---|---|
| instrument_id | Unique instrument ID |
| instrument_type | loan, bond, equity, interbank_funding, trade_credit |
| issuer_id | Entity that issued or originated the instrument |
| currency | Contract currency |
| issue_date | Origination or issue date |
| maturity_date | Contractual maturity date, if applicable |
| rate_type | fixed, floating, or not_applicable |
| annual_interest_rate | Annual contractual rate as a decimal; 0.08 means 8% |
| data_origin | synthetic or public |

**Key:** `instrument_id`

`issuer_id` must reference a valid entity. A missing maturity date is permitted for instruments without a defined maturity.

## 4. Financial Exposures — `exposures.csv`

One row per directed financial obligation or holding at a particular reporting date.

| Field | Meaning |
|---|---|
| exposure_id | Unique exposure-record ID |
| instrument_id | Related instrument ID |
| creditor_id | Entity providing funds or holding a debt claim |
| debtor_id | Entity owing the debt |
| outstanding_amount | Outstanding principal or contractual exposure, in the stated currency |
| currency | Currency of the exposure |
| as_of_date | Reporting date |
| data_origin | synthetic or public |

**Key:** `exposure_id`

`instrument_id`, `creditor_id`, and `debtor_id` must reference valid records. Creditor and debtor must differ. Amounts must be positive.

For equity holdings, the ownership relationship will be modelled explicitly in a later extension rather than treating equity as ordinary debt.

## 5. Financial Positions — `financial_positions.csv`

One row per entity and reporting date.

| Field | Meaning |
|---|---|
| entity_id | Participant ID |
| as_of_date | Reporting date |
| cash_and_liquid_assets | Liquid assets |
| total_assets | Total assets |
| total_liabilities | Total liabilities |
| equity_capital | Accounting equity |
| short_term_liabilities | Liabilities due within the defined short-term horizon |
| data_origin | synthetic or public |

All monetary amounts must use the entity's base currency unless a separate currency is explicitly specified.

Accounting validation must check:

**Total assets = total liabilities + equity capital**

Any rounding tolerance must be documented. Derived financial positions must not double-count exposure records.

## 6. Market Observations — `market_observations.csv`

One row per market variable, observation date, and applicable market or instrument.

| Field | Meaning |
|---|---|
| observation_id | Unique observation ID |
| observation_date | Date of observation |
| variable_name | Interest rate, bond yield, asset price, exchange rate, or other documented variable |
| market_or_instrument_id | Applicable market or instrument identifier, if relevant |
| value | Numeric observation |
| unit | Percent, decimal rate, currency per unit, price, or another explicit unit |
| data_origin | synthetic or public |
| source_reference | Source identifier or documentation reference |

Market series must have consistent units and documented frequency.

## 7. Data quality requirements

1. Primary identifiers must be unique and non-empty.
2. Foreign-key references must resolve to existing records.
3. Dates must be valid and contract timelines must be consistent.
4. Monetary amounts must be finite, with permitted signs defined by the field's financial meaning.
5. Currency and unit conventions must be consistent.
6. Synthetic records must never be labelled as observed public data.
7. Missing values must be explicitly permitted or rejected by field.
8. Dataset generation must be reproducible using a recorded random seed.
9. Generated records must pass validation before entering the simulation.
10. Every simulated output must identify the input dataset version and scenario assumptions used.

## 8. Dataset generation strategy

The initial ecosystem will be generated synthetically with controlled relationships among banks, NBFCs, funds, corporations, and suppliers.

Generation will proceed in dependency order:

1. Entities
2. Financial instruments
3. Financial exposures and holdings
4. Financial positions
5. Market observations
6. Validation and consistency checks

The initial dataset is a research simulation, not a representation of confidential real-world counterparty exposures. Public data may later calibrate selected parameters, with source and transformation details preserved.

## 9. Modelling boundaries

- Funding withdrawals are liquidity outflows, not automatically accounting losses.
- Interest-rate shocks affect instruments according to their repricing, maturity, and valuation characteristics.
- Credit losses depend on exposure, probability of default, and loss given default, with recoveries and collateral treated consistently.
- A single underlying economic loss must not be counted multiple times across connected entities.
- Synthetic simulation performance does not establish predictive accuracy for real financial crises.