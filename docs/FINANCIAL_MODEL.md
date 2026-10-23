# FINCONTAGION-X — Financial Model and Accounting Rules

## 1. Purpose

This document defines the accounting identities, financial quantities, and transmission rules used by the synthetic ecosystem and stress-testing engine.

All monetary values must include an identifiable currency and reporting date.

## 2. Entity balance sheet

Each bank, NBFC, fund, or corporation has a financial position represented by:

**Assets = Liabilities + Equity**

- **Assets:** cash, loans receivable, bonds held, other investments, and other assets.
- **Liabilities:** deposits, borrowings, bonds issued, trade payables, and other obligations.
- **Equity:** the residual accounting interest after liabilities are deducted from assets.

The initial synthetic balance sheet must satisfy the accounting identity within a documented rounding tolerance.

A supplier or other non-financial participant may use a simplified balance sheet, but it must obey the same accounting identity.

## 3. Credit exposure

For a debt claim from creditor A to debtor B:

- A records a financial asset.
- B records a corresponding financial liability, subject to contractual scope and accounting treatment.
- Outstanding principal is not automatically equal to the eventual loss.

For a simplified credit-loss model:

**Expected credit loss = Exposure at default × Probability of default × Loss given default**

Where:
- Exposure at default (EAD) is the expected outstanding exposure when default occurs.
- Probability of default (PD) is the estimated probability of default over the chosen horizon.
- Loss given default (LGD) is the fraction of the exposure not recovered after default.

PD and LGD must be represented on consistent scales between 0 and 1.

This formula is a modelling approximation. Collateral, guarantees, recoveries, discounting, and timing may require additional treatment.

## 4. Interest-rate risk

An interest-rate shock does not affect every instrument identically.

- Fixed-rate debt generally retains its contractual coupon until repricing or maturity.
- Floating-rate debt may reprice according to its reference rate, reset date, and contractual spread.
- The market value of a fixed-rate bond generally falls when comparable market yields rise.
- Borrowers with floating-rate debt may face higher interest expenses after their reset dates.

The model must distinguish contractual cash-flow changes from market-value changes. It must not apply the same interest-cost increase to every instrument without considering contract terms.

## 5. Liquidity and funding withdrawals

Liquidity represents the ability to meet obligations when they fall due.

A funding withdrawal reduces available cash or other liquid resources and may require refinancing, asset sales, or emergency funding.

**Funding withdrawal ≠ automatic accounting loss.**

The simulation must record the cash outflow and separately model any subsequent costs, asset-sale discounts, or defaults caused by liquidity pressure.

Liquidity shortfall must be calculated against obligations due within a specified time horizon, not simply from total liabilities.

## 6. Asset-price shocks

A market-price decline changes the value of assets exposed to that market.

For a simplified holding:

**Stressed asset value = Initial asset value × (1 − price decline)**

This calculation applies only to assets covered by the relevant shock. The model must not apply an equity-market decline indiscriminately to cash, loans, or unrelated instruments.

If an asset is sold after its market value has already been reduced, the same price decline must not be charged again as a separate loss.

## 7. Contagion channels

The initial system will distinguish these transmission mechanisms:

1. **Direct credit channel:** a debtor's default creates losses for creditors.
2. **Funding channel:** withdrawals or non-renewal of funding create liquidity pressure.
3. **Market channel:** price declines reduce the value of exposed holdings.
4. **Fire-sale channel:** forced asset sales can depress prices and affect other holders.
5. **Corporate payment channel:** delayed or missed payments can weaken suppliers and other counterparties.
6. **Interest-rate channel:** changes in borrowing costs and market valuations affect entities differently according to contract terms.

Each simulated effect must record its initiating event, affected entity, transmission mechanism, and calculation basis.

## 8. System-wide accounting safeguards

- A loan's asset and corresponding liability must be reconciled where both sides are represented.
- Inter-entity claims must not be added to external system-wide wealth as if they were new resources.
- Internal claims and obligations must be distinguished from exposures to entities outside the modelled ecosystem.
- Credit losses, liquidity outflows, asset revaluations, and funding costs must be tracked separately.
- The same economic loss must not be counted multiple times as it propagates through the network.
- Every scenario must retain its starting state and assumptions so that stressed results can be compared consistently.

## 9. Scenario conventions

The initial illustrative scenario uses:
- Interest-rate shock: +3 percentage points.
- Asset-price decline: 10% for explicitly eligible assets.
- Funding withdrawals: 5% of the eligible funding base.
- Contagion propagation: moderate, to be translated into explicit model parameters before simulation.

These are experimental assumptions, not forecasts or claims about actual financial conditions.

## 10. Model limitations

The first implementation will be a simplified, transparent simulation. It will not claim to reproduce a central bank's regulatory stress test or predict financial crises accurately.

All assumptions, simplifications, calibration choices, and validation results must be documented.