# Funding Model Training: Carry Forecasting

## Objective

Slipstream needs an estimate of future funding payments \(\hat{F}\) to pair with the
price-alpha forecast when solving the Kelly-style optimisation:

\[
\alpha^{\text{total}} = \alpha^{\text{price}} - \hat{F}
\]

The funding model therefore predicts the **H-hour forward sum of funding rates** for
each asset, normalised so that one unit corresponds to one unit of funding
volatility. The prediction is converted back to raw funding at execution time via
the current EWMA funding volatility.

---

## Data Pipeline

- **Inputs**: 4-hour funding histories for the full Hyperliquid universe.
- **Feature engineering**: reuse the same EWMA stack as the alpha model — spans
  \[2, 4, 8, 16, 32, 64\] hours, clipped to \(\pm 5\) after normalisation by the
  128-hour EWMA funding volatility.
- **Targets**: sum of the next `H / 4` funding prints, normalised by the same
  EWMA volatility and winsorised to \(\pm 10\).
- **Warm-up**: observations enter the training set only after accumulating at
  least 128 hours of history.

The helper `prepare_funding_training_data()` aligns the feature matrix, target,
and scaling series (`vol_scale`) and enforces all guardrails.

---

## Training & Validation

- Ridge regression with the same bootstrap + walk-forward stack used by the
  alpha model.
- Quantile diagnostics: every run prints a 10-decile table summarising mean
  predictions, realised forward funding, and t-stats. This provides an immediate
  check that the carry signal is strongest in the tails.
- Robust PCA metadata is unnecessary (funding is purely asset-specific), so the
  pipeline is lighter than the price-alpha stack.

### Example: H = 4 hours (n_bootstrap = 10)

| Quantile | Count | Pred µ | Actual µ | t-stat | Sig |
|---------:|------:|-------:|---------:|-------:|:---:|
| 0 | 53,148 | -1.54 | -0.83 | -151.5 | *** |
| 1 | 53,147 | -0.20 | -0.04 |  -8.8 | *** |
| 2 | 53,147 |  0.36 |  0.39 | 118.0 | *** |
| 3 | 53,147 |  0.72 |  0.69 | 196.2 | *** |
| 4 | 53,147 |  1.08 |  0.99 | 248.9 | *** |
| 5 | 53,147 |  1.53 |  1.34 | 300.7 | *** |
| 6 | 53,147 |  2.11 |  1.76 | 350.3 | *** |
| 7 | 53,147 |  2.99 |  2.29 | 390.3 | *** |
| 8 | 53,937 |  5.00 |  4.82 | 375.4 | *** |
| 9 | 52,358 |  6.75 |  8.49 | 784.3 | *** |

All deciles are strongly significant (|t| ≫ 2), with the top bucket capturing the
largest positive carry. The walk-forward \(R^2_{\text{oos}}\) for this run is ~0.72,
demonstrating that funding persistence is considerably stronger than price alpha.

---

## Usage

- Run `scripts/find_optimal_H_funding.py` to sweep candidate horizons. Outputs
  are written to `data/features/funding_models/` and include the quantile table,
  λ, bootstrap diagnostics, and comparison CSV.
- Combine the predicted carry with the price-alpha forecast by rescaling the
  normalised prediction with `vol_scale` and subtracting it from the price alpha
  before feeding the optimisation stage.

---

## Next Steps

1. Evaluate longer horizons (e.g. 24 h, 48 h) to see how carry persistence fades.
2. Consider augmenting features with cross-sectional signals (e.g. funding z-score
   vs. peers) if additional predictive power is needed.
3. Integrate the funding forecast artefacts into the portfolio simulation to
   verify that the combined alpha + funding stack matches theoretical expectations.
