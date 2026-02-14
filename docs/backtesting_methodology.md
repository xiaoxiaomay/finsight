# Backtesting Methodology

## Philosophy

Our backtesting framework prioritizes **realism over optimism**. Every design
choice aims to produce results that translate to live trading, following the
principle: "If it can't survive walk-forward validation, it doesn't trade."

## Anti-Overfitting Measures

### 1. Point-in-Time Data Enforcement

All data is accessed strictly as of its **filing date** (for fundamentals) or
**publication date** (for macro indicators). This prevents look-ahead bias,
which is the #1 source of backtest overfitting.

```
Factor Signal at time T uses only data available before T:
  - Prices: Close on day T-1
  - Fundamentals: Only filings with filing_date <= T
  - Macro: Only observations with release_date <= T
```

### 2. Walk-Forward Validation

We use expanding-window walk-forward with anchored training:

```
Fold 1: Train [2020-01 → 2021-12] → Test [2022-01 → 2022-06]
Fold 2: Train [2020-01 → 2022-06] → Test [2022-07 → 2022-12]
Fold 3: Train [2020-01 → 2022-12] → Test [2023-01 → 2023-06]
Fold 4: Train [2020-01 → 2023-06] → Test [2023-07 → 2023-12]
Fold 5: Train [2020-01 → 2023-12] → Test [2024-01 → 2024-06]
```

Key properties:
- **No future leakage**: Test period always follows training period
- **Expanding window**: Training set grows, mimicking real-world accumulation
- **OOS validation**: Combined OOS metrics reported as primary results

### 3. Realistic Transaction Costs

| Cost Component | Default Value | Description |
|----------------|---------------|-------------|
| Commission | $0.005/share | Broker commission per share |
| Spread | 5 bps | Half bid-ask spread |
| Market impact | 10 bps | Price impact of trading |

Total round-trip cost for a $100 stock: ~$0.30 (30 bps)

### 4. Turnover Constraints

- Maximum position weight: 5% of portfolio
- Maximum sector weight: 25% of portfolio
- Rebalancing frequency: Monthly (reduces turnover vs. daily)
- Transaction cost penalty in optimization objective

## Factor Construction

### Cross-Sectional Z-Scoring

All factor values are normalized cross-sectionally (across all stocks on each date):

```python
z_score = (value - cross_sectional_mean) / cross_sectional_std
z_score = clip(z_score, -3, +3)  # Winsorize at ±3σ
```

This ensures:
- Factors are comparable across time (no time-series drift)
- Outliers are capped to prevent single-stock domination
- Each factor contributes proportionally to the composite

### Composite Signal

```
Composite = Σ (weight_i × z_score_i)
```

Default weights:

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Momentum (12-1) | 15% | Strongest academic evidence |
| Earnings Yield | 10% | Value: P/E inverse |
| Book/Market | 10% | Value: Fama-French HML |
| EV/EBITDA | 10% | Enterprise value metric |
| ROE | 10% | Quality: profitability |
| Gross Profitability | 10% | Quality: Novy-Marx factor |
| Accruals | 10% | Quality: earnings quality |
| Asset Growth | 10% | Quality: investment conservatism |
| Low Volatility | 10% | Low vol anomaly |
| Short-Term Reversal | 5% | Liquidity provision |

## Statistical Significance

### Alpha t-Statistic

We test whether strategy alpha is statistically different from zero:

```
H₀: α = 0 (no excess return)
H₁: α ≠ 0

α = mean(portfolio_return - benchmark_return) × 252
t = α / (std(active_returns) / √n × √252)

Significant if p-value < 0.05
```

**Harvey, Liu, and Zhu (2016) adjustment**: For multiple factor testing,
a t-statistic > 3.0 is preferred over the traditional 2.0 threshold.

### Sharpe Ratio Confidence

Sharpe ratio standard error (Lo, 2002):

```
SE(Sharpe) ≈ √((1 + 0.5 × Sharpe²) / n)
```

We report Sharpe with 95% confidence intervals:
```
Sharpe ± 1.96 × SE(Sharpe)
```

## Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Ann. Return | `(1 + total)^(252/n) - 1` | Compound annual growth |
| Ann. Volatility | `std(daily) × √252` | Annualized risk |
| Sharpe Ratio | `ann_return / ann_vol` | Risk-adjusted return (>1.0 good) |
| Sortino Ratio | `ann_return / downside_dev` | Downside risk-adjusted (>1.5 good) |
| Max Drawdown | `min(cumulative peak-to-trough)` | Worst loss from peak |
| Calmar Ratio | `ann_return / |max_drawdown|` | Return per unit of max loss |
| Information Ratio | `alpha / tracking_error` | Active return per unit of risk |
| Win Rate | `P(daily_return > 0)` | Fraction of positive days |

## Regime Awareness

The strategy monitors market regime and adjusts factor weights:

| Regime | Detection Rule | Factor Tilt |
|--------|---------------|-------------|
| Bull | SP500 > 200DMA, VIX < 20, yield > 0 | Momentum, Earnings Yield, ROE |
| Bear | SP500 < 200DMA, VIX > 25 | Low Vol, Quality, Accruals |
| High Vol | VIX > 30 | Low Vol, Quality (defensive) |
| Recovery | SP500 crossing above 200DMA | Value, Reversal |
| Neutral | None of the above | Balanced weights |

## Known Limitations

1. **Survivorship bias**: Universe based on current S&P 500 constituents,
   not historical membership. Partially mitigated by using 5-year windows.

2. **Small-cap coverage**: Limited to large/mid-cap stocks due to data
   availability. Factor premiums may differ in small-cap universe.

3. **Execution assumptions**: Assumes trades execute at close prices.
   Real-world slippage may be higher during volatile periods.

4. **Factor crowding**: Popular factors may be crowded, reducing forward
   returns. We monitor factor returns for decay.

## References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers.
- Novy-Marx, R. (2013). The other side of value: The gross profitability premium.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns.
- Lo, A. W. (2002). The statistics of Sharpe ratios.
