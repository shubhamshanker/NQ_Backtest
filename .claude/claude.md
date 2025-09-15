# claude.md — Robust Best Practices for Claude Code

Strict, concise rules for trading/backtesting code.

---

## Global — Most Important Rules (Top priority)

1. **Limit scope; perfect the details.** Limit the number of simultaneous features or details. Pick the essential items and obsess over each one — make them **perfect, robust, modular, and reusable**.
2. **Follow instructions to the letter.** Execute the task exactly as specified. If you encounter blockers, ambiguity, or must take a different approach, **stop, document the reason, and ask for permission** before proceeding.
3. **Write clean, succinct, robust Python.** Code must be easy to read, well-typed, and aim for correctness. **Include unit and integration tests** for complex or uncertain tasks. Strive for correctness in calculations (target: 99%+ where feasible) and explicitly document assumptions.
4. **Contect Updated** Use @docs folder to maintain or revive complete context to do the task given to the best of your ability. 

---

## Core Rules

1. **Use actual data.** For quick tests slice *real* data; never rely on synthetic samples for validation.
2. **Timezone:** Files are in **Chicago time** — convert to **New York time (09:30–16:00 ET)** and validate DST handling.
3. **Accuracy:** All calculations must be exact. **Entry is always at the next candle open** unless otherwise specified.
4. **Be minimal:** Implement only what is asked. If requirements are unclear, ask before proceeding.
5. **Build slowly & lock:** Add one feature at a time, fully test it, and keep changes small. Avoid large, risky commits.
6. **PARQUET ONLY:** Use Parquet as the canonical data source. Paraquet has NQ time data. **Never** silently fall back to CSV or other formats; be explicit if alternate sources are used.

---

## Environment

* Lock dependencies via `requirements.txt`, `poetry.lock`, or equivalent.
* Use a reproducible environment (venv/conda/poetry).
* Pin versions for critical libs (`pandas`, `numpy`, etc.).
* Lint, format and type-check before committing.

---

## Data Handling

* Use timezone-aware datetimes only.
* Convert Chicago → New York and validate DST transitions.
* Filter to **09:30–16:00 ET** for session-based strategies.
* Keep raw files immutable; write processed outputs to `/processed/`.

```python
# Timezone conversion
idx = pd.to_datetime(df.index)
idx = idx.tz_localize('America/Chicago')
idx = idx.tz_convert('America/New_York')
df.index = idx
# filter session
df = df.between_time('09:30', '16:00')
```

---

## Trading Rules

* Signals computed on candle `t` → execute at open of `t+1` (no intra-candle lookahead).
* Market orders assume fill at next open; limit orders should be simulated with a book model.

```python
if df.signal.iloc[i]:
    entry_price = df.open.iloc[i+1]
```

---

## Backtesting Checklist

* No lookahead bias.
* Entry = next open; exits modeled realistically.
* Fees and slippage must be modeled and configurable.
* Position sizing must be explicit and auditable.
* Equity must reconcile with trades; run balance checks.
* Verify results are stable across small slices and a full run.

---

## Testing

* **Unit tests:** core P\&L calculation, fees, slippage, position sizing.
* **Integration tests:** small real-data slice compared to a golden output.
* **Regression tests:** store golden full-run outputs and diff for changes.
* **Property tests:** invariants (e.g. no negative balance unless margin allows).

---

## Pre-commit Rules

Before committing changes, follow this strict pre-commit flow **every time**:

1. Run the full test suite (unit + critical integration slices).
2. Run linters and type checkers (`flake8`/`ruff`, `black`, `mypy` or equivalent).
3. Run a quick integration run on a small real-data slice and compare against a golden output (pass/fail).
4. Remove fluff and unnecessary files (temporary files, experimental notebooks, compiled caches): e.g. `__pycache__/`, `*.pyc`, large unused datasets.
5. Update the `CHANGELOG.md` with a concise note: what changed, why, and how to use it.
6. Stage only relevant files and craft a short, clear commit message.

**Commit message template (short & sweet):**

```
<scope>: brief summary (max 72 chars)

One-line reason / why (optional 1–2 lines).
Refs: #issue-number (if applicable)
```

**Example:**

```
backtest: add slippage model and tests

Adds percentage-based slippage and unit tests for P&L reconciliation.
```

Only commit after all pre-commit steps succeed.

---

## Logging & Provenance

Log the following for every run:

* `run_id`, git hash, environment (python + pinned deps), data hashes, NY timestamps.
* Runtime, P\&L, max drawdown, Sharpe, trades count.
* Export trade-level ledger (parquet) with tz-aware timestamps.

---

## Backtest Statistics Specification (NQ Futures — 1 contract = \$20/point)

**Units & conversions:**

* 1 point = \$20 per contract. Multiply points by 20 to convert to USD for NQ.
* Show both **points** and **dollars** side-by-side on all reports.

### Ultra-Important Metrics (must always be calculated)

* **Expectancy** (per trade): in points and \$.

  * Formula: `Expectancy = WinRate * AvgWinPoints - LossRate * AvgLossPoints`.
  * Convert to USD: `Expectancy_USD = Expectancy_points * 20 * contracts`.
* **Profit Factor:** `Gross Profit / Gross Loss`.
* **Max Drawdown:** show as **points**, **USD**, and **%** (peak-to-trough equity loss).
* **Win Rate × Payoff Ratio:** `WinRate` and `PayoffRatio = AvgWin / AvgLoss`.
* **Sharpe** (and/or **Sortino**) Ratio — annualized when using daily returns.
* **Average Daily Points** — mean net points per trading day.
* **Average Win Points** — mean points per winning trade.
* **Average Loss Points** — mean points per losing trade.
* **CAGR** — annualized growth %.

### Full Statistics Set (robust implementation required)

**Performance:**

* Net Profit (USD and %)
* Gross Profit (USD)
* Gross Loss (USD)
* Profit Factor
* Expectancy (points and USD)
* Win Rate (%) / Loss Rate (%)
* Payoff Ratio (Avg Win ÷ Avg Loss)
* Average Trade (points and USD)
* CAGR

**Risk & Drawdown:**

* Max Drawdown (USD, %, points)
* Ulcer Index
* Recovery Factor (Net Profit ÷ Max Drawdown)
* R-Multiple distribution
* Risk of Ruin

**Trade Dynamics:**

* Total Trades
* Average Holding Period
* Longest Trade Duration
* Trades per Day / Week
* Time to Recover from Drawdown

**Return Quality:**

* Sharpe Ratio, Sortino Ratio, Calmar Ratio, MAR Ratio
* Return skewness & kurtosis

**Equity & Volatility:**

* Standard Deviation of returns
* Coefficient of Variation (CV)
* Win/Loss streaks (longest runs)
* Equity curve stability (R² regression fit)
* Monthly / Weekly breakdown of P\&L

**Points-Based (NQ-specific):**

* Average Daily Points
* Average Win Points
* Average Loss Points

---

## Reporting & Display Rules

* **Always display both points and USD** for NQ (clearly labeled).
* Use **NY timezone** for all timestamps in reports and trade logs.
* Provide a trade-level ledger with these columns at minimum:
  `trade_id, entry_time_NY, exit_time_NY, instrument, contracts, entry_price, exit_price, points, USD, fees, slippage, pnl_usd, running_equity`.
* Round displayed dollar values to two decimals; points to two decimal places (or as appropriate for instrument tick size).
* Export final summaries to `parquet` and provide `csv` and `json` derivatives for quick inspection.
* Visuals: include equity curve, drawdown plot, distribution of returns, and R-multiple histogram.
* When reporting ratios (Sharpe, Sortino), include the data frequency and the annualization method.

---

## Performance

* Vectorize heavy ops with pandas/numpy.
* Chunk & stream large datasets.
* Profile and cache expensive steps.

---

## Style

* Single responsibility per module.
* Explicit configs (YAML/JSON) and small typed functions.
* Keep code readable and documented; prefer clarity over cleverness.

---

## Checklists

**Pre-run:**

* Data hashes verified
* Timezone converted & session filtered
* Tests pass

**Post-run:**

* Equity reconciles with trades
* No invalid balances
* Golden outputs match (if applicable)

---

## Reminders

* **Accuracy > speed.**
* **Build small, test big.**
* **Next-open entry is sacred.**

---

*Document maintained for Claude Code contributors — keep it concise and update `CHANGELOG.md` for every change.*
