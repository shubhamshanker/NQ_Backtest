# claude.md — Robust Best Practices for Claude Code

Strict, concise rules for trading/backtesting code.

---

## Core Rules

1. **Use actual data.** For quick tests, slice real data, never synthetic.
2. **Timezone:** Files in Chicago time → convert to **New York time (09:30–16:00)**.
3. **Accuracy:** All calcs must be exact. Entry always at **next candle open**.
4. **Be minimal:** Do only what’s asked. Clarify if unclear.
5. **Build slowly & lock:** One feature at a time, tested, no changes unless necessary.

---

## Environment

* Lock dependencies (`requirements.txt` or `poetry.lock`).
* Use reproducible env (venv/conda/poetry).
* Pin versions (`pandas`, `numpy`, etc.).
* Lint, format, type-check before commit.

---

## Data Handling

* Use tz-aware datetimes only.
* Convert Chicago → New York, validate DST.
* Filter to 09:30–16:00 ET.
* Keep raw files immutable; processed go in `/processed/`.

```python
# Timezone conversion
idx = pd.to_datetime(df.index).tz_localize('America/Chicago')
df.index = idx.tz_convert('America/New_York')
df = df.between_time('09:30','16:00')
```

---

## Trading Rules

* Signals computed at candle `t` → execute at open of `t+1`.
* No intra-candle lookahead.
* Market orders fill at next open, limit orders obey book simulation.

```python
if df.signal.iloc[i]:
    entry = df.open.iloc[i+1]
```

---

## Backtesting Checklist

* No lookahead bias.
* Entry = next open, exits realistic.
* Fees/slippage modeled.
* Position sizing explicit.
* Equity reconciles with trades.
* Results stable across slices & full run.

---

## Testing

* Unit: core P\&L, fees, slippage.
* Integration: small real slice vs golden output.
* Regression: store golden full-run results.
* Property tests: invariants (e.g. no negative balance unless margin allows).

---

## Logging

Log run\_id, git hash, env, data hashes, NY timestamps, runtime, P\&L, drawdown, Sharpe, trades.

---

## Performance

* Vectorize in pandas/numpy.
* Chunk large data.
* Profile + cache heavy ops.

---

## Style

* One responsibility per module.
* Explicit configs (YAML/JSON).
* Small, typed functions.

---

## Checklists

**Pre-run:**

* Data hashes match
* TZ converted, hours filtered
* Tests pass

**Post-run:**

* Equity = trades
* No invalid balances
* Golden outputs match

---

## Reminders

* **Accuracy > speed.**
* **Build small, test big.**
* **Next-open entry is sacred.**
