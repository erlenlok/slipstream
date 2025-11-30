# Data Safety Guarantees

## Critical Fix: Data Truncation Prevention (Nov 3, 2025)

### Issue
The `scripts/data_load.py` script was previously doing a **direct overwrite** with `to_csv()`, which caused catastrophic data loss when running with `--days` shorter than the existing dataset.

**Example of the problem:**
- Existing data: April 17 - Nov 3 (200 days)
- Running `--days 30` would fetch Oct 4 - Nov 3
- **Result: All data from April 17 - Oct 4 was DELETED**

### Fix Implemented

Three layers of protection have been added:

#### 1. **Merge Function** (`_merge_with_existing()`)
Every data write now:
1. Reads existing CSV if it exists
2. Concatenates existing + new data
3. Removes duplicates (keeping newer data)
4. Sorts by datetime index
5. Logs the merge operation

**Result:** Historical data is ALWAYS preserved.

#### 2. **Pre-Flight Check** (`build_for_universe()`)
Before fetching data for all markets:
1. Checks BTC/ETH sample files
2. Compares existing data range vs fetch request
3. Displays clear warning if fetch starts after existing data
4. Confirms that merge function will preserve data

**Example output:**
```
================================================================================
PRE-FLIGHT CHECK: BTC_candles_4h.csv
  Existing data: 2025-04-17 to 2025-11-03 (200 days)
  Fetch request:  2025-10-27 to 2025-11-03 (7 days)
  ⚠️  WARNING: Fetch starts 193 days AFTER existing data!
  ✓  SAFE: Merge function will preserve historical data
================================================================================
```

#### 3. **Merge Logging** (per-asset)
Each asset shows detailed merge statistics:
```
  Saving BTC:
    Merged data: 1201 existing + 43 new = 1201 total rows
      Old range: 2025-04-17 08:00:00+00:00 to 2025-11-03 08:00:00+00:00
      New range: 2025-10-27 08:00:00+00:00 to 2025-11-03 08:00:00+00:00
      Final range: 2025-04-17 08:00:00+00:00 to 2025-11-03 08:00:00+00:00
```

## Safe Usage

### For Daily Updates
```bash
# Safe: Updates with recent data, preserves history
uv run hl-load --all --days 7
```

### For Backfills
```bash
# Safe: Extends history backward
uv run hl-load --all --days 365
```

### For Specific Date Ranges
```bash
# Safe: Uses explicit dates
uv run hl-load --all --start 2025-01-01 --end 2025-11-03
```

## What Changed

### Before (UNSAFE):
```python
candles_resampled.to_csv(path, index=True)  # Direct overwrite
```

### After (SAFE):
```python
candles_final = _merge_with_existing(candles_resampled, path)  # Merge
candles_final.to_csv(path, index=True)  # Write combined data
```

## Verification

After any data load operation, verify data integrity:

```bash
# Check row count (should never decrease)
wc -l data/market_data/BTC_candles_4h.csv

# Check date range (start should never move forward)
head -3 data/market_data/BTC_candles_4h.csv
tail -3 data/market_data/BTC_candles_4h.csv
```

## Guarantee

**This will NEVER happen again.** The script now:
- ✅ Preserves all historical data automatically
- ✅ Warns when fetch window is smaller than existing data
- ✅ Logs every merge operation with full visibility
- ✅ Handles duplicates safely (keeps newer data)
- ✅ Maintains data chronological order

Running `hl-load` with any parameters is now **always safe**.
