# Next Steps: Historical Data Acquisition

You're all set up! Here's what to do next:

## Current Status ✅

- ✅ Signal generation framework implemented (`src/slipstream/signals/`)
- ✅ S3 downloader script ready (`scripts/fetch_s3_historical.py`)
- ✅ Test notebook created (`notebooks/momentum_panel_test.ipynb`)
- ✅ AWS CLI installed via uv
- ✅ Documentation written

## What You Need To Do

### 1. Install LZ4 (1 minute)

```bash
sudo apt install lz4
```

### 2. Get AWS Credentials (5 minutes)

**Why:** Hyperliquid's S3 bucket requires AWS credentials (you pay egress ~$5-20)

**Steps:**
1. Go to https://aws.amazon.com/free
2. Sign up (free tier works)
3. IAM Console → Users → Create user
4. Attach policy: `AmazonS3ReadOnlyAccess`
5. Create access key → Copy credentials

### 3. Configure AWS (30 seconds)

```bash
aws configure
```

Paste:
- Access Key ID: `[your key]`
- Secret Access Key: `[your secret]`
- Region: `us-east-1`
- Format: `json`

### 4. Test Setup (1 minute)

```bash
./.aws_setup_test.sh
```

Should see: `✓ All checks passed!`

### 5. Download Historical Data

#### Option A: Quick Test (5 minutes)

Test with 1 week of BTC data first:

```bash
python scripts/fetch_s3_historical.py \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --coins BTC

# Validate
python scripts/fetch_s3_historical.py --validate --coins BTC
```

#### Option B: Full Historical (Hours/Days)

Download everything from Oct 2023 to where API data starts:

```bash
# Run in background (survives SSH disconnect)
nohup python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    > s3_download.log 2>&1 &

# Monitor
tail -f s3_download.log

# Check progress
cat data/s3_historical/progress/download_state.json
```

**Safe to interrupt:** Ctrl+C anytime. Rerun to resume.

### 6. After Download Complete

Merge S3 + API data in your notebooks:

```python
import pandas as pd

# Load both sources (4-hour bars)
s3_candles = pd.read_csv('data/s3_historical/candles/BTC_candles_4h.csv',
                          parse_dates=['datetime'])
api_candles = pd.read_csv('data/market_data/BTC_candles_4h.csv',
                           parse_dates=['datetime'])

# Merge (S3 has older data, API has recent)
combined = pd.concat([s3_candles, api_candles]) \
    .drop_duplicates(subset=['datetime']) \
    .sort_values('datetime') \
    .reset_index(drop=True)

# Now you have full historical coverage!
# Oct 2023 → Present
```

## Key Points

**Resumable:** Press Ctrl+C anytime. Progress saved every 10 items.

**Disk-Efficient:** Downloads 1 file → processes → deletes. Only stores final candles.

**Separate Storage:**
- API data: `data/market_data/` (recent ~7 months)
- S3 data: `data/s3_historical/` (Oct 2023 - March 2025)

**Cost:** ~$0.09/GB AWS egress. Total: $5-20 depending on coins/dates.

## Troubleshooting

Problem: "Access Denied"
→ `aws configure` with valid credentials

Problem: "lz4 not found"
→ `sudo apt install lz4`

Problem: "No coins found"
→ Use `--coins BTC ETH SOL` to specify

## Documentation

- `S3_SETUP_README.md` - Quick start guide
- `docs/S3_HISTORICAL_DATA.md` - Comprehensive documentation
- `.aws_setup_test.sh` - Test script

## After You Have Data

1. Run notebooks to generate signals
2. Build PCA factors with full history
3. Optimize holding period H* with proper walk-forward
4. Backtest with transaction costs
5. Iterate on strategy

## Questions?

Check:
- `docs/S3_HISTORICAL_DATA.md`
- [Hyperliquid Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
- `CLAUDE.md` for development workflow
