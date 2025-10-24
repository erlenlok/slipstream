# S3 Historical Data - Quick Start

This README gets you set up to download full historical data from Hyperliquid's S3 archive.

## The Problem

Hyperliquid's API only provides ~5,000 recent candles (~7 months for 1h data). For serious backtesting, you need years of data.

## The Solution

Download L2 orderbook snapshots from S3, extract mid-prices, aggregate to OHLCV candles, delete raw data immediately to save disk.

## Quick Setup (5 minutes)

### 1. Install LZ4 Decompression

```bash
sudo apt update
sudo apt install lz4
```

### 2. Get AWS Credentials

You need a free AWS account:

1. Go to https://aws.amazon.com/free
2. Sign up (free tier, no credit card needed for this)
3. Go to IAM Console → Users → Create User
4. Attach policy: `AmazonS3ReadOnlyAccess`
5. Security credentials → Create access key
6. Copy `Access Key ID` and `Secret Access Key`

### 3. Configure AWS

```bash
aws configure
```

Enter:
- AWS Access Key ID: `[paste your key]`
- AWS Secret Access Key: `[paste your secret]`
- Default region name: `us-east-1`
- Default output format: `json`

### 4. Test Setup

```bash
./.aws_setup_test.sh
```

Should output: `✓ All checks passed!`

## Download Historical Data

### Option A: Test First (Recommended)

Start with a small test to validate:

```bash
# Download 1 week of BTC data (takes ~5 min)
python scripts/fetch_s3_historical.py \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --coins BTC

# Validate
python scripts/fetch_s3_historical.py --validate --coins BTC
```

### Option B: Full Historical Download

```bash
# Download all coins from Oct 2023 to where API data starts
# This will take HOURS/DAYS depending on number of coins
# Cost: ~$5-20 in AWS egress fees

python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27

# Run in background (survives SSH disconnect)
nohup python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    > s3_download.log 2>&1 &

# Monitor progress
tail -f s3_download.log
```

### Resumable

**Safe to interrupt!** Press Ctrl+C anytime. Progress is saved. Rerun the same command to resume.

## Key Features

✅ **Resumable** - Tracks progress in `data/s3_historical/progress/download_state.json`
✅ **Disk-efficient** - Downloads, processes, deletes one file at a time
✅ **Validates** - Compare against API data to ensure quality
✅ **Separate storage** - `data/s3_historical/` vs `data/market_data/`

## Output

Candles saved to:
```
data/s3_historical/candles/
    BTC_candles_1h.csv
    ETH_candles_1h.csv
    SOL_candles_1h.csv
    ...
```

Format: `datetime,open,high,low,close,volume`

Note: `volume` will be NaN (L2 snapshots don't have volume)

## Cost

**AWS S3 Data Transfer:** ~$0.09/GB

**Estimates:**
- 1 week, 1 coin: ~$0.05
- 1 year, 10 coins: ~$5
- 1.5 years, 100 coins: ~$15-20

## Troubleshooting

### "Access Denied"
→ Run `aws configure` and enter valid credentials

### "lz4: command not found"
→ Run `sudo apt install lz4`

### "No coins found"
→ Either specify `--coins BTC ETH` or ensure `data/market_data/` has some candles

### Progress is slow
→ Normal! S3 downloads are throttled. Consider:
- Running overnight
- Limiting to specific coins first
- Using `nohup` to run in background

## Next Steps

After downloading S3 data:

1. **Validate:** `python scripts/fetch_s3_historical.py --validate`
2. **Merge with API data:** In notebooks, concatenate S3 + API candles
3. **Build PCA factors:** Use merged data for `scripts/build_pca_factor.py`
4. **Backtest:** Full historical coverage for strategy validation

## Full Documentation

See `docs/S3_HISTORICAL_DATA.md` for comprehensive guide.

## Support

Questions? Check:
- `docs/S3_HISTORICAL_DATA.md` - Detailed guide
- [Hyperliquid Historical Data Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
