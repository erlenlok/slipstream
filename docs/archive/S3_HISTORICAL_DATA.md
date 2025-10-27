# S3 Historical Data Guide

This guide explains how to fetch historical OHLCV candles from Hyperliquid's S3 archive.

## Overview

Hyperliquid provides historical L2 orderbook snapshots via S3, but **NOT** pre-computed candles. We download L2 snapshots, extract mid-prices, aggregate to 1-hour candles, then immediately delete the raw data to save disk space.

**Key Features:**
- ✅ Resumable - tracks progress, safe to interrupt and restart
- ✅ Disk-efficient - processes and deletes files one at a time
- ✅ Validates against existing API data
- ✅ Separate from API data (`data/s3_historical/` vs `data/market_data/`)

## Prerequisites

### 1. Install AWS CLI

Already installed via uv:
```bash
uv pip install awscli
```

### 2. Install LZ4 Decompression Tool

```bash
sudo apt update
sudo apt install lz4
```

### 3. Configure AWS Credentials

You need an AWS account (free tier works) to download from S3:

```bash
# Option 1: Interactive configuration
aws configure
# Enter:
#   AWS Access Key ID: [your key]
#   AWS Secret Access Key: [your secret]
#   Default region: us-east-1
#   Default output format: json

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID=your_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_here
```

**To get AWS credentials:**
1. Go to https://aws.amazon.com
2. Sign up for free tier (no credit card required for this use case)
3. Go to IAM → Users → Create user
4. Attach policy: `AmazonS3ReadOnlyAccess`
5. Create access key → Copy credentials

**Cost:** ~$0.09/GB for data transfer. Expect ~$5-10 total for full historical download (2023-present).

## Usage

### Determine Date Range

First, check where your API data ends:

```bash
# Check most recent API candle
head -5 data/market_data/BTC_candles_1h.csv
```

You want S3 data to cover the period BEFORE your API data starts.

Example: If API data starts March 28, 2025, fetch S3 from Oct 2023 to March 27, 2025.

### Download Historical Data

```bash
# Fetch historical data (this will take hours/days depending on range)
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27

# Process specific coins only
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    --coins BTC ETH SOL

# Resume after interruption (automatically skips completed)
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27
```

**The script will:**
1. Check `data/s3_historical/progress/download_state.json` for already-processed items
2. Download one L2 snapshot file at a time
3. Parse mid-prices and create 1h OHLCV candle
4. Append to `data/s3_historical/candles/{COIN}_candles_1h.csv`
5. Delete the raw L2 file immediately
6. Update progress tracker
7. Repeat for all hours and all coins

**Safe to interrupt:** Press Ctrl+C anytime. Progress is saved every 10 items. Simply rerun the same command to resume.

### Validate Against API Data

After downloading, validate data quality by comparing overlapping periods:

```bash
python scripts/fetch_s3_historical.py --validate

# Validate specific coins
python scripts/fetch_s3_historical.py --validate --coins BTC ETH
```

This compares close prices between S3 candles and API candles for overlapping timestamps. Should show <1% difference.

## Data Format

S3 candles are saved in the same format as API candles:

```
data/s3_historical/
  candles/
    BTC_candles_1h.csv   # datetime,open,high,low,close,volume
    ETH_candles_1h.csv
    SOL_candles_1h.csv
    ...
  progress/
    download_state.json  # Resumption state
```

**Note:** `volume` column will be NaN since L2 snapshots don't contain trade volume.

## Merging with API Data

Once you have both S3 historical data and API recent data, merge them:

```python
# Load both
s3_candles = pd.read_csv('data/s3_historical/candles/BTC_candles_1h.csv', parse_dates=['datetime'])
api_candles = pd.read_csv('data/market_data/BTC_candles_1h.csv', parse_dates=['datetime'])

# Concatenate and deduplicate
combined = pd.concat([s3_candles, api_candles]).drop_duplicates(subset=['datetime']).sort_values('datetime')

# Use combined for backtesting
```

## Troubleshooting

### Access Denied Error

```
An error occurred (AccessDenied) when calling the ListObjectsV2 operation
```

**Solution:** Configure AWS credentials (see Prerequisites section 3)

### LZ4 Not Found

```
lz4 tool not found. Install with: sudo apt install lz4
```

**Solution:** Run `sudo apt install lz4`

### Missing Data for Certain Hours

Some coins may not have L2 snapshots for all hours (delisted, new listings, data gaps).

**Solution:** This is expected. The script will skip missing files and continue.

### Disk Space Issues

If `/tmp` fills up:

```bash
# Use different temp directory
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    --temp-dir /path/to/larger/disk
```

### Slow Download Speed

S3 downloads can be slow. Consider:
- Fetching smaller date ranges
- Limiting to specific coins with `--coins`
- Running overnight/over weekend

## Cost Estimation

**AWS S3 egress pricing:** ~$0.09/GB

**Estimated data size:**
- 1 coin, 1 hour, 1 L2 snapshot: ~50-500KB (compressed)
- 1 coin, 1 year: ~4-40GB raw (we delete after processing)
- 100 coins, 1.5 years: ~600-6000GB raw (scary, but we process 1 file at a time!)

**Expected total cost:** $5-20 depending on date range and number of coins

**Optimization:** Start with just BTC/ETH/SOL to validate the approach before downloading full universe.

## Example Workflow

```bash
# 1. Install dependencies
sudo apt install lz4
uv pip install awscli

# 2. Configure AWS
aws configure

# 3. Test with small range and few coins first
python scripts/fetch_s3_historical.py \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --coins BTC ETH

# 4. Validate
python scripts/fetch_s3_historical.py --validate --coins BTC ETH

# 5. If good, fetch full range
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27

# 6. Run in background with nohup (survives SSH disconnect)
nohup python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    > s3_download.log 2>&1 &

# 7. Monitor progress
tail -f s3_download.log
```

## References

- [Hyperliquid Historical Data Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
- S3 bucket: `s3://hyperliquid-archive/market_data/`
- Format: `YYYYMMDD/H/l2Book/COIN.lz4`
