#!/bin/bash
# Quick AWS setup test script

echo "=== AWS S3 Historical Data Setup Test ==="
echo ""

# Check if lz4 is installed
echo "1. Checking lz4 tool..."
if command -v lz4 &> /dev/null; then
    echo "   ✓ lz4 is installed"
else
    echo "   ✗ lz4 NOT installed"
    echo "   Install with: sudo apt install lz4"
    exit 1
fi

# Check if AWS CLI is available
echo ""
echo "2. Checking AWS CLI..."
if uv run aws --version &> /dev/null; then
    echo "   ✓ AWS CLI is available"
    uv run aws --version
else
    echo "   ✗ AWS CLI NOT available"
    exit 1
fi

# Check if AWS credentials are configured
echo ""
echo "3. Checking AWS credentials..."
if uv run aws sts get-caller-identity &> /dev/null; then
    echo "   ✓ AWS credentials configured"
    uv run aws sts get-caller-identity
else
    echo "   ✗ AWS credentials NOT configured"
    echo "   Run: aws configure"
    echo "   Or set: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi

# Test S3 access to Hyperliquid bucket
echo ""
echo "4. Testing S3 access to Hyperliquid bucket..."
echo "   Attempting to download a small test file..."

TEST_FILE="20231023/10/l2Book/BTC.lz4"
S3_PATH="s3://hyperliquid-archive/market_data/$TEST_FILE"

uv run aws s3 cp "$S3_PATH" /tmp/test_btc.lz4 --request-payer requester --quiet

if [ $? -eq 0 ]; then
    echo "   ✓ Successfully downloaded test file"

    # Test decompression
    echo ""
    echo "5. Testing LZ4 decompression..."
    lz4 -d -c /tmp/test_btc.lz4 > /tmp/test_btc_decompressed 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "   ✓ Successfully decompressed"

        # Show first few lines
        echo ""
        echo "Sample data (first 3 lines):"
        head -3 /tmp/test_btc_decompressed 2>/dev/null | sed 's/^/   /'

        # Cleanup
        rm -f /tmp/test_btc.lz4 /tmp/test_btc_decompressed

        echo ""
        echo "=== ✓ All checks passed! ===="
        echo ""
        echo "You're ready to run:"
        echo "  python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2024-01-01 --coins BTC"
        exit 0
    else
        echo "   ✗ Decompression failed"
        exit 1
    fi
else
    echo "   ✗ Failed to download from S3"
    echo "   This might mean:"
    echo "     - AWS credentials don't have S3 read permission"
    echo "     - Network issues"
    echo "     - File doesn't exist at this path"
    exit 1
fi
