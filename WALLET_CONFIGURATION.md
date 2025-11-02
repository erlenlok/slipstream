# Hyperliquid Wallet Configuration

## The Issue (Now Fixed!)

Previously, the system was confused about which wallet address to use when fetching positions. This caused positions to appear as "not found" even though trades were executing successfully.

## Understanding Hyperliquid's Wallet Architecture

Hyperliquid uses **two separate wallet addresses** for trading:

### 1. **API Vault** (Sub-account for signing)
- **Environment Variable**: `HYPERLIQUID_API_KEY`
- **Your Address**: `0x998c0B58193faca878B55aE29165b68167A1BD30`
- **Purpose**: Signs API requests and executes trades on behalf of the main wallet
- **Derived from**: `HYPERLIQUID_API_SECRET` (private key)
- **Has positions**: ❌ NO - this is just a signing vault with $0 balance

### 2. **Main Wallet** (Where funds and positions live)
- **Environment Variable**: `HYPERLIQUID_MAIN_WALLET`
- **Your Address**: `0xFd5cf66Cf037140A477419B89656E5F735fa82f4`
- **Purpose**: Holds your actual USDC balance and perpetual positions
- **Visible in**: Hyperliquid UI, account balances, position queries
- **Has positions**: ✅ YES - this is where all your positions and funds are

### How They Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading Flow                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Exchange.market_open()                                  │
│     ↓                                                       │
│  2. Signed with API Vault (0x998c...)                      │
│     ↓                                                       │
│  3. Trade executes ON BEHALF OF Main Wallet (0xFd5c...)    │
│     ↓                                                       │
│  4. Position appears in Main Wallet                         │
│                                                             │
│  5. get_current_positions() must query MAIN WALLET!         │
│     (not API vault)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Required Environment Variables

Add these to your environment (`.bashrc`, crontab, or systemd service):

```bash
# API vault (signs trades)
export HYPERLIQUID_API_SECRET="0xc414e72b..."  # Your private key
export HYPERLIQUID_API_KEY="0x998c0B58193faca878B55aE29165b68167A1BD30"

# Main wallet (holds positions) - NEW!
export HYPERLIQUID_MAIN_WALLET="0xFd5cf66Cf037140A477419B89656E5F735fa82f4"
```

## The Fix

Updated `src/slipstream/gradient/live/execution.py`:

### Before (Broken):
```python
def get_current_positions(config):
    api_key = os.getenv("HYPERLIQUID_API_KEY")  # ❌ Wrong! This is the vault
    user_state = info.user_state(api_key)       # ❌ Queries vault (empty)
```

### After (Fixed):
```python
def get_current_positions(config):
    main_wallet = os.getenv("HYPERLIQUID_MAIN_WALLET")     # ✅ Correct!
    state = _fetch_clearinghouse_state(base_url, main_wallet)  # ✅ Queries main wallet
```

## Verification

Run the test script to verify everything works:

```bash
export HYPERLIQUID_MAIN_WALLET="0xFd5cf66Cf037140A477419B89656E5F735fa82f4"
python test_full_workflow.py
```

Expected output:
```
✓ Order FILLED!
✓ SUCCESS! BTC position found: $19.87
🎉 Full workflow test PASSED!
```

Clean up test position:
```bash
python test_full_workflow.py --close
```

## Adding to Cron

Update your crontab to include the HYPERLIQUID_MAIN_WALLET variable:

```bash
crontab -e
```

Add at the top:
```cron
HYPERLIQUID_API_SECRET=0xc414e72b...
HYPERLIQUID_API_KEY=0x998c0B58193faca878B55aE29165b68167A1BD30
HYPERLIQUID_MAIN_WALLET=0xFd5cf66Cf037140A477419B89656E5F735fa82f4

# Your rebalance job
0 */4 * * * cd /root/slipstream && /root/.local/bin/uv run python -m slipstream.gradient.live.rebalance >> /root/slipstream/logs/rebalance.log 2>&1
```

## Summary

**The confusion was**: "API key" and "wallet address" are different concepts in Hyperliquid!

- **API key** = vault that signs (but doesn't hold positions)
- **Main wallet** = where your money and positions actually live

Now the system correctly queries the main wallet for positions while still using the API vault to sign trades.
