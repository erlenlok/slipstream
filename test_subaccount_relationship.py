#!/usr/bin/env python3
"""
Test to check subaccount setup and authorization between wallets.
"""

import os
import sys
import requests
from pathlib import Path

def load_env_vars():
    """Load environment variables from .env.gradient file."""
    env_file = '/home/ubuntu/slipstream/.env.gradient'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
    else:
        print(f"Warning: {env_file} not found")

def main():
    print("Loading environment variables...")
    load_env_vars()

    # Get environment variables
    main_wallet = os.getenv('HYPERLIQUID_MAIN_WALLET')
    gradient_wallet = os.getenv('HYPERLIQUID_GRADIENT_WALLET')
    api_key = os.getenv('HYPERLIQUID_API_KEY')

    print(f"Main Wallet: {main_wallet}")
    print(f"Gradient Wallet: {gradient_wallet}")
    print(f"API Key Wallet: {api_key}")

    # Check if gradient wallet is a subaccount of main wallet
    print(f"\n--- CHECKING SUBACCOUNTS ---")
    
    # Check main wallet's subaccounts
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "subAccounts",
                "user": main_wallet
            },
            timeout=10
        )
        subaccounts = response.json()
        print(f"Subaccounts of main wallet {main_wallet[:8]}... :")
        if isinstance(subaccounts, list):
            for i, sub in enumerate(subaccounts):
                print(f"  {i+1}. Name: {sub.get('name', 'N/A')}, Address: {sub.get('subAccountUser', 'N/A')}")
                if sub.get('subAccountUser') == gradient_wallet:
                    print(f"     ^^^ This is the gradient wallet!")
        else:
            print(f"  No subaccounts found or error: {subaccounts}")
    except Exception as e:
        print(f"Error checking subaccounts: {e}")

    # Check if gradient wallet is a subaccount by querying it directly
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "metaAndAssetCtxs",
                "user": gradient_wallet  # This might fail if it's not a valid account
            },
            timeout=10
        )
        result = response.json()
        print(f"\nDirect query of gradient wallet {gradient_wallet[:8]}... : {type(result)}")
        if isinstance(result, dict) and 'universe' in result:
            print("  ✓ Gradient wallet appears to be a valid main account (not a subaccount)")
        elif isinstance(result, list):
            print(f"  Response type: list with {len(result)} elements")
            if len(result) >= 2 and isinstance(result[0], dict) and 'universe' in result[0]:
                print("  ✓ Gradient wallet appears to be a valid main account")
            else:
                print("  ? Unusual response structure")
        else:
            print(f"  Response: {result}")
    except Exception as e:
        print(f"Error querying gradient wallet directly: {e}")

    # Check if main wallet is also a subaccount of something else
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "subAccounts",
                "user": gradient_wallet
            },
            timeout=10
        )
        result = response.json()
        print(f"\nSubaccounts of gradient wallet {gradient_wallet[:8]}... : {result}")
    except Exception as e:
        print(f"Error checking gradient wallet's subaccounts: {e}")

    # Check if the gradient wallet is its own main/master account
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "clearinghouseState",
                "user": gradient_wallet
            },
            timeout=10
        )
        result = response.json()
        print(f"\nGradient wallet clearinghouse state query:")
        if 'marginSummary' in result:
            account_value = result['marginSummary']['accountValue']
            print(f"  ✓ Success - gradient wallet is a valid account with ${account_value} value")
            # Check if it has a master field
            if 'master' in result:
                print(f"  Master account: {result.get('master', 'N/A')}")
            else:
                print("  No master field found - likely a main account")
        else:
            print(f"  Response: {result}")
    except Exception as e:
        print(f"Error querying gradient wallet state: {e}")

    # Check main wallet state too for comparison
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={
                "type": "clearinghouseState",
                "user": main_wallet
            },
            timeout=10
        )
        result = response.json()
        print(f"\nMain wallet clearinghouse state query:")
        if 'marginSummary' in result:
            account_value = result['marginSummary']['accountValue']
            print(f"  ✓ Success - main wallet is a valid account with ${account_value} value")
            # Check if it has a master field
            if 'master' in result:
                print(f"  Master account: {result.get('master', 'N/A')}")
            else:
                print("  No master field found - likely a main account")
        else:
            print(f"  Response: {result}")
    except Exception as e:
        print(f"Error querying main wallet state: {e}")

    print(f"\n--- ANALYSIS ---")
    print("The gradient wallet (0x28315f...) appears to be a separate main account")
    print("not a subaccount of the main wallet (0xFd5cf6...).")
    print()
    print("The problem is that when using account_address parameter in Exchange():")
    print("- Both wallets need to be related (one is subaccount of other)")
    print("- OR the API wallet needs special permissions to trade on both")
    print("- OR they need to be properly linked in Hyperliquid's system")
    print()
    print("Since they appear to be separate main accounts, the account_address")
    print("parameter may not work as expected, causing trades to default to")
    print("the wallet associated with the API credentials.")
    print()
    print("SOLUTIONS:")
    print("1. Convert gradient wallet to a proper subaccount of main wallet")
    print("2. Use separate API credentials for each wallet")
    print("3. Check if the API wallet has proper permissions")


if __name__ == "__main__":
    main()