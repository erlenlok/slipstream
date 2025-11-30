#!/usr/bin/env python3
"""
Configuration test script to verify Gradient strategy configuration and wallet selection.
"""

import os
import sys
from pathlib import Path

# Add src to path to import the Gradient config
sys.path.insert(0, str(Path(__file__).parent / "src"))

from slipstream.strategies.gradient.live.config import load_config, validate_config

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

    # Check environment variables
    api_key = os.getenv('HYPERLIQUID_API_KEY')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET')
    main_wallet = os.getenv('HYPERLIQUID_MAIN_WALLET')
    gradient_wallet = os.getenv('HYPERLIQUID_GRADIENT_WALLET')
    brawler_wallet = os.getenv('HYPERLIQUID_BRAWLER_WALLET')

    print(f"Environment variables:")
    print(f"  HYPERLIQUID_API_KEY: {api_key[:8] if api_key else 'None'}...")
    print(f"  HYPERLIQUID_API_SECRET: {'✓' if api_secret else '✗'}")
    print(f"  HYPERLIQUID_MAIN_WALLET: {main_wallet}")
    print(f"  HYPERLIQUID_GRADIENT_WALLET: {gradient_wallet}")
    print(f"  HYPERLIQUID_BRAWLER_WALLET: {brawler_wallet}")

    # Check if config file exists
    config_path = "config/gradient_live.json"
    if not Path(config_path).exists():
        print(f"\n⚠️  Config file {config_path} does not exist!")
        print("This might be why the rebalance isn't working properly.")
        
        # Try to find other possible config files
        config_dir = Path("config")
        if config_dir.exists():
            print(f"\nLooking for other config files in {config_dir}:")
            for file in config_dir.glob("*.json"):
                print(f"  - {file.name}")
    else:
        print(f"\n✓ Found config file: {config_path}")
        
        try:
            print(f"\nLoading configuration...")
            config = load_config(config_path)
            
            print(f"\n✓ Configuration loaded successfully!")
            print(f"Using API endpoint: {config.api_endpoint}")
            print(f"Dry run mode: {config.dry_run}")
            print(f"Capital: ${config.capital_usd:,.2f}")
            
            # Validate config
            validate_config(config)
            
        except Exception as e:
            print(f"\n❌ Error loading configuration: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTesting wallet selection logic as implemented in execution.py:")
    
    # This mimics what happens in _prepare_hyperliquid_context
    target_wallet = os.getenv("HYPERLIQUID_GRADIENT_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")
    print(f"Selected target wallet: {target_wallet}")
    
    if target_wallet == gradient_wallet:
        print("✅ Gradient wallet selected correctly!")
    elif target_wallet == main_wallet:
        print("⚠️  Main wallet selected (gradient wallet not available)")
    else:
        print("❌ Neither gradient nor main wallet selected!")
    
    # Also test the _resolve_main_wallet function logic (used for fetching positions)
    main_wallet_resolve = os.getenv("HYPERLIQUID_GRADIENT_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")
    print(f"For fetching positions, using wallet: {main_wallet_resolve}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"• API key/secret available: {'Yes' if api_key and api_secret else 'No'}")
    print(f"• Gradient wallet configured: {'Yes' if gradient_wallet else 'No'}")
    print(f"• Target wallet for trading: {target_wallet}")
    print(f"• Code should trade on gradient wallet: {'Yes' if target_wallet == gradient_wallet else 'No'}")
    print(f"• Config file exists: {'Yes' if Path(config_path).exists() else 'No'}")
    print(f"="*60)


if __name__ == "__main__":
    main()