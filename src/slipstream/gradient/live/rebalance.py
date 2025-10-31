#!/usr/bin/env python3
"""
Main entry point for Gradient strategy rebalancing.

This script is called by cron every 4 hours to rebalance the portfolio.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

from .config import load_config, validate_config
from .data import fetch_live_data, compute_live_signals, validate_market_data, validate_signals
from .portfolio import construct_target_portfolio
from .execution import (
    get_current_positions,
    execute_rebalance_with_stages,
    validate_execution_results
)
from .performance import PerformanceTracker
from .notifications import send_telegram_rebalance_alert_sync


def setup_logging(config):
    """Configure logging to file and console."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"rebalance_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


def run_rebalance():
    """
    Execute a single rebalance cycle.

    Steps:
    1. Load configuration
    2. Fetch latest market data
    3. Compute momentum signals
    4. Construct target portfolio
    5. Get current positions
    6. Execute rebalance orders
    7. Log results
    """
    try:
        # Load and validate configuration
        config = load_config()
        validate_config(config)
        logger = setup_logging(config)

        # Initialize performance tracker
        tracker = PerformanceTracker(log_dir=config.log_dir)

        logger.info("=" * 80)
        logger.info("Starting Gradient rebalance cycle")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Capital: ${config.capital_usd:,.2f}")
        logger.info(f"Concentration: {config.concentration_pct}%")
        logger.info(f"Dry-run: {config.dry_run}")
        logger.info("=" * 80)

        # Step 1: Fetch data
        logger.info("Fetching latest market data...")
        market_data = fetch_live_data(config)
        validate_market_data(market_data)
        logger.info(f"Fetched data for {len(market_data['assets'])} assets")

        # Step 2: Compute signals
        logger.info("Computing momentum signals...")
        signals = compute_live_signals(market_data, config)
        validate_signals(signals, config)
        logger.info(f"Computed signals for {len(signals)} assets")

        # Step 3: Construct target portfolio
        logger.info("Constructing target portfolio...")
        target_positions = construct_target_portfolio(signals, config)
        n_long = sum(1 for p in target_positions.values() if p > 0)
        n_short = sum(1 for p in target_positions.values() if p < 0)
        total_long = sum(p for p in target_positions.values() if p > 0)
        total_short = abs(sum(p for p in target_positions.values() if p < 0))
        logger.info(f"Target: {n_long} long, {n_short} short positions")
        logger.info(f"Target exposure: ${total_long:,.2f} long, ${total_short:,.2f} short")

        # Step 4: Get current positions
        logger.info("Fetching current positions...")
        current_positions = get_current_positions(config)
        logger.info(f"Current: {len(current_positions)} positions")

        # Step 5: Execute rebalance with two-stage limit→market execution
        logger.info("Executing rebalance (two-stage: limit → market)...")
        execution_results = execute_rebalance_with_stages(
            target_positions,
            current_positions,
            config
        )

        # Validate and log results
        validate_execution_results(execution_results, config)
        logger.info("Rebalance complete!")
        logger.info(f"Stage 1 fills: {execution_results['stage1_filled']}")
        logger.info(f"Stage 2 fills: {execution_results['stage2_filled']}")
        logger.info(f"Total turnover: ${execution_results['total_turnover']:,.2f}")

        # Log to performance tracker
        rebalance_timestamp = datetime.now()
        tracker.log_rebalance(
            timestamp=rebalance_timestamp,
            target_positions=target_positions,
            execution_results=execution_results,
            config=config
        )
        tracker.log_positions(
            timestamp=rebalance_timestamp,
            positions=target_positions,
            config=config
        )

        # Send Telegram notification
        if config.alerts_enabled:
            logger.info("Sending Telegram notification...")
            rebalance_data = {
                "timestamp": rebalance_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "n_long": n_long,
                "n_short": n_short,
                "n_positions": len(target_positions),
                "total_turnover": execution_results['total_turnover'],
                "stage1_filled": execution_results['stage1_filled'],
                "stage2_filled": execution_results['stage2_filled'],
                "errors": len(execution_results.get('errors', [])),
                "dry_run": config.dry_run,
            }
            telegram_success = send_telegram_rebalance_alert_sync(rebalance_data, config)
            if telegram_success:
                logger.info("✓ Telegram notification sent")
            else:
                logger.warning("✗ Failed to send Telegram notification")

        if config.dry_run:
            logger.warning("DRY-RUN MODE: No actual orders were placed")

        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"FATAL ERROR during rebalance: {e}", exc_info=True)
        # TODO: Send alert
        return 1


def main():
    """Entry point for command-line execution."""
    exit_code = run_rebalance()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
