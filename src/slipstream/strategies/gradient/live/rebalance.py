#!/usr/bin/env python3
"""
Main entry point for Gradient strategy rebalancing.

This script is called by cron every 4 hours to rebalance the portfolio.
"""

import sys
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config, validate_config
from .data import fetch_live_data, compute_live_signals, validate_market_data, validate_signals
from .portfolio import construct_target_portfolio
from .execution import (
    get_current_positions,
    execute_rebalance_with_stages,
    validate_execution_results
)
from .performance import PerformanceTracker, compute_signal_tracking_metrics
from .notifications import send_telegram_rebalance_alert_sync
from ..sensitivity import compute_log_returns


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

        signal_tracking_summary = None
        signal_timestamp_value = signals["signal_timestamp"].iloc[0]
        if hasattr(signal_timestamp_value, "to_pydatetime"):
            signal_timestamp = signal_timestamp_value.to_pydatetime()
        else:
            signal_timestamp = datetime.fromisoformat(str(signal_timestamp_value))

        previous_signals = tracker.get_latest_signals()
        if previous_signals:
            try:
                forecast_timestamp = datetime.fromisoformat(previous_signals["timestamp"])
            except ValueError:
                forecast_timestamp = datetime.fromisoformat(previous_signals["timestamp"].split(".")[0])

            panel_with_returns = compute_log_returns(market_data["panel"].copy())
            realized_slice = panel_with_returns[
                panel_with_returns["timestamp"] == signal_timestamp_value
            ]

            if realized_slice.empty:
                logger.warning(
                    "Signal tracking skipped: no realized returns available for %s",
                    signal_timestamp.isoformat(timespec="minutes"),
                )
            else:
                realized_map = (
                    realized_slice.set_index("asset")["log_return"].to_dict()
                )
                asset_records: List[Dict[str, Any]] = []

                for entry in previous_signals.get("signals", []):
                    asset = entry.get("asset")
                    if not asset:
                        continue
                    forecast = entry.get("momentum_score")
                    realized = realized_map.get(asset)
                    if forecast is None or realized is None:
                        continue
                    try:
                        forecast_val = float(forecast)
                        realized_val = float(realized)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(forecast_val) or math.isnan(realized_val):
                        continue

                    asset_records.append(
                        {
                            "asset": asset,
                            "forecast": forecast_val,
                            "realized": realized_val,
                            "abs_error": abs(realized_val - forecast_val),
                            "squared_error": (realized_val - forecast_val) ** 2,
                        }
                    )

                if asset_records:
                    metrics = compute_signal_tracking_metrics(asset_records)
                    tracker.log_signal_performance(
                        evaluation_timestamp=signal_timestamp,
                        forecast_timestamp=forecast_timestamp,
                        metrics=metrics,
                        asset_records=asset_records,
                    )
                    corr_str = (
                        f"{metrics['pearson_corr']:.3f}"
                        if metrics.get("pearson_corr") is not None
                        else "n/a"
                    )
                    hit_str = (
                        f"{metrics['hit_rate']:.1%}"
                        if metrics.get("hit_rate") is not None
                        else "n/a"
                    )
                    logger.info(
                        "Signal tracking (%s → %s): corr=%s, hit_rate=%s, MAE=%.4f, RMSE=%.4f",
                        forecast_timestamp.isoformat(timespec="minutes"),
                        signal_timestamp.isoformat(timespec="minutes"),
                        corr_str,
                        hit_str,
                        metrics["mae"],
                        metrics["rmse"],
                    )
                    signal_tracking_summary = {
                        "forecast_timestamp": forecast_timestamp.isoformat(),
                        "evaluation_timestamp": signal_timestamp.isoformat(),
                        "metrics": metrics,
                        "n_assets": len(asset_records),
                    }
                else:
                    logger.info(
                        "Signal tracking skipped: no overlapping assets between forecast at %s and realized returns at %s",
                        forecast_timestamp.isoformat(timespec="minutes"),
                        signal_timestamp.isoformat(timespec="minutes"),
                    )

        tracker.log_signals(signal_timestamp, signals)

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
        target_order_count = execution_results.get("target_order_count", len(target_positions))
        stage1_asset_fills = execution_results.get("stage1_asset_fills", execution_results["stage1_filled"])
        stage2_asset_fills = execution_results.get("stage2_asset_fills", execution_results["stage2_filled"])

        logger.info("Rebalance complete!")
        logger.info(
            f"Stage 1 fills: {stage1_asset_fills}/{target_order_count} assets "
            f"(${execution_results.get('stage1_fill_notional', 0.0):,.2f})"
        )
        logger.info(
            f"Stage 2 fills: {stage2_asset_fills}/{target_order_count} assets "
            f"(${execution_results.get('stage2_fill_notional', 0.0):,.2f})"
        )
        logger.info(f"Total turnover: ${execution_results['total_turnover']:,.2f}")

        passive_fill_rate = execution_results.get("passive_fill_rate")
        aggressive_fill_rate = execution_results.get("aggressive_fill_rate")
        if passive_fill_rate is not None:
            logger.info(f"Passive fill rate: {passive_fill_rate:.1%}")
        if aggressive_fill_rate is not None:
            logger.info(f"Aggressive fallback rate: {aggressive_fill_rate:.1%}")

        passive_stats = execution_results.get("passive_slippage", {}) or {}
        aggressive_stats = execution_results.get("aggressive_slippage", {}) or {}
        total_stats = execution_results.get("total_slippage", {}) or {}

        def _fmt_slippage(stats: dict) -> str:
            if not stats or stats.get("total_usd", 0) == 0 or stats.get("weighted_bps") is None:
                return "n/a"
            return f"{stats['weighted_bps']:.2f} bps on ${stats['total_usd']:,.2f}"

        logger.info(f"Passive slippage: {_fmt_slippage(passive_stats)}")
        logger.info(f"Aggressive slippage: {_fmt_slippage(aggressive_stats)}")
        logger.info(f"Total slippage: {_fmt_slippage(total_stats)}")

        # Log to performance tracker
        rebalance_timestamp = datetime.now()
        tracker.log_rebalance(
            timestamp=rebalance_timestamp,
            target_positions=target_positions,
            execution_results=execution_results,
            config=config
        )

        # Fetch and log ACTUAL positions from Hyperliquid (not targets)
        try:
            actual_positions = get_current_positions(config)
            logger.info(f"Actual positions after rebalance: {len(actual_positions)} assets")
            tracker.log_positions(
                timestamp=rebalance_timestamp,
                positions=actual_positions,
                config=config
            )
        except Exception as e:
            logger.warning(f"Failed to fetch actual positions for logging: {e}")
            # Fallback to target positions if fetch fails
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
                "total_orders": target_order_count,
                "passive_fill_rate": execution_results.get("passive_fill_rate"),
                "aggressive_fill_rate": execution_results.get("aggressive_fill_rate"),
                "passive_slippage": execution_results.get("passive_slippage"),
                "aggressive_slippage": execution_results.get("aggressive_slippage"),
                "total_slippage": execution_results.get("total_slippage"),
                "stage2_highlights": execution_results.get("stage2_highlights", []),
                "signal_tracking": signal_tracking_summary,
                "stage1_asset_fills": stage1_asset_fills,
                "stage2_asset_fills": stage2_asset_fills,
                "stage1_fill_notional": execution_results.get("stage1_fill_notional"),
                "stage2_fill_notional": execution_results.get("stage2_fill_notional"),
                "total_target_usd": execution_results.get("total_target_usd"),
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
