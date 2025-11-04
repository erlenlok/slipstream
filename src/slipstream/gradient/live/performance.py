"""Performance tracking and aggregation for live trading."""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


class PerformanceTracker:
    """Track and aggregate trading performance metrics."""

    def __init__(self, log_dir: str = "/var/log/gradient"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rebalance_log = self.log_dir / "rebalance_history.jsonl"
        self.positions_log = self.log_dir / "positions_history.jsonl"
        self.signals_log = self.log_dir / "signal_history.jsonl"
        self.signal_performance_log = self.log_dir / "signal_performance.jsonl"

    def log_rebalance(
        self,
        timestamp: datetime,
        target_positions: Dict[str, float],
        execution_results: Dict[str, Any],
        config: Any
    ) -> None:
        """
        Log a rebalance event.

        Args:
            timestamp: Rebalance timestamp
            target_positions: Target positions dict
            execution_results: Execution results from execute_rebalance_with_stages
            config: Configuration object
        """
        record = {
            "timestamp": timestamp.isoformat(),
            "capital_usd": config.capital_usd,
            "n_positions": len(target_positions),
            "n_long": sum(1 for p in target_positions.values() if p > 0),
            "n_short": sum(1 for p in target_positions.values() if p < 0),
            "gross_exposure": sum(abs(p) for p in target_positions.values()),
            "net_exposure": sum(target_positions.values()),
            "stage1_filled": execution_results.get("stage1_filled", 0),
            "stage2_filled": execution_results.get("stage2_filled", 0),
            "total_orders": execution_results.get(
                "target_order_count",
                len(execution_results.get("stage1_orders", [])),
            ),
            "total_turnover": execution_results.get("total_turnover", 0),
            "errors": len(execution_results.get("errors", [])),
            "dry_run": config.dry_run,
            "stage1_asset_fills": execution_results.get("stage1_asset_fills"),
            "stage2_asset_fills": execution_results.get("stage2_asset_fills"),
            "stage1_fill_notional": execution_results.get("stage1_fill_notional"),
            "stage2_fill_notional": execution_results.get("stage2_fill_notional"),
            "total_target_usd": execution_results.get("total_target_usd"),
        }

        passive_stats = execution_results.get("passive_slippage", {}) or {}
        aggressive_stats = execution_results.get("aggressive_slippage", {}) or {}
        total_stats = execution_results.get("total_slippage", {}) or {}

        record.update(
            {
                "passive_fill_rate": execution_results.get("passive_fill_rate"),
                "aggressive_fill_rate": execution_results.get("aggressive_fill_rate"),
                "passive_slippage_bps": passive_stats.get("weighted_bps"),
                "aggressive_slippage_bps": aggressive_stats.get("weighted_bps"),
                "total_slippage_bps": total_stats.get("weighted_bps"),
                "passive_slippage_usd": passive_stats.get("total_usd"),
                "aggressive_slippage_usd": aggressive_stats.get("total_usd"),
                "total_slippage_usd": total_stats.get("total_usd"),
                "passive_slippage_samples": passive_stats.get("count"),
                "aggressive_slippage_samples": aggressive_stats.get("count"),
            }
        )

        # Append to log
        with open(self.rebalance_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_positions(
        self,
        timestamp: datetime,
        positions: Dict[str, float],
        config: Any
    ) -> None:
        """
        Log current positions snapshot.

        Args:
            timestamp: Snapshot timestamp
            positions: Current positions dict
            config: Configuration object
        """
        record = {
            "timestamp": timestamp.isoformat(),
            "positions": positions,
            "total_value": sum(abs(p) for p in positions.values()),
            "n_positions": len(positions),
        }

        with open(self.positions_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_signals(
        self,
        timestamp: datetime,
        signals: pd.DataFrame,
    ) -> None:
        """Log the forecast signals used at a rebalance."""
        if len(signals) == 0:
            return

        record = {
            "timestamp": timestamp.isoformat(),
            "n_assets": int(len(signals)),
            "signals": [],
        }

        for row in signals.itertuples(index=False):
            record["signals"].append(
                {
                    "asset": row.asset,
                    "momentum_score": float(row.momentum_score),
                    "vol_24h": float(row.vol_24h),
                    "adv_usd": float(row.adv_usd),
                    "include_in_universe": bool(row.include_in_universe),
                }
            )

        with open(self.signals_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_latest_signals(self) -> Optional[Dict[str, Any]]:
        """Load the most recent logged signals."""
        if not self.signals_log.exists():
            return None

        last_line = ""
        with open(self.signals_log, "r") as f:
            for line in f:
                if line.strip():
                    last_line = line

        if not last_line:
            return None

        return json.loads(last_line)

    def log_signal_performance(
        self,
        evaluation_timestamp: datetime,
        forecast_timestamp: datetime,
        metrics: Dict[str, Optional[float]],
        asset_records: List[Dict[str, Any]],
    ) -> None:
        """Persist signal tracking error metrics for later analysis."""
        record = {
            "evaluation_timestamp": evaluation_timestamp.isoformat(),
            "forecast_timestamp": forecast_timestamp.isoformat(),
            "lag_hours": (evaluation_timestamp - forecast_timestamp).total_seconds() / 3600.0,
            "n_assets": len(asset_records),
            "metrics": {
                key: (float(value) if value is not None else None)
                for key, value in metrics.items()
            },
            "assets": asset_records,
        }

        with open(self.signal_performance_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_recent_rebalance(self) -> Optional[Dict[str, Any]]:
        """Get most recent rebalance record."""
        if not self.rebalance_log.exists():
            return None

        with open(self.rebalance_log, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        return json.loads(lines[-1])

    def get_rebalances_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Get all rebalances since a given timestamp."""
        if not self.rebalance_log.exists():
            return []

        records = []
        with open(self.rebalance_log, "r") as f:
            for line in f:
                record = json.loads(line)
                record_time = datetime.fromisoformat(record["timestamp"])
                if record_time >= since:
                    records.append(record)

        return records

    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get aggregated summary for a specific day.

        Args:
            date: Date to summarize (defaults to today)

        Returns:
            Dictionary with daily performance metrics
        """
        if date is None:
            date = datetime.now()

        # Get start and end of day
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        rebalances = []
        with open(self.rebalance_log, "r") as f:
            for line in f:
                record = json.loads(line)
                record_time = datetime.fromisoformat(record["timestamp"])
                if start <= record_time < end:
                    rebalances.append(record)

        if not rebalances:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "n_rebalances": 0,
                "message": "No rebalances on this day"
            }

        # Aggregate metrics
        total_turnover = sum(r["total_turnover"] for r in rebalances)
        total_errors = sum(r["errors"] for r in rebalances)
        total_stage1_fills = sum(r["stage1_filled"] for r in rebalances)
        total_stage2_fills = sum(r["stage2_filled"] for r in rebalances)
        total_orders = sum(r["total_orders"] for r in rebalances)
        total_target_usd = sum((r.get("total_target_usd") or 0.0) for r in rebalances)
        total_stage1_notional = sum((r.get("stage1_fill_notional") or 0.0) for r in rebalances)
        total_stage2_notional = sum((r.get("stage2_fill_notional") or 0.0) for r in rebalances)
        total_target_usd = sum((r.get("total_target_usd") or 0.0) for r in rebalances)
        total_stage1_notional = sum((r.get("stage1_fill_notional") or 0.0) for r in rebalances)
        total_stage2_notional = sum((r.get("stage2_fill_notional") or 0.0) for r in rebalances)

        avg_positions = sum(r["n_positions"] for r in rebalances) / len(rebalances)
        avg_gross_exposure = sum(r["gross_exposure"] for r in rebalances) / len(rebalances)
        avg_net_exposure = sum(r["net_exposure"] for r in rebalances) / len(rebalances)

        # Calculate fill rates
        if total_target_usd > 0:
            passive_fill_rate = total_stage1_notional / total_target_usd
            aggressive_fill_rate = total_stage2_notional / total_target_usd
        else:
            passive_fill_rate = total_stage1_fills / total_orders if total_orders > 0 else 0
            aggressive_fill_rate = total_stage2_fills / total_orders if total_orders > 0 else 0

        # Aggregate slippage
        passive_slip_usd = 0.0
        aggressive_slip_usd = 0.0
        total_slip_usd = 0.0
        passive_weighted_sum = 0.0
        aggressive_weighted_sum = 0.0
        passive_samples = 0
        aggressive_samples = 0

        for record in rebalances:
            ps_usd = record.get("passive_slippage_usd") or 0.0
            ag_usd = record.get("aggressive_slippage_usd") or 0.0
            passive_slip_usd += ps_usd
            aggressive_slip_usd += ag_usd
            total_slip_usd += ps_usd + ag_usd

            ps_bps = record.get("passive_slippage_bps")
            if ps_bps is not None and ps_usd > 0:
                passive_weighted_sum += ps_bps * ps_usd
            passive_samples += record.get("passive_slippage_samples") or 0

            ag_bps = record.get("aggressive_slippage_bps")
            if ag_bps is not None and ag_usd > 0:
                aggressive_weighted_sum += ag_bps * ag_usd
            aggressive_samples += record.get("aggressive_slippage_samples") or 0

        passive_slip_bps = passive_weighted_sum / passive_slip_usd if passive_slip_usd > 0 else None
        aggressive_slip_bps = (
            aggressive_weighted_sum / aggressive_slip_usd if aggressive_slip_usd > 0 else None
        )
        total_slip_bps = None
        if total_slip_usd > 0:
            total_weighted_sum = passive_weighted_sum + aggressive_weighted_sum
            total_slip_bps = total_weighted_sum / total_slip_usd

        return {
            "date": date.strftime("%Y-%m-%d"),
            "n_rebalances": len(rebalances),
            "total_turnover": total_turnover,
            "avg_turnover_per_rebalance": total_turnover / len(rebalances),
            "total_errors": total_errors,
            "avg_positions": avg_positions,
            "avg_gross_exposure": avg_gross_exposure,
            "avg_net_exposure": avg_net_exposure,
            "total_orders": total_orders,
            "passive_fills": total_stage1_fills,
            "aggressive_fills": total_stage2_fills,
            "passive_fill_rate": passive_fill_rate,
            "aggressive_fill_rate": aggressive_fill_rate,
            "total_target_usd": total_target_usd,
            "stage1_fill_notional": total_stage1_notional,
            "stage2_fill_notional": total_stage2_notional,
            "rebalances": rebalances,
            "passive_slippage_bps": passive_slip_bps,
            "aggressive_slippage_bps": aggressive_slip_bps,
            "total_slippage_bps": total_slip_bps,
            "passive_slippage_usd": passive_slip_usd,
            "aggressive_slippage_usd": aggressive_slip_usd,
            "total_slippage_usd": total_slip_usd,
            "passive_slippage_samples": passive_samples,
            "aggressive_slippage_samples": aggressive_samples,
        }

    def get_all_time_summary(self) -> Dict[str, Any]:
        """Get all-time aggregated statistics."""
        if not self.rebalance_log.exists():
            return {"message": "No rebalance history found"}

        rebalances = []
        with open(self.rebalance_log, "r") as f:
            for line in f:
                rebalances.append(json.loads(line))

        if not rebalances:
            return {"message": "No rebalances logged"}

        first_rebalance = datetime.fromisoformat(rebalances[0]["timestamp"])
        last_rebalance = datetime.fromisoformat(rebalances[-1]["timestamp"])
        days_running = (last_rebalance - first_rebalance).days + 1

        total_turnover = sum(r["total_turnover"] for r in rebalances)
        total_errors = sum(r["errors"] for r in rebalances)
        total_stage1_fills = sum(r["stage1_filled"] for r in rebalances)
        total_stage2_fills = sum(r["stage2_filled"] for r in rebalances)
        total_orders = sum(r["total_orders"] for r in rebalances)

        passive_slip_usd = 0.0
        aggressive_slip_usd = 0.0
        passive_weighted_sum = 0.0
        aggressive_weighted_sum = 0.0
        passive_samples = 0
        aggressive_samples = 0

        for record in rebalances:
            ps_usd = record.get("passive_slippage_usd") or 0.0
            ag_usd = record.get("aggressive_slippage_usd") or 0.0
            passive_slip_usd += ps_usd
            aggressive_slip_usd += ag_usd

            ps_bps = record.get("passive_slippage_bps")
            if ps_bps is not None and ps_usd > 0:
                passive_weighted_sum += ps_bps * ps_usd
            passive_samples += record.get("passive_slippage_samples") or 0

            ag_bps = record.get("aggressive_slippage_bps")
            if ag_bps is not None and ag_usd > 0:
                aggressive_weighted_sum += ag_bps * ag_usd
            aggressive_samples += record.get("aggressive_slippage_samples") or 0

        total_slip_usd = passive_slip_usd + aggressive_slip_usd
        passive_slip_bps = passive_weighted_sum / passive_slip_usd if passive_slip_usd > 0 else None
        aggressive_slip_bps = (
            aggressive_weighted_sum / aggressive_slip_usd if aggressive_slip_usd > 0 else None
        )
        total_slip_bps = None
        if total_slip_usd > 0:
            total_slip_bps = (passive_weighted_sum + aggressive_weighted_sum) / total_slip_usd

        passive_fill_rate = (
            total_stage1_notional / total_target_usd
            if total_target_usd > 0
            else (total_stage1_fills / total_orders if total_orders > 0 else 0)
        )
        aggressive_fill_rate = (
            total_stage2_notional / total_target_usd
            if total_target_usd > 0
            else (total_stage2_fills / total_orders if total_orders > 0 else 0)
        )

        return {
            "first_rebalance": first_rebalance.isoformat(),
            "last_rebalance": last_rebalance.isoformat(),
            "days_running": days_running,
            "total_rebalances": len(rebalances),
            "total_turnover": total_turnover,
            "avg_turnover_per_rebalance": total_turnover / len(rebalances),
            "total_errors": total_errors,
            "total_orders": total_orders,
            "passive_fills": total_stage1_fills,
            "aggressive_fills": total_stage2_fills,
            "passive_fill_rate": passive_fill_rate,
            "aggressive_fill_rate": aggressive_fill_rate,
            "total_target_usd": total_target_usd,
            "stage1_fill_notional": total_stage1_notional,
            "stage2_fill_notional": total_stage2_notional,
            "passive_slippage_bps": passive_slip_bps,
            "aggressive_slippage_bps": aggressive_slip_bps,
            "total_slippage_bps": total_slip_bps,
            "passive_slippage_usd": passive_slip_usd,
            "aggressive_slippage_usd": aggressive_slip_usd,
            "total_slippage_usd": total_slip_usd,
            "passive_slippage_samples": passive_samples,
            "aggressive_slippage_samples": aggressive_samples,
        }


def compute_signal_tracking_metrics(
    records: List[Dict[str, float]]
) -> Dict[str, Optional[float]]:
    """
    Calculate summary statistics comparing forecasted and realized returns.

    Args:
        records: List of {"forecast": float, "realized": float}

    Returns:
        Dictionary with correlation, MAE, RMSE, and directional hit rate.
    """
    if not records:
        return {
            "pearson_corr": None,
            "mae": None,
            "rmse": None,
            "hit_rate": None,
        }

    forecasts = np.array([float(r["forecast"]) for r in records], dtype=float)
    realized = np.array([float(r["realized"]) for r in records], dtype=float)

    diffs = realized - forecasts
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))

    hit_mask = (forecasts > 0) & (realized > 0) | (forecasts < 0) & (realized < 0)
    neutral_mask = (forecasts == 0) | (realized == 0)
    considered = len(records) - int(neutral_mask.sum())
    hit_rate = float(hit_mask.sum() / considered) if considered > 0 else None

    if len(records) > 1 and np.std(forecasts) > 0 and np.std(realized) > 0:
        corr = float(np.corrcoef(forecasts, realized)[0, 1])
    else:
        corr = None

    return {
        "pearson_corr": corr,
        "mae": mae,
        "rmse": rmse,
        "hit_rate": hit_rate,
    }
