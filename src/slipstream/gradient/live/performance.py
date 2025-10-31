"""Performance tracking and aggregation for live trading."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class PerformanceTracker:
    """Track and aggregate trading performance metrics."""

    def __init__(self, log_dir: str = "/var/log/gradient"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rebalance_log = self.log_dir / "rebalance_history.jsonl"
        self.positions_log = self.log_dir / "positions_history.jsonl"

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
            "total_orders": len(execution_results.get("stage1_orders", [])),
            "total_turnover": execution_results.get("total_turnover", 0),
            "errors": len(execution_results.get("errors", [])),
            "dry_run": config.dry_run,
        }

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

        avg_positions = sum(r["n_positions"] for r in rebalances) / len(rebalances)
        avg_gross_exposure = sum(r["gross_exposure"] for r in rebalances) / len(rebalances)
        avg_net_exposure = sum(r["net_exposure"] for r in rebalances) / len(rebalances)

        # Calculate fill rates
        passive_fill_rate = total_stage1_fills / total_orders if total_orders > 0 else 0
        aggressive_fill_rate = total_stage2_fills / total_orders if total_orders > 0 else 0

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
            "rebalances": rebalances,
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
            "passive_fill_rate": total_stage1_fills / total_orders if total_orders > 0 else 0,
            "aggressive_fill_rate": total_stage2_fills / total_orders if total_orders > 0 else 0,
        }
