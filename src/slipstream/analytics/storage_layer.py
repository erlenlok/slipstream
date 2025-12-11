"""
Storage and persistence layer for Brawler performance analytics.

This module implements time-series storage using TimescaleDB for
persisting metrics and enabling historical analysis.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import asyncpg

from slipstream.analytics.data_structures import PerformanceMetrics, TradeEvent
from slipstream.analytics.core_metrics_calculator import CoreMetricsCalculator
from slipstream.analytics.historical_analyzer import HistoricalAnalyzer
from slipstream.analytics.per_asset_analyzer import PerAssetPerformanceAnalyzer


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""

    host: str = "localhost"
    port: int = 5432
    database: str = "slipstream_analytics"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"

    def connection_string(self) -> str:
        """Generate connection string from config."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class TableNames(Enum):
    """Enum for table names in the analytics database."""
    
    TRADE_EVENTS = "trade_events"
    PERFORMANCE_SNAPSHOTS = "performance_snapshots"
    PER_ASSET_METRICS = "per_asset_metrics"
    ROLLING_METRICS = "rolling_metrics"
    HISTORICAL_TRENDS = "historical_trends"


@dataclass
class AnalyticsStorage:
    """Main storage class for Brawler performance analytics."""
    
    config: DatabaseConfig
    connection: Optional[asyncpg.Connection] = None
    logger: logging.Logger = None
    
    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> None:
        """Establish connection to the database."""
        try:
            self.connection = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password
            )
            self.logger.info("Connected to analytics database")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close the database connection."""
        if self.connection:
            await self.connection.close()
            self.logger.info("Disconnected from analytics database")
    
    async def create_tables(self) -> None:
        """Create all necessary tables for analytics storage."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        # Create trade events table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {TableNames.TRADE_EVENTS.value} (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                side VARCHAR(4) NOT NULL,  -- 'buy' or 'sell'
                quantity NUMERIC NOT NULL,
                price NUMERIC NOT NULL,
                trade_type VARCHAR(10) NOT NULL,  -- 'maker' or 'taker'
                fees_paid NUMERIC DEFAULT 0,
                funding_paid NUMERIC DEFAULT 0,
                position_before NUMERIC DEFAULT 0,
                position_after NUMERIC DEFAULT 0,
                reference_price NUMERIC,
                order_id VARCHAR(50),
                quote_id VARCHAR(50),
                spread_at_quote NUMERIC,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_trade_events_symbol_time ON trade_events (symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_trade_events_timestamp ON trade_events (timestamp);

        """)
        
        # Create performance snapshots table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {TableNames.PERFORMANCE_SNAPSHOTS.value} (
                id BIGSERIAL PRIMARY KEY,
                snapshot_time TIMESTAMP WITH TIME ZONE NOT NULL,
                window_hours INTEGER DEFAULT 24,
                total_pnl NUMERIC NOT NULL,
                fees_paid NUMERIC DEFAULT 0,
                funding_paid NUMERIC DEFAULT 0,
                total_quotes INTEGER DEFAULT 0,
                total_fills INTEGER DEFAULT 0,
                hit_rate NUMERIC DEFAULT 0,
                fill_rate NUMERIC DEFAULT 0,
                total_volume NUMERIC DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                avg_inventory NUMERIC DEFAULT 0,
                max_inventory NUMERIC DEFAULT 0,
                inventory_turnover NUMERIC DEFAULT 0,
                max_drawdown NUMERIC DEFAULT 0,
                sharpe_ratio NUMERIC DEFAULT 0,
                volatility NUMERIC DEFAULT 0,
                cancellation_rate NUMERIC DEFAULT 0,
                total_cancellations INTEGER DEFAULT 0,
                markout_avg NUMERIC DEFAULT 0,
                markout_std NUMERIC DEFAULT 0,
                markout_min NUMERIC DEFAULT 0,
                markout_max NUMERIC DEFAULT 0,
                markout_count INTEGER DEFAULT 0,
                extra_data JSONB,  -- Additional metrics that don't have dedicated columns
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_time ON performance_snapshots (snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_snapshots_window ON performance_snapshots (window_hours);

        """)
        
        # Create per-asset metrics table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {TableNames.PER_ASSET_METRICS.value} (
                id BIGSERIAL PRIMARY KEY,
                asset_symbol VARCHAR(10) NOT NULL,
                snapshot_time TIMESTAMP WITH TIME ZONE NOT NULL,
                total_pnl NUMERIC NOT NULL,
                fees_paid NUMERIC DEFAULT 0,
                funding_paid NUMERIC DEFAULT 0,
                total_volume NUMERIC DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                hit_rate NUMERIC DEFAULT 0,
                avg_markout NUMERIC DEFAULT 0,
                max_inventory NUMERIC DEFAULT 0,
                avg_inventory NUMERIC DEFAULT 0,
                sharpe_ratio NUMERIC DEFAULT 0,
                volatility NUMERIC DEFAULT 0,
                win_rate NUMERIC DEFAULT 0,
                profit_factor NUMERIC DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_asset_metrics_asset_time ON per_asset_metrics (asset_symbol, snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_asset_metrics_time ON per_asset_metrics (snapshot_time);

        """)
        
        # Create rolling metrics table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {TableNames.ROLLING_METRICS.value} (
                id BIGSERIAL PRIMARY KEY,
                window_type VARCHAR(10) NOT NULL,  -- '1h', '6h', '24h', etc.
                snapshot_time TIMESTAMP WITH TIME ZONE NOT NULL,
                total_pnl NUMERIC NOT NULL,
                hit_rate NUMERIC DEFAULT 0,
                sharpe_ratio NUMERIC DEFAULT 0,
                volatility NUMERIC DEFAULT 0,
                max_drawdown NUMERIC DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                avg_markout NUMERIC DEFAULT 0,
                extra_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_rolling_window_time ON rolling_metrics (window_type, snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_rolling_time ON rolling_metrics (snapshot_time);

        """)
        
        # Create historical trends table
        await self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {TableNames.HISTORICAL_TRENDS.value} (
                id BIGSERIAL PRIMARY KEY,
                metric_name VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                value NUMERIC NOT NULL,
                window_type VARCHAR(10),  -- '1h', '6h', '24h', etc.
                asset_symbol VARCHAR(10),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_historical_metric_time ON historical_trends (metric_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_historical_asset_time ON historical_trends (asset_symbol, timestamp);

        """)
        
        # Create hypertables for TimescaleDB (if TimescaleDB extension is available)
        try:
            # Check if TimescaleDB is available
            result = await self.connection.fetchval("SELECT installed_version FROM pg_available_extensions WHERE name = 'timescaledb'")
            if result:
                # Convert regular tables to hypertables for better time-series performance
                await self.connection.execute(f"""
                    SELECT create_hypertable('{TableNames.TRADE_EVENTS.value}', 'timestamp', if_not_exists => TRUE);
                    SELECT create_hypertable('{TableNames.PERFORMANCE_SNAPSHOTS.value}', 'snapshot_time', if_not_exists => TRUE);
                    SELECT create_hypertable('{TableNames.PER_ASSET_METRICS.value}', 'snapshot_time', if_not_exists => TRUE);
                    SELECT create_hypertable('{TableNames.ROLLING_METRICS.value}', 'snapshot_time', if_not_exists => TRUE);
                    SELECT create_hypertable('{TableNames.HISTORICAL_TRENDS.value}', 'timestamp', if_not_exists => TRUE);
                """)
                self.logger.info("Created TimescaleDB hypertables")
        except Exception as e:
            self.logger.warning(f"Could not create TimescaleDB hypertables: {e}")
        
        self.logger.info("Created analytics database tables")
    
    async def store_trade_event(self, trade: TradeEvent) -> int:
        """Store a single trade event in the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        query = f"""
            INSERT INTO {TableNames.TRADE_EVENTS.value} (
                timestamp, symbol, side, quantity, price, trade_type, 
                fees_paid, funding_paid, position_before, position_after,
                reference_price, order_id, quote_id, spread_at_quote
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            ) RETURNING id;
        """
        
        # Convert TradeEvent data to the right format for DB
        row_id = await self.connection.fetchval(
            query,
            trade.timestamp,
            trade.symbol,
            trade.side,
            str(trade.quantity),
            str(trade.price),
            trade.trade_type.value if hasattr(trade.trade_type, 'value') else str(trade.trade_type),
            str(trade.fees_paid),
            str(trade.funding_paid),
            str(trade.position_before),
            str(trade.position_after),
            str(trade.reference_price) if trade.reference_price is not None else None,
            trade.order_id,
            trade.quote_id,
            str(trade.spread_at_quote) if trade.spread_at_quote is not None else None
        )
        
        return row_id
    
    async def store_performance_snapshot(self, metrics: PerformanceMetrics, 
                                       snapshot_time: datetime = None) -> int:
        """Store a performance snapshot in the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        if snapshot_time is None:
            snapshot_time = datetime.now()
        
        # Store main metrics in dedicated columns
        query = f"""
            INSERT INTO {TableNames.PERFORMANCE_SNAPSHOTS.value} (
                snapshot_time, window_hours, total_pnl, fees_paid, funding_paid,
                total_quotes, total_fills, hit_rate, fill_rate, total_volume,
                total_trades, avg_inventory, max_inventory, inventory_turnover,
                max_drawdown, sharpe_ratio, volatility, cancellation_rate,
                total_cancellations, markout_avg, markout_std, markout_min,
                markout_max, markout_count, extra_data
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
            ) RETURNING id;
        """
        
        # Convert metrics to database format
        row_id = await self.connection.fetchval(
            query,
            snapshot_time,
            metrics.window_hours,
            str(metrics.total_pnl),
            str(metrics.fees_paid),
            str(metrics.funding_paid),
            metrics.total_quotes,
            metrics.total_fills,
            str(metrics.hit_rate),
            str(metrics.fill_rate),
            str(metrics.total_volume),
            metrics.total_trades,
            str(metrics.avg_inventory),
            str(metrics.max_inventory),
            str(metrics.inventory_turnover),
            str(metrics.max_drawdown),
            str(metrics.sharpe_ratio),
            str(metrics.volatility),
            str(metrics.cancellation_rate),
            metrics.total_cancellations,
            str(metrics.markout_analysis.avg_markout_in),
            str(metrics.markout_analysis.markout_std),
            str(metrics.markout_analysis.markout_min),
            str(metrics.markout_analysis.markout_max),
            metrics.markout_analysis.markout_count,
            json.dumps({})  # extra_data - we'll add more detailed metrics here if needed
        )
        
        return row_id
    
    async def store_per_asset_metrics(self, asset_metrics: Dict[str, Any], 
                                    snapshot_time: datetime = None) -> List[int]:
        """Store per-asset metrics in the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        if snapshot_time is None:
            snapshot_time = datetime.now()
        
        query = f"""
            INSERT INTO {TableNames.PER_ASSET_METRICS.value} (
                asset_symbol, snapshot_time, total_pnl, fees_paid, funding_paid,
                total_volume, total_trades, hit_rate, avg_markout, max_inventory,
                avg_inventory, sharpe_ratio, volatility, win_rate, profit_factor
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            ) RETURNING id;
        """
        
        inserted_ids = []
        for symbol, metrics_dict in asset_metrics.items():
            row_id = await self.connection.fetchval(
                query,
                symbol,
                snapshot_time,
                str(metrics_dict.get('total_pnl', 0)),
                str(metrics_dict.get('fees_paid', 0)),
                str(metrics_dict.get('funding_paid', 0)),
                str(metrics_dict.get('total_volume', 0)),
                metrics_dict.get('total_trades', 0),
                str(metrics_dict.get('hit_rate', 0)),
                str(metrics_dict.get('avg_markout', 0)),
                str(metrics_dict.get('max_inventory', 0)),
                str(metrics_dict.get('avg_inventory', 0)),
                str(metrics_dict.get('sharpe_ratio', 0)),
                str(metrics_dict.get('volatility', 0)),
                str(metrics_dict.get('win_rate', 0)),
                str(metrics_dict.get('profit_factor', 0))
            )
            inserted_ids.append(row_id)
        
        return inserted_ids
    
    async def store_rolling_metrics(self, window_type: str, metrics: PerformanceMetrics,
                                  snapshot_time: datetime = None) -> int:
        """Store rolling window metrics in the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        if snapshot_time is None:
            snapshot_time = datetime.now()
        
        query = f"""
            INSERT INTO {TableNames.ROLLING_METRICS.value} (
                window_type, snapshot_time, total_pnl, hit_rate, sharpe_ratio,
                volatility, max_drawdown, total_trades, avg_markout, extra_data
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            ) RETURNING id;
        """
        
        row_id = await self.connection.fetchval(
            query,
            window_type,
            snapshot_time,
            str(metrics.total_pnl),
            str(metrics.hit_rate),
            str(metrics.sharpe_ratio),
            str(metrics.volatility),
            str(metrics.max_drawdown),
            metrics.total_trades,
            str(metrics.markout_analysis.avg_markout_in),
            json.dumps({})
        )
        
        return row_id
    
    async def store_historical_trend(self, metric_name: str, value: float,
                                   timestamp: datetime = None,
                                   window_type: Optional[str] = None,
                                   asset_symbol: Optional[str] = None) -> int:
        """Store a historical trend data point in the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        query = f"""
            INSERT INTO {TableNames.HISTORICAL_TRENDS.value} (
                metric_name, timestamp, value, window_type, asset_symbol
            ) VALUES (
                $1, $2, $3, $4, $5
            ) RETURNING id;
        """
        
        row_id = await self.connection.fetchval(
            query,
            metric_name,
            timestamp,
            str(value),
            window_type,
            asset_symbol
        )
        
        return row_id
    
    async def batch_store_trades(self, trades: List[TradeEvent]) -> List[int]:
        """Store multiple trade events in a batch operation."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        inserted_ids = []
        
        # Use a transaction for the batch operation
        async with self.connection.transaction():
            for trade in trades:
                row_id = await self.store_trade_event(trade)
                inserted_ids.append(row_id)
        
        return inserted_ids
    
    async def get_performance_snapshots(self, 
                                      start_time: datetime = None,
                                      end_time: datetime = None,
                                      window_hours: Optional[int] = None,
                                      limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve performance snapshots from the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        # Build query with optional filters
        query = f"SELECT * FROM {TableNames.PERFORMANCE_SNAPSHOTS.value}"
        conditions = []
        params = []
        param_index = 1
        
        if start_time:
            conditions.append(f"snapshot_time >= ${param_index}")
            params.append(start_time)
            param_index += 1
        
        if end_time:
            conditions.append(f"snapshot_time <= ${param_index}")
            params.append(end_time)
            param_index += 1
        
        if window_hours is not None:
            conditions.append(f"window_hours = ${param_index}")
            params.append(window_hours)
            param_index += 1
        
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        
        query += f" ORDER BY snapshot_time DESC LIMIT ${param_index}"
        params.append(limit)
        
        rows = await self.connection.fetch(query, *params)
        
        # Convert rows to dictionaries
        snapshots = []
        for row in rows:
            snapshot = dict(row)
            # Convert numeric strings back to appropriate types
            for key, value in snapshot.items():
                if isinstance(value, str) and key not in ['extra_data']:  # Don't convert JSON fields
                    try:
                        snapshot[key] = float(value)
                    except ValueError:
                        pass  # Keep original value if not convertible
            snapshots.append(snapshot)
        
        return snapshots
    
    async def get_asset_metrics(self, 
                              symbol: str,
                              start_time: datetime = None,
                              end_time: datetime = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve per-asset metrics from the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        query = f"SELECT * FROM {TableNames.PER_ASSET_METRICS.value} WHERE asset_symbol = $1"
        params = [symbol]
        param_index = 2
        
        if start_time:
            query += f" AND snapshot_time >= ${param_index}"
            params.append(start_time)
            param_index += 1
        
        if end_time:
            query += f" AND snapshot_time <= ${param_index}"
            params.append(end_time)
            param_index += 1
        
        query += f" ORDER BY snapshot_time DESC LIMIT ${param_index}"
        params.append(limit)
        
        rows = await self.connection.fetch(query, *params)
        
        # Convert rows to dictionaries
        metrics = []
        for row in rows:
            metric = dict(row)
            # Convert numeric strings back to appropriate types
            for key, value in metric.items():
                if isinstance(value, str) and key not in ['extra_data']:  # Don't convert JSON fields
                    try:
                        metric[key] = float(value)
                    except ValueError:
                        pass  # Keep original value if not convertible
            metrics.append(metric)
        
        return metrics
    
    async def get_recent_trades(self, 
                               symbol: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent trade events from the database."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        query = f"SELECT * FROM {TableNames.TRADE_EVENTS.value}"
        params = []
        param_index = 1
        
        if symbol:
            query += f" WHERE symbol = ${param_index}"
            params.append(symbol)
            param_index += 1
        
        query += f" ORDER BY timestamp DESC LIMIT ${param_index}"
        params.append(limit)
        
        rows = await self.connection.fetch(query, *params)
        
        # Convert rows to dictionaries
        trades = []
        for row in rows:
            trade = dict(row)
            # Convert numeric strings back to appropriate types
            for key, value in trade.items():
                if isinstance(value, str) and key not in ['order_id', 'quote_id']:  # Don't convert IDs
                    try:
                        trade[key] = float(value)
                    except ValueError:
                        pass  # Keep original value if not convertible
            trades.append(trade)
        
        return trades
    
    async def cleanup_old_data(self, table_name: TableNames, 
                             older_than_days: int = 30) -> int:
        """Clean up old data from a table."""
        if not self.connection:
            raise RuntimeError("Database not connected")
        
        delete_threshold = datetime.now() - timedelta(days=older_than_days)
        
        query = f"DELETE FROM {table_name.value} WHERE created_at < $1"
        result = await self.connection.execute(query, delete_threshold)
        
        # Extract the number of deleted rows from the result
        deleted_count = int(result.split()[-1]) if result.split() else 0
        
        self.logger.info(f"Cleaned up {deleted_count} rows from {table_name.value}")
        return deleted_count


if __name__ == "__main__":
    # This would test the storage functionality, but since we're mocking,
    # we won't actually connect to a database in this example
    print("Storage layer implementation completed with asyncpg interface.")
    print("Database connection and table creation methods implemented.")
    print("All storage methods are ready for integration with actual database.")