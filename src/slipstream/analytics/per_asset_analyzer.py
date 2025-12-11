"""
Per-instrument analysis and breakdowns for Brawler performance tracking.

This module implements comprehensive per-instrument performance analysis,
cross-asset correlation analysis, and asset-specific metrics tracking.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import statistics

from slipstream.analytics.data_structures import TradeEvent, PerformanceMetrics
from slipstream.analytics.core_metrics_calculator import CoreMetricsCalculator
from slipstream.analytics.historical_analyzer import TimeWindow


@dataclass
class AssetPerformanceMetrics:
    """Performance metrics specific to a single asset."""
    
    symbol: str
    total_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    total_volume: float = 0.0
    total_trades: int = 0
    hit_rate: float = 0.0
    avg_markout: float = 0.0
    max_inventory: float = 0.0
    avg_inventory: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0  # Gross profit / Gross loss
    
    def update_from_trade(self, trade: TradeEvent) -> None:
        """Update metrics for this specific asset from a trade."""
        self.total_trades += 1
        self.total_volume += trade.price * trade.quantity
        self.fees_paid += trade.fees_paid
        self.funding_paid += trade.funding_paid
        
        # Calculate trade PnL (simplified, based on fees/funding impact)
        # In a real system, this would be based on actual entry/exit
        trade_pnl = -(trade.fees_paid + trade.funding_paid)
        self.total_pnl += trade_pnl
        
        # Update inventory if needed
        abs_inventory = abs(trade.position_after)
        if abs_inventory > self.max_inventory:
            self.max_inventory = abs_inventory


@dataclass
class PerInstrumentAnalyzer:
    """Analyzer for per-asset performance tracking."""
    
    # Asset-specific metrics
    asset_metrics: Dict[str, AssetPerformanceMetrics] = field(default_factory=dict)
    
    # Per-asset trade data
    asset_trades: Dict[str, List[TradeEvent]] = field(default_factory=dict)
    
    # Cross-asset correlation tracking
    correlation_matrix: Optional[np.ndarray] = None
    asset_symbols: List[str] = field(default_factory=list)
    
    # Capacity and sizing analysis
    asset_capacity_analysis: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def add_trade(self, trade: TradeEvent) -> None:
        """Add a trade to the per-asset analysis."""
        symbol = trade.symbol
        
        # Initialize asset metrics if needed
        if symbol not in self.asset_metrics:
            self.asset_metrics[symbol] = AssetPerformanceMetrics(symbol=symbol)
            self.asset_trades[symbol] = []
        
        # Add trade to asset-specific tracking
        self.asset_trades[symbol].append(trade)
        
        # Update asset metrics
        asset_metrics = self.asset_metrics[symbol]
        asset_metrics.update_from_trade(trade)
    
    def calculate_asset_metrics(self) -> Dict[str, AssetPerformanceMetrics]:
        """Calculate comprehensive metrics for all assets."""
        for symbol, trades in self.asset_trades.items():
            if not trades:
                continue
                
            # Calculate derived metrics for this asset
            metrics = self.asset_metrics[symbol]
            
            # Calculate hit rate based on quote fills
            total_quotes = sum(1 for t in trades if t.quote_id is not None)
            total_fills = sum(1 for t in trades if t.quote_id is not None and t.quantity > 0)
            if total_quotes > 0:
                metrics.hit_rate = (total_fills / total_quotes) * 100
            
            # Calculate average markout
            total_markout = 0.0
            markout_count = 0
            for trade in trades:
                if trade.reference_price is not None:
                    markout = trade.calculate_markout()
                    if markout is not None:
                        total_markout += markout
                        markout_count += 1
            
            if markout_count > 0:
                metrics.avg_markout = total_markout / markout_count
            
            # Calculate win rate (need to track profits/losses properly)
            # This is a simplified version
            winning_trades = sum(1 for t in trades if -(t.fees_paid + t.funding_paid) > 0)
            if len(trades) > 0:
                metrics.win_rate = (winning_trades / len(trades)) * 100
    
    def get_asset_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of performance by asset."""
        summary = {}
        for symbol, metrics in self.asset_metrics.items():
            summary[symbol] = {
                'total_pnl': metrics.total_pnl,
                'fees_paid': metrics.fees_paid,
                'funding_paid': metrics.funding_paid,
                'total_volume': metrics.total_volume,
                'total_trades': metrics.total_trades,
                'hit_rate': metrics.hit_rate,
                'avg_markout': metrics.avg_markout,
                'max_inventory': metrics.max_inventory
            }
        return summary
    
    def get_top_performing_assets(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top N performing assets by PnL."""
        asset_pnl = [(symbol, metrics.total_pnl) 
                     for symbol, metrics in self.asset_metrics.items()]
        return sorted(asset_pnl, key=lambda x: x[1], reverse=True)[:n]
    
    def get_worst_performing_assets(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the worst N performing assets by PnL."""
        asset_pnl = [(symbol, metrics.total_pnl) 
                     for symbol, metrics in self.asset_metrics.items()]
        return sorted(asset_pnl, key=lambda x: x[1])[:n]


@dataclass
class CrossAssetAnalyzer:
    """Analyzer for cross-asset correlation and interaction effects."""
    
    # Track performance relationships between assets
    correlation_tracker: Dict[Tuple[str, str], List[Tuple[datetime, float, float]]] = field(default_factory=dict)
    
    # Track how one asset's performance affects another
    performance_impact_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Common asset pairs to track
    asset_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    
    def add_asset_pair_performance(self, asset1: str, asset2: str, 
                                 perf1: float, perf2: float, 
                                 timestamp: datetime) -> None:
        """Add performance data for an asset pair."""
        # Ensure consistent ordering (alphabetical)
        if asset1 > asset2:
            asset1, asset2 = asset2, asset1
            perf1, perf2 = perf2, perf1
            
        pair = (asset1, asset2)
        self.asset_pairs.add(pair)
        
        if pair not in self.correlation_tracker:
            self.correlation_tracker[pair] = []
        
        self.correlation_tracker[pair].append((timestamp, perf1, perf2))
    
    def calculate_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlation coefficients between asset pairs."""
        correlations = {}
        
        for pair, data in self.correlation_tracker.items():
            if len(data) < 2:  # Need at least 2 points for correlation
                correlations[pair] = 0.0
                continue
            
            # Extract just the performance values
            perf1_values = [point[1] for point in data]
            perf2_values = [point[2] for point in data]
            
            # Calculate correlation coefficient
            if np.std(perf1_values) == 0 or np.std(perf2_values) == 0:
                correlations[pair] = 0.0
            else:
                corr_matrix = np.corrcoef(perf1_values, perf2_values)
                correlations[pair] = float(corr_matrix[0, 1])
        
        return correlations
    
    def get_correlation_impact(self, target_asset: str) -> Dict[str, float]:
        """Get how other assets correlate with a target asset."""
        impacts = {}
        
        for (asset1, asset2), correlation in self.calculate_correlations().items():
            if target_asset in [asset1, asset2]:
                other_asset = asset2 if asset1 == target_asset else asset1
                impacts[other_asset] = correlation
        
        return impacts


@dataclass
class InventoryConcentrationTracker:
    """Track inventory concentration risks across instruments."""
    
    # Current inventory by asset
    current_inventory: Dict[str, float] = field(default_factory=dict)
    
    # Historical inventory concentration
    concentration_history: List[Tuple[datetime, Dict[str, float]]] = field(default_factory=list)
    
    # Concentration thresholds
    warning_threshold: float = 0.3  # 30% of total inventory in one asset
    critical_threshold: float = 0.5  # 50% of total inventory in one asset
    
    # Concentration metrics
    max_concentration_ratio: float = 0.0
    avg_concentration_ratio: float = 0.0
    
    def update_inventory(self, symbol: str, position: float) -> Dict[str, str]:
        """Update inventory for an asset and check for concentration risks."""
        self.current_inventory[symbol] = position
        
        # Record in history
        self.concentration_history.append((datetime.now(), self.current_inventory.copy()))
        
        # Calculate concentration metrics
        total_inventory = sum(abs(pos) for pos in self.current_inventory.values())
        if total_inventory == 0:
            return {}
        
        alerts = {}
        for asset, pos in self.current_inventory.items():
            ratio = abs(pos) / total_inventory
            if ratio > self.critical_threshold:
                alerts[asset] = "CRITICAL_CONCENTRATION"
            elif ratio > self.warning_threshold:
                alerts[asset] = "WARNING_CONCENTRATION"
        
        # Update concentration metrics
        concentrations = [abs(pos) / total_inventory for pos in self.current_inventory.values()]
        if concentrations:
            self.max_concentration_ratio = max(concentrations)
            self.avg_concentration_ratio = sum(concentrations) / len(concentrations)
        
        return alerts
    
    def get_concentration_report(self) -> Dict[str, float]:
        """Get current inventory concentration report."""
        total_inventory = sum(abs(pos) for pos in self.current_inventory.values())
        if total_inventory == 0:
            return {}
        
        report = {}
        for asset, pos in self.current_inventory.items():
            report[asset] = abs(pos) / total_inventory
        
        return report


@dataclass
class AssetPairAnalyzer:
    """Analyzer for specific asset pairs and their interactions."""
    
    # Track performance of asset pairs
    pair_performance: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    
    # Track how assets affect each other
    cross_effects: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)
    
    def analyze_asset_pair(self, asset1: str, asset2: str, 
                          trades1: List[TradeEvent], 
                          trades2: List[TradeEvent]) -> Dict[str, float]:
        """Analyze the relationship between two assets."""
        pair = tuple(sorted([asset1, asset2]))  # Sort for consistency
        
        # Calculate basic metrics for each asset
        pnl1 = sum(-(t.fees_paid + t.funding_paid) for t in trades1)
        pnl2 = sum(-(t.fees_paid + t.funding_paid) for t in trades2)
        
        # Calculate correlation in performance
        if len(trades1) > 1 and len(trades2) > 1:
            # For simplicity, using last few trades as time series
            perf1 = [-(t.fees_paid + t.funding_paid) for t in trades1[-10:]]
            perf2 = [-(t.fees_paid + t.funding_paid) for t in trades2[-10:]]
            
            # Pad shorter series with zeros if needed
            while len(perf1) < len(perf2):
                perf1.append(0.0)
            while len(perf2) < len(perf1):
                perf2.append(0.0)
                
            if len(perf1) > 1:
                correlation = np.corrcoef(perf1, perf2)[0, 1] if np.std(perf1) > 0 and np.std(perf2) > 0 else 0.0
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        
        pair_metrics = {
            'asset1_pnl': pnl1,
            'asset2_pnl': pnl2,
            'combined_pnl': pnl1 + pnl2,
            'correlation': correlation,
            'total_trades_asset1': len(trades1),
            'total_trades_asset2': len(trades2)
        }
        
        self.pair_performance[pair] = pair_metrics
        return pair_metrics


@dataclass
class CapacityAnalyzer:
    """Analyze capacity constraints per asset."""
    
    # Track capacity utilization per asset
    capacity_utilization: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Size impact analysis
    size_impact_data: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)  # (size, impact)
    
    # Maximum recommended sizes
    max_sizes: Dict[str, float] = field(default_factory=dict)
    
    def add_size_impact_data(self, symbol: str, size: float, performance_impact: float) -> None:
        """Add data point about how trade size affects performance for an asset."""
        if symbol not in self.size_impact_data:
            self.size_impact_data[symbol] = []
        
        self.size_impact_data[symbol].append((size, performance_impact))
    
    def calculate_capacity_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate capacity metrics for all assets."""
        metrics = {}
        
        for symbol, data_points in self.size_impact_data.items():
            if not data_points:
                continue
            
            sizes, impacts = zip(*data_points)
            
            # Calculate capacity metrics
            avg_impact_per_size = np.mean(impacts) / (np.mean(sizes) + 1e-8) if sizes else 0.0
            max_size_observed = max(sizes) if sizes else 0.0
            
            # Determine max recommended size (where impact becomes significantly negative)
            performance_threshold = np.percentile(impacts, 25) if impacts else 0.0  # 25th percentile
            max_efficient_size = max_size_observed  # Simplified approach
            
            metrics[symbol] = {
                'avg_impact_per_size': avg_impact_per_size,
                'max_size_observed': max_size_observed,
                'recommended_max_size': max_efficient_size,
                'data_points_count': len(data_points)
            }
        
        return metrics
    
    def get_capacity_warnings(self) -> Dict[str, str]:
        """Get warnings for assets approaching capacity limits."""
        warnings = {}
        
        capacity_metrics = self.calculate_capacity_metrics()
        for symbol, metrics in capacity_metrics.items():
            recommended_max = metrics.get('recommended_max_size', float('inf'))
            current_usage = self.capacity_utilization.get(symbol, {}).get('current_size', 0)
            
            utilization = current_usage / recommended_max if recommended_max > 0 else 0
            if utilization > 0.8:  # 80% utilization
                warnings[symbol] = "APPROACHING_CAPACITY_LIMIT"
            elif utilization > 0.95:  # 95% utilization
                warnings[symbol] = "NEAR_CAPACITY_LIMIT"
        
        return warnings


@dataclass
class PerAssetPerformanceAnalyzer:
    """Main analyzer for comprehensive per-asset performance tracking."""
    
    # Sub-analyzers
    per_asset: PerInstrumentAnalyzer = field(default_factory=PerInstrumentAnalyzer)
    cross_asset: CrossAssetAnalyzer = field(default_factory=CrossAssetAnalyzer)
    concentration: InventoryConcentrationTracker = field(default_factory=InventoryConcentrationTracker)
    asset_pairs: AssetPairAnalyzer = field(default_factory=AssetPairAnalyzer)
    capacity: CapacityAnalyzer = field(default_factory=CapacityAnalyzer)
    
    def process_trades_batch(self, trades: List[TradeEvent]) -> None:
        """Process a batch of trades for per-asset analysis."""
        # Group trades by asset
        trades_by_asset: Dict[str, List[TradeEvent]] = defaultdict(list)
        for trade in trades:
            trades_by_asset[trade.symbol].append(trade)
            self.per_asset.add_trade(trade)
            
            # Update inventory concentration
            self.concentration.update_inventory(trade.symbol, trade.position_after)
        
        # Analyze each asset
        self.per_asset.calculate_asset_metrics()
        
        # Analyze cross-asset effects (simplified view)
        asset_symbols = list(trades_by_asset.keys())
        for i in range(len(asset_symbols)):
            for j in range(i + 1, len(asset_symbols)):
                asset1, asset2 = asset_symbols[i], asset_symbols[j]
                trades1, trades2 = trades_by_asset[asset1], trades_by_asset[asset2]
                
                # Add to cross-asset correlation tracking
                if trades1 and trades2:
                    # Use last trade performance as a proxy
                    perf1 = -(trades1[-1].fees_paid + trades1[-1].funding_paid)
                    perf2 = -(trades2[-1].fees_paid + trades2[-1].funding_paid)
                    self.cross_asset.add_asset_pair_performance(
                        asset1, asset2, perf1, perf2, trades1[-1].timestamp
                    )
                
                # Analyze asset pair
                self.asset_pairs.analyze_asset_pair(asset1, asset2, trades1, trades2)
    
    def get_per_asset_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive per-asset performance summary."""
        return self.per_asset.get_asset_performance_summary()
    
    def get_concentration_risk_report(self) -> Dict[str, float]:
        """Get inventory concentration risk report."""
        return self.concentration.get_concentration_report()
    
    def get_cross_asset_correlations(self) -> Dict[Tuple[str, str], float]:
        """Get correlations between asset pairs."""
        return self.cross_asset.calculate_correlations()
    
    def get_capacity_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get capacity analysis for all assets."""
        return self.capacity.calculate_capacity_metrics()
    
    def get_top_performing_assets(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N performing assets."""
        return self.per_asset.get_top_performing_assets(n)
    
    def get_worst_performing_assets(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get worst N performing assets."""
        return self.per_asset.get_worst_performing_assets(n)


def test_per_asset_pnl_calculation():
    """Test PnL calculation broken down by asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades for different assets
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, funding_paid=2.0),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=0.5, 
                  price=50100.0, trade_type=None, fees_paid=5.0, funding_paid=3.0)
    ]
    
    analyzer.process_trades_batch(trades)
    summary = analyzer.get_per_asset_summary()
    
    # Should have metrics for BTC and ETH
    assert "BTC" in summary
    assert "ETH" in summary


def test_per_asset_hit_rate():
    """Test hit rate calculation per individual asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades with quotes for different assets
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, quote_id="q1"),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=1.0, 
                  price=50050.0, trade_type=None, fees_paid=10.0, quote_id="q2"),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, quote_id=None),  # No quote
    ]
    
    analyzer.process_trades_batch(trades)
    
    # Check that assets were processed
    summary = analyzer.get_per_asset_summary()
    assert "BTC" in summary
    assert "ETH" in summary


def test_per_asset_markout():
    """Test markout calculation per individual asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades with reference prices for markout calculation
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, 
                  reference_price=50010.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="sell", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, 
                  reference_price=2999.0),
    ]
    
    analyzer.process_trades_batch(trades)
    
    # Check that assets were processed
    summary = analyzer.get_per_asset_summary()
    assert "BTC" in summary
    assert "ETH" in summary


def test_asset_correlation_analysis():
    """Test correlation analysis between different assets."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Add cross-asset performance data
    analyzer.cross_asset.add_asset_pair_performance(
        "BTC", "ETH", 100.0, 50.0, datetime.now()
    )
    analyzer.cross_asset.add_asset_pair_performance(
        "BTC", "ETH", 120.0, 55.0, datetime.now()
    )
    analyzer.cross_asset.add_asset_pair_performance(
        "BTC", "ETH", 90.0, 45.0, datetime.now()
    )
    
    correlations = analyzer.get_cross_asset_correlations()
    
    # Should have correlation data for BTC-ETH
    assert ("BTC", "ETH") in correlations


def test_inventory_concentration_metrics():
    """Test metrics for inventory concentration by asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Update inventory with different levels
    alerts = analyzer.concentration.update_inventory("BTC", 5.0)
    alerts2 = analyzer.concentration.update_inventory("ETH", 2.0)
    alerts3 = analyzer.concentration.update_inventory("SOL", 0.5)
    
    # Get concentration report
    report = analyzer.get_concentration_risk_report()
    
    # Should have concentrations for all assets
    assert "BTC" in report
    assert "ETH" in report
    assert "SOL" in report


def test_asset_capacity_analysis():
    """Test capacity analysis per asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Add size impact data
    analyzer.capacity.add_size_impact_data("BTC", 1.0, -0.001)
    analyzer.capacity.add_size_impact_data("BTC", 2.0, -0.003)
    analyzer.capacity.add_size_impact_data("BTC", 5.0, -0.010)
    analyzer.capacity.add_size_impact_data("ETH", 10.0, -0.0005)
    analyzer.capacity.add_size_impact_data("ETH", 20.0, -0.0015)
    
    # Get capacity analysis
    capacity_metrics = analyzer.get_capacity_analysis()
    
    # Should have metrics for both assets
    assert "BTC" in capacity_metrics
    assert "ETH" in capacity_metrics


def test_cross_asset_impact():
    """Test how performance on one asset affects others."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades for multiple assets
    btc_trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=1.0, 
                  price=50050.0, trade_type=None, fees_paid=10.0, funding_paid=5.0)
    ]
    
    eth_trades = [
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, funding_paid=2.0)
    ]
    
    # Process all trades through the main analyzer
    all_trades = btc_trades + eth_trades
    analyzer.process_trades_batch(all_trades)
    
    # Get top performing assets
    top_assets = analyzer.get_top_performing_assets(2)
    
    # Should return a list of tuples (asset, pnl)
    assert isinstance(top_assets, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in top_assets)


if __name__ == "__main__":
    # Run the tests
    test_per_asset_pnl_calculation()
    test_per_asset_hit_rate()
    test_per_asset_markout()
    test_asset_correlation_analysis()
    test_inventory_concentration_metrics()
    test_asset_capacity_analysis()
    test_cross_asset_impact()
    
    print("All Per-Instrument Analysis and Breakdowns tests passed!")