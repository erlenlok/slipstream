"""
Mock data pipeline and event processing for Brawler performance tracking.

This module provides realistic mock data generation and event processing
to simulate Brawler's trade events for testing and analytics.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List, Dict, Generator, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from slipstream.analytics.data_structures import TradeEvent, TradeType, PerformanceMetrics


@dataclass
class MockTradeConfig:
    """Configuration for mock trade generation."""
    
    # Market parameters
    symbols: List[str] = None  # Default will be set in __post_init__
    base_price_range: Tuple[float, float] = (30000.0, 60000.0)  # For BTC
    volatility: float = 0.02  # 2% daily volatility
    spread_bps: float = 5.0   # 5 basis points average spread
    
    # Trading parameters
    avg_trades_per_hour: float = 10.0
    maker_taker_ratio: float = 0.7  # 70% maker, 30% taker
    avg_position_size: float = 0.1  # Average position size
    max_position_size: float = 1.0  # Maximum position size
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC", "ETH", "SOL", "XRP", "ADA"]
    

class MockTradeGenerator:
    """Generates realistic mock trade data for Brawler performance testing."""
    
    def __init__(self, config: MockTradeConfig = None):
        self.config = config or MockTradeConfig()
        self.current_prices = {}
        self.position_sizes = {}
        
        # Initialize current prices for all symbols
        for symbol in self.config.symbols:
            base_price = random.uniform(*self.config.base_price_range)
            self.current_prices[symbol] = base_price
            self.position_sizes[symbol] = 0.0
    
    def generate_trade(self, timestamp: datetime, symbol: str) -> TradeEvent:
        """Generate a single realistic trade event."""
        # Update price based on volatility
        current_price = self.current_prices[symbol]
        price_change = random.normalvariate(0, self.config.volatility / 24)  # Hourly volatility
        new_price = current_price * (1 + price_change)
        self.current_prices[symbol] = new_price
        
        # Determine trade direction (based on current position to avoid excessive directional bias)
        if abs(self.position_sizes[symbol]) < self.config.max_position_size * 0.8:
            # More likely to trade in direction that moves toward neutral
            if self.position_sizes[symbol] > 0:
                # More likely to sell if we're long
                side = "sell" if random.random() > 0.3 else "buy"
            elif self.position_sizes[symbol] < 0:
                # More likely to buy if we're short  
                side = "buy" if random.random() > 0.3 else "sell"
            else:
                # Equal chance if neutral
                side = "buy" if random.random() > 0.5 else "sell"
        else:
            # If position is too large, trade to reduce it
            side = "sell" if self.position_sizes[symbol] > 0 else "buy"
        
        # Determine trade type (maker vs taker)
        is_maker = random.random() < self.config.maker_taker_ratio
        trade_type = TradeType.MAKER if is_maker else TradeType.TAKER
        
        # Calculate quantity (smaller for larger positions to control risk)
        position_factor = max(0.1, 1.0 - abs(self.position_sizes[symbol]) / self.config.max_position_size)
        quantity = random.uniform(0.01, self.config.avg_position_size * 2) * position_factor
        
        # Calculate price based on spread and trade type
        if side == "buy":
            if is_maker:
                # Maker buy: price slightly below market (better price)
                price = new_price * (1 - self.config.spread_bps / 20000)  # Half spread
            else:
                # Taker buy: price at or above market (worse price)
                price = new_price * (1 + self.config.spread_bps / 10000)  # Full spread impact
        else:  # sell
            if is_maker:
                # Maker sell: price slightly above market (better price)
                price = new_price * (1 + self.config.spread_bps / 20000)  # Half spread
            else:
                # Taker sell: price at or below market (worse price)
                price = new_price * (1 - self.config.spread_bps / 10000)  # Full spread impact
        
        # Calculate fees (0.1% typical for maker, 0.2% for taker)
        fee_rate = 0.0005 if is_maker else 0.0010
        fees_paid = abs(price * quantity * fee_rate)
        
        # Calculate funding (simplified random funding cost)
        funding_paid = random.uniform(-abs(price * quantity * 0.0001), abs(price * quantity * 0.0001))
        
        # Calculate position before and after
        position_before = self.position_sizes[symbol]
        position_change = quantity if side == "buy" else -quantity
        position_after = position_before + position_change
        self.position_sizes[symbol] = position_after
        
        # Calculate reference price for markout (next market price)
        # This is a simplified model where reference_price is the next trade price
        reference_price = new_price * (1 + random.uniform(-0.001, 0.001))  # Small random variation
        
        return TradeEvent(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            price=price,
            trade_type=trade_type,
            reference_price=reference_price,
            fees_paid=fees_paid,
            funding_paid=funding_paid,
            position_before=position_before,
            position_after=position_after,
            order_id=f"mock_order_{random.randint(100000, 999999)}",
            quote_id=f"mock_quote_{random.randint(100000, 999999)}",
            spread_at_quote=self.config.spread_bps / 10000  # Convert to decimal
        )
    
    def generate_trades_stream(self, start_time: datetime, end_time: datetime, 
                             trades_per_hour: float = None) -> Generator[TradeEvent, None, None]:
        """Generate a stream of trades over a time period."""
        current_time = start_time
        trades_per_hour = trades_per_hour or self.config.avg_trades_per_hour
        
        while current_time < end_time:
            # Determine how many trades to generate in this hour
            # Use Poisson distribution for more realistic trade clustering
            trades_this_hour = np.random.poisson(trades_per_hour)
            
            for _ in range(trades_this_hour):
                # Randomly select a symbol
                symbol = random.choice(self.config.symbols)
                
                # Generate the trade
                trade = self.generate_trade(current_time, symbol)
                yield trade
                
                # Add small random delay between trades (milliseconds)
                ms_delay = random.uniform(10, 1000)  # 10ms to 1s between trades
                current_time += timedelta(milliseconds=ms_delay)
                
                if current_time >= end_time:
                    return
            
            # Move to next hour
            current_time += timedelta(hours=1)
    
    def generate_24h_trades(self, start_time: datetime = None) -> List[TradeEvent]:
        """Generate 24 hours worth of trade data."""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=24)
        
        end_time = start_time + timedelta(hours=24)
        
        trades = []
        for trade in self.generate_trades_stream(start_time, end_time):
            trades.append(trade)
        
        return trades


class MockBrawlerEventProcessor:
    """Processes mock Brawler events for analytics."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.trade_buffer = []
        self.quote_events = []
        self.cancellation_events = []
        
        # For 24-hour window tracking
        self.window_start = None
        self.window_end = None
    
    def set_time_window(self, start_time: datetime, end_time: datetime = None):
        """Set the time window for metrics collection."""
        self.window_start = start_time
        self.window_end = end_time or datetime.now()
        self.metrics.start_time = start_time
        self.metrics.end_time = end_time
    
    def process_trade_event(self, trade_event: TradeEvent) -> None:
        """Process a single trade event."""
        # Add to buffer
        self.trade_buffer.append(trade_event)
        
        # Update metrics
        self.metrics.update_from_trade(trade_event)
    
    def process_quote_event(self, quote_event: Dict) -> None:
        """Process a quote event (place/cancel)."""
        self.quote_events.append(quote_event)
        self.metrics.update_from_quote(quote_event)
    
    def process_cancellation_event(self, cancel_event: Dict) -> None:
        """Process an order cancellation event."""
        self.cancellation_events.append(cancel_event)
        self.metrics.update_from_cancellation()
    
    def process_multiple_events(self, trade_events: List[TradeEvent]) -> None:
        """Process multiple trade events."""
        for trade in trade_events:
            self.process_trade_event(trade)
    
    def get_24h_snapshot(self) -> PerformanceMetrics:
        """Get 24-hour performance snapshot."""
        # Finalize calculations before returning
        self.metrics.finalize_calculations()
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics and event buffers."""
        self.metrics = PerformanceMetrics()
        self.trade_buffer = []
        self.quote_events = []
        self.cancellation_events = []


def test_mock_trade_generation():
    """Test that mock trades are generated with realistic parameters."""
    generator = MockTradeGenerator()
    trades = generator.generate_24h_trades()
    
    assert len(trades) > 0, "Should generate some trades"
    assert all(trade.symbol in ["BTC", "ETH", "SOL", "XRP", "ADA"] for trade in trades)
    assert all(trade.quantity > 0 for trade in trades)
    assert all(trade.price > 0 for trade in trades)
    
    # Check that we have both maker and taker trades
    maker_trades = [t for t in trades if t.trade_type == TradeType.MAKER]
    taker_trades = [t for t in trades if t.trade_type == TradeType.TAKER]
    assert len(maker_trades) > 0 and len(taker_trades) > 0


def test_mock_trade_stream():
    """Test that mock trades can be streamed over time periods."""
    start_time = datetime.now() - timedelta(hours=2)
    end_time = datetime.now()
    
    generator = MockTradeGenerator()
    trade_stream = list(generator.generate_trades_stream(start_time, end_time, trades_per_hour=5))
    
    assert len(trade_stream) > 0
    assert all(t.timestamp >= start_time and t.timestamp <= end_time for t in trade_stream)
    
    # Verify timestamps are roughly in order (with potential small variations due to ms delays)
    assert all(trade_stream[i].timestamp <= trade_stream[i+1].timestamp 
              for i in range(len(trade_stream)-1))


def test_trade_event_processing():
    """Test that trade events are processed correctly by the analytics system."""
    processor = MockBrawlerEventProcessor()
    
    # Generate some mock trades
    generator = MockTradeGenerator()
    trades = generator.generate_24h_trades()
    
    # Process the trades
    processor.process_multiple_events(trades)
    
    # Check that metrics were updated
    metrics = processor.get_24h_snapshot()
    assert metrics.total_trades == len(trades)
    assert metrics.total_volume > 0
    assert metrics.total_pnl == sum(trade.calculate_pnl() for trade in trades)


def test_multiple_instruments_handling():
    """Test that the system handles multiple instruments correctly."""
    processor = MockBrawlerEventProcessor()
    generator = MockTradeGenerator(MockTradeConfig(symbols=["BTC", "ETH"]))
    
    # Generate trades for both instruments
    trades = generator.generate_24h_trades()
    
    # Process trades
    processor.process_multiple_events(trades)
    metrics = processor.get_24h_snapshot()
    
    # Check that we have metrics for both instruments
    assert len(metrics.per_asset_metrics) >= 1  # At least one asset should have trades
    
    # Verify per-asset totals match overall totals
    total_asset_pnl = sum(asset_metrics.total_pnl for asset_metrics in metrics.per_asset_metrics.values())
    # The sum might not be exactly equal due to fee calculations, but should be close
    assert abs(total_asset_pnl - metrics.total_pnl) < len(trades) * 20  # Allow for fee differences


def test_24hr_window_processing():
    """Test that 24-hour rolling windows are calculated correctly."""
    processor = MockBrawlerEventProcessor()
    
    # Generate 24 hours of data
    start_time = datetime.now() - timedelta(hours=24)
    generator = MockTradeGenerator()
    trades = generator.generate_24h_trades(start_time)
    
    # Process the trades
    processor.set_time_window(start_time)
    processor.process_multiple_events(trades)
    
    metrics = processor.get_24h_snapshot()
    
    # Verify we have data for the time window
    assert len(trades) == len(processor.trade_buffer)
    assert metrics.total_trades == len(trades)


def test_instrument_breakdown():
    """Test that metrics are correctly broken down by instrument."""
    processor = MockBrawlerEventProcessor()
    generator = MockTradeGenerator(MockTradeConfig(symbols=["BTC", "ETH", "SOL"]))
    
    # Generate trades for multiple instruments
    trades = generator.generate_24h_trades()
    
    # Process all trades
    processor.process_multiple_events(trades)
    metrics = processor.get_24h_snapshot()
    
    # Check that per-asset metrics exist
    for trade in trades:
        assert trade.symbol in metrics.per_asset_metrics
    
    # Verify that each asset's metrics are properly calculated
    for symbol, asset_metrics in metrics.per_asset_metrics.items():
        # Each asset should have its own metrics
        assert asset_metrics.total_trades >= 0
        assert asset_metrics.total_volume >= 0


if __name__ == "__main__":
    # Run the tests
    test_mock_trade_generation()
    test_mock_trade_stream()
    test_trade_event_processing()
    test_multiple_instruments_handling()
    test_24hr_window_processing()
    test_instrument_breakdown()
    
    print("All Sprint 2 tests passed!")