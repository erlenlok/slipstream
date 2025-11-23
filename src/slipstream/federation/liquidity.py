"""
Liquidity Surface Mapping for Capacity-Based Allocation.

This module implements live modeling of "Slippage vs. Size" for each asset using
Square-Root Law impact models, enabling capacity-based position sizing as specified
in the federated vision document.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class MarketDataPoint:
    """Represents a single market data point used for liquidity analysis."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float  # 24h volume or other volume measure
    bid_price: float
    ask_price: float
    spread: float  # bid-ask spread
    bid_size: float
    ask_size: float
    total_liquidity: float  # Total depth in order book


@dataclass
class TradeSizeSlippage:
    """Represents a trade size and observed slippage for model calibration."""
    symbol: str
    timestamp: datetime
    trade_size: float  # In quote currency or base currency
    realized_slippage: float  # In bps or percentage
    market_impact: float  # Price impact from the trade
    execution_style: str  # 'maker', 'taker_aggressive', 'taker_passive', etc.
    liquidity_conditions: str  # 'high', 'medium', 'low' liquidity at time of trade


@dataclass
class LiquidityModel:
    """Parameters for a liquidity impact model."""
    symbol: str
    timestamp: datetime
    model_type: str  # 'square_root', 'linear', 'power_law', etc.
    impact_coefficient: float  # Coefficient for impact calculation
    volatility_factor: float  # Adjusts for market volatility
    liquidity_factor: float  # Adjusts for market liquidity
    r_squared: float  # Goodness of fit measure
    last_updated: datetime


@dataclass
class CapacityAnalysisResult:
    """Result of capacity analysis for an asset."""
    symbol: str
    timestamp: datetime
    current_price: float
    estimated_capacity: float  # Maximum position size before significant impact
    max_position_size: float  # Recommended maximum position size based on liquidity
    slippage_at_max_size: float  # Expected slippage at max position size
    liquidity_score: float  # 0-100 score of market liquidity
    volatility_adjusted_capacity: float  # Capacity adjusted for current volatility
    model_confidence: float  # Confidence in the liquidity model
    capacity_trend: str  # 'increasing', 'decreasing', 'stable'


@dataclass
class PositionSizeRecommendation:
    """Recommendation for capacity-based position sizing."""
    strategy_id: str
    symbol: str
    recommended_size: float
    max_allowed_size: float
    current_liquidity_estimate: float
    safety_factor: float  # Factor to reduce position size for safety
    reason: str  # Reason for the recommendation


class MarketDataProvider(Protocol):
    """
    Protocol for data providers that the liquidity mapper can use to
    collect market data.
    """
    async def get_market_data(self, symbols: List[str],
                             start_time: datetime,
                             end_time: datetime) -> Dict[str, List[MarketDataPoint]]:
        """Get market data for specified symbols."""
        ...

    async def get_trade_data(self, symbols: List[str],
                            start_time: datetime,
                            end_time: datetime) -> Dict[str, List[TradeSizeSlippage]]:
        """Get trade data with size and slippage information."""
        ...

    async def get_current_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions for a symbol."""
        ...


class LiquiditySurfaceMapper:
    """
    Maps liquidity surfaces to model "Slippage vs. Size" using Square-Root Law impact models.
    
    This component enables capacity-based position sizing by modeling how trade size
    affects slippage/impact in real-time, as specified in the federated vision.
    """
    
    def __init__(self,
                 data_provider: Optional[MarketDataProvider] = None,
                 min_data_points_for_model: int = 10,
                 default_safety_factor: float = 0.5,  # Use 50% of theoretical capacity
                 model_recalibration_interval: timedelta = timedelta(hours=1)):
        """
        Initialize the liquidity surface mapper.
        
        Args:
            data_provider: Optional data provider for market data
            min_data_points_for_model: Minimum points needed to build reliable model
            default_safety_factor: Factor to reduce position sizes for safety
            model_recalibration_interval: How often to recalibrate models
        """
        self.data_provider = data_provider
        self.min_data_points_for_model = min_data_points_for_model
        self.default_safety_factor = default_safety_factor
        self.model_recalibration_interval = model_recalibration_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Store data for modeling
        self._market_data: Dict[str, List[MarketDataPoint]] = {}  # symbol -> market data
        self._trade_slippage_data: Dict[str, List[TradeSizeSlippage]] = {}  # symbol -> trade slippage
        self._current_models: Dict[str, LiquidityModel] = {}  # symbol -> current model
        self._capacity_history: Dict[str, List[CapacityAnalysisResult]] = {}  # symbol -> capacity analysis
        self._last_calibration: Dict[str, datetime] = {}  # symbol -> last calibration time
        
        # Square-root law model parameters
        self._sqrt_model_formula = "impact = k * sqrt(volume) * volatility_factor / sqrt(liquidity_factor)"

    async def start(self):
        """Start the liquidity surface mapper."""
        self._running = True
        self.logger.info("Liquidity Surface Mapper started")
        
        if self.data_provider:
            # Initial data loading would happen here in a real system
            pass
        
        self.logger.info("Liquidity surface mapping started")

    async def stop(self):
        """Stop the liquidity surface mapper."""
        self._running = False
        self.logger.info("Liquidity Surface Mapper stopped")

    async def record_market_data(self, data: MarketDataPoint):
        """Record market data point for liquidity analysis."""
        symbol = data.symbol
        
        if symbol not in self._market_data:
            self._market_data[symbol] = []
            
        self._market_data[symbol].append(data)
        self.logger.debug(f"Recorded market data for {symbol}: volume={data.volume}, spread={data.spread:.4f}")

    async def record_trade_slippage(self, trade: TradeSizeSlippage):
        """Record trade size and observed slippage for model calibration."""
        symbol = trade.symbol
        
        if symbol not in self._trade_slippage_data:
            self._trade_slippage_data[symbol] = []
            
        self._trade_slippage_data[symbol].append(trade)
        self.logger.debug(f"Recorded trade slippage for {symbol}: size={trade.trade_size}, slippage={trade.realized_slippage:.4f}")

    async def build_liquidity_model(self, symbol: str) -> Optional[LiquidityModel]:
        """
        Build or update the liquidity impact model for a symbol using Square-Root Law.
        
        Square-root law: impact = k * sqrt(volume) * volatility_factor / sqrt(liquidity_factor)
        
        Args:
            symbol: The symbol to model
            
        Returns:
            Liquidity model parameters or None if insufficient data
        """
        # Get trade data for this symbol
        trade_data = self._trade_slippage_data.get(symbol, [])
        
        if len(trade_data) < self.min_data_points_for_model:
            self.logger.warning(f"Insufficient trade data for {symbol}: {len(trade_data)} points (min {self.min_data_points_for_model})")
            return None
        
        # Prepare data for fitting
        sizes = []
        slippages = []
        volatilities = []
        liquidities = []
        
        for trade in trade_data:
            sizes.append(trade.trade_size)
            slippages.append(abs(trade.realized_slippage))  # Use absolute slippage
            # For volatility and liquidity factors, we could use market data or assume 1.0 for now
            volatilities.append(1.0)  # Simplified - could be derived from market data
            liquidities.append(1.0)   # Simplified - could be derived from market data
        
        if len(sizes) < 2:
            return None
        
        try:
            # Fit square root model: impact = k * sqrt(volume) * volatility_factor / sqrt(liquidity_factor)
            # For now, we'll fit: y = a * sqrt(x) where y is slippage and x is trade_size
            sizes_array = np.array(sizes)
            slippages_array = np.array(slippages)
            
            # Define the square root function for fitting
            def sqrt_model(x, a):
                return a * np.sqrt(x)
            
            # Fit the model
            popt, pcov = curve_fit(sqrt_model, sizes_array, slippages_array, p0=[0.001])
            impact_coefficient = popt[0]
            
            # Calculate R-squared
            fitted_values = sqrt_model(sizes_array, impact_coefficient)
            ss_res = np.sum((slippages_array - fitted_values) ** 2)
            ss_tot = np.sum((slippages_array - np.mean(slippages_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            # Create model object
            model = LiquidityModel(
                symbol=symbol,
                timestamp=datetime.now(),
                model_type="square_root",
                impact_coefficient=impact_coefficient,
                volatility_factor=1.0,  # Simplified
                liquidity_factor=1.0,   # Simplified
                r_squared=r_squared,
                last_updated=datetime.now()
            )
            
            # Store the model
            self._current_models[symbol] = model
            
            self.logger.info(f"Built liquidity model for {symbol}: k={impact_coefficient:.6f}, R²={r_squared:.3f}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building liquidity model for {symbol}: {e}")
            return None

    async def get_capacity_analysis(self, symbol: str) -> Optional[CapacityAnalysisResult]:
        """
        Get capacity analysis for an asset based on the liquidity model.
        
        Args:
            symbol: The symbol to analyze
            
        Returns:
            Capacity analysis result or None if insufficient data
        """
        # Get current market data
        market_data_list = self._market_data.get(symbol, [])
        if not market_data_list:
            return None
        
        current_data = market_data_list[-1]  # Latest data point
        
        # Get the current model (or build a new one if needed)
        model = self._current_models.get(symbol)
        if not model:
            model = await self.build_liquidity_model(symbol)
        
        if not model:
            return None
        
        # Calculate estimated capacity based on the model
        # For the square root model, we can solve for the size at which 
        # slippage reaches a certain threshold (e.g., 10 bps = 0.001)
        slippage_threshold = 0.001  # 10 bps
        
        # From: slippage = k * sqrt(size) => size = (slippage / k) ^ 2
        estimated_capacity = ((slippage_threshold / model.impact_coefficient) ** 2) if model.impact_coefficient > 0 else float('inf')
        
        # Calculate max position as a percentage of estimated capacity
        # Use a safety factor to avoid pushing up against the limit
        max_position_size = estimated_capacity * self.default_safety_factor
        
        # Calculate expected slippage at max position size
        expected_slippage = model.impact_coefficient * np.sqrt(max_position_size)
        
        # Calculate liquidity score (0-100) based on various factors
        liquidity_score = self._calculate_liquidity_score(current_data, model)
        
        # Adjust for volatility conditions
        volatility_adjusted_capacity = max_position_size * (1.0 if current_data.spread < 0.001 else 0.8)  # Crude volatility adjustment
        
        result = CapacityAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_data.price,
            estimated_capacity=estimated_capacity,
            max_position_size=max_position_size,
            slippage_at_max_size=expected_slippage,
            liquidity_score=liquidity_score,
            volatility_adjusted_capacity=volatility_adjusted_capacity,
            model_confidence=model.r_squared,
            capacity_trend="stable"  # Could be calculated based on historical capacity estimates
        )
        
        # Store the result
        if symbol not in self._capacity_history:
            self._capacity_history[symbol] = []
        self._capacity_history[symbol].append(result)
        
        self.logger.info(f"Capacity analysis for {symbol}: est_cap={estimated_capacity:.2f}, "
                        f"max_pos={max_position_size:.2f}, liquidity_score={liquidity_score:.1f}")
        
        return result

    def _calculate_liquidity_score(self, market_data: MarketDataPoint, model: LiquidityModel) -> float:
        """Calculate a liquidity score from 0-100 based on market conditions and model quality."""
        # Base score on spread (lower spread = higher liquidity)
        spread_score = max(0, 100 - (market_data.spread / market_data.price) * 100000)  # Spread in bps
        
        # Factor in model confidence
        model_score = model.r_squared * 100
        
        # Factor in volume (higher volume = higher liquidity)
        volume_score = min(100, (market_data.volume / 1000000) * 50)  # Scale based on typical volume
        
        # Combine scores (with weights)
        total_score = (spread_score * 0.5) + (model_score * 0.3) + (volume_score * 0.2)
        return max(0, min(100, total_score))  # Cap between 0 and 100

    async def get_position_size_recommendation(self, 
                                             strategy_id: str,
                                             symbol: str,
                                             base_suggestion: float,
                                             safety_factor: Optional[float] = None) -> Optional[PositionSizeRecommendation]:
        """
        Get capacity-based position sizing recommendation.
        
        Args:
            strategy_id: The strategy requesting the position size
            symbol: The symbol to trade
            base_suggestion: Base position size suggestion (could be from alpha model)
            safety_factor: Optional override for safety factor (defaults to config)
            
        Returns:
            Position size recommendation
        """
        capacity_analysis = await self.get_capacity_analysis(symbol)
        if not capacity_analysis:
            return None
        
        # Determine safety factor to use
        factor_to_use = safety_factor if safety_factor is not None else self.default_safety_factor
        
        # Cap the position size based on liquidity capacity
        max_allowed_size = capacity_analysis.volatility_adjusted_capacity
        
        # Apply safety factor to the maximum allowed size
        recommended_size = min(base_suggestion, max_allowed_size * factor_to_use)
        
        # Ensure recommendation is reasonable (not negative or zero)
        recommended_size = max(0, recommended_size)
        
        recommendation = PositionSizeRecommendation(
            strategy_id=strategy_id,
            symbol=symbol,
            recommended_size=recommended_size,
            max_allowed_size=max_allowed_size,
            current_liquidity_estimate=capacity_analysis.estimated_capacity,
            safety_factor=factor_to_use,
            reason=f"Liquidity-based capping: original={base_suggestion:.2f}, "
                   f"capped_at={max_allowed_size:.2f} with safety_factor={factor_to_use}"
        )
        
        self.logger.info(f"Position recommendation for {strategy_id} {symbol}: "
                        f"recommended={recommended_size:.2f}, max_allowed={max_allowed_size:.2f}")
        
        return recommendation

    async def get_liquidity_surface_snapshot(self, symbols: Optional[List[str]] = None) -> Dict[str, CapacityAnalysisResult]:
        """
        Get a snapshot of liquidity conditions for multiple symbols.
        
        Args:
            symbols: List of symbols to analyze (defaults to all tracked symbols)
            
        Returns:
            Dictionary mapping symbol to capacity analysis
        """
        if symbols is None:
            # Get all symbols we have market data for
            symbols = list(set(list(self._market_data.keys())))
        
        snapshot = {}
        for symbol in symbols:
            analysis = await self.get_capacity_analysis(symbol)
            if analysis:
                snapshot[symbol] = analysis
        
        return snapshot

    async def update_model_if_needed(self, symbol: str) -> bool:
        """
        Update the liquidity model if it's time for recalibration.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            True if model was updated, False otherwise
        """
        last_cal = self._last_calibration.get(symbol, datetime.min)
        if datetime.now() - last_cal > self.model_recalibration_interval:
            model = await self.build_liquidity_model(symbol)
            if model:
                self._last_calibration[symbol] = datetime.now()
                return True
        return False

    async def get_liquidity_trends(self, symbol: str, days: int = 7) -> List[CapacityAnalysisResult]:
        """
        Get liquidity trends for a symbol over time.
        
        Args:
            symbol: The symbol to analyze
            days: Number of days to look back
            
        Returns:
            List of historical capacity analysis results
        """
        if symbol not in self._capacity_history:
            return []
            
        cutoff_date = datetime.now() - timedelta(days=days)
        trends = [r for r in self._capacity_history[symbol] if r.timestamp >= cutoff_date]
        
        # Sort by date
        trends.sort(key=lambda x: x.timestamp)
        return trends

    async def get_capacity_alerts(self) -> List[Dict[str, Any]]:
        """
        Get alerts for symbols with low liquidity or capacity issues.
        
        Returns:
            List of capacity alerts
        """
        alerts = []
        
        # Check all symbols we're tracking
        all_symbols = set(list(self._market_data.keys()))
        
        for symbol in all_symbols:
            analysis = await self.get_capacity_analysis(symbol)
            if not analysis:
                continue
                
            # Alert if liquidity score is low
            if analysis.liquidity_score < 20:  # Very low liquidity
                alerts.append({
                    'symbol': symbol,
                    'alert_type': 'LOW_LIQUIDITY',
                    'severity': 'HIGH',
                    'message': f'Very low liquidity detected for {symbol} (score: {analysis.liquidity_score:.1f})',
                    'current_price': analysis.current_price,
                    'suggested_position_limit': analysis.max_position_size,
                    'timestamp': analysis.timestamp
                })
            elif analysis.liquidity_score < 50:  # Low liquidity
                alerts.append({
                    'symbol': symbol,
                    'alert_type': 'LOW_LIQUIDITY',
                    'severity': 'MEDIUM',
                    'message': f'Low liquidity detected for {symbol} (score: {analysis.liquidity_score:.1f})',
                    'current_price': analysis.current_price,
                    'suggested_position_limit': analysis.max_position_size,
                    'timestamp': analysis.timestamp
                })
        
        return alerts


class MockMarketDataProvider:
    """
    Mock data provider for testing purposes.
    """
    
    def __init__(self):
        self._market_data = {}
        self._trade_data = {}
        
    async def get_market_data(self, symbols: List[str],
                             start_time: datetime,
                             end_time: datetime) -> Dict[str, List[MarketDataPoint]]:
        for symbol in symbols:
            if symbol not in self._market_data:
                # Generate mock market data
                data = []
                base_time = start_time
                base_price = 45000.0 if symbol == "BTC" else 3000.0
                
                for i in range(50):  # 50 data points
                    spread = np.random.uniform(0.1, 2.0)  # Random spread
                    data.append(MarketDataPoint(
                        symbol=symbol,
                        timestamp=base_time + timedelta(minutes=i*30),  # Every 30 minutes
                        price=base_price * (1 + np.random.normal(0, 0.002)),
                        volume=np.random.uniform(1000000, 5000000),  # Random volume
                        bid_price=base_price - spread/2,
                        ask_price=base_price + spread/2,
                        spread=spread,
                        bid_size=np.random.uniform(1, 10),
                        ask_size=np.random.uniform(1, 10),
                        total_liquidity=np.random.uniform(100, 500)
                    ))
                    base_price = data[-1].price
                    
                self._market_data[symbol] = data
        
        result = {}
        for symbol in symbols:
            result[symbol] = [d for d in self._market_data[symbol] 
                             if start_time <= d.timestamp <= end_time]
        return result

    async def get_trade_data(self, symbols: List[str],
                            start_time: datetime,
                            end_time: datetime) -> Dict[str, List[TradeSizeSlippage]]:
        for symbol in symbols:
            if symbol not in self._trade_data:
                # Generate mock trade data that follows square root law
                data = []
                base_time = start_time
                
                for i in range(30):  # 30 trade data points
                    trade_size = np.random.uniform(0.1, 5.0)  # Random trade size
                    # Generate slippage following square root law with some noise
                    base_coefficient = 0.001 if symbol == "BTC" else 0.005  # BTC more liquid
                    base_slippage = base_coefficient * np.sqrt(trade_size)
                    noise = np.random.normal(0, base_slippage * 0.2)  # 20% noise
                    realized_slippage = base_slippage + noise
                    
                    data.append(TradeSizeSlippage(
                        symbol=symbol,
                        timestamp=base_time + timedelta(minutes=i*60),
                        trade_size=trade_size,
                        realized_slippage=realized_slippage,
                        market_impact=realized_slippage,
                        execution_style="taker",
                        liquidity_conditions="medium"
                    ))
                
                self._trade_data[symbol] = data
        
        result = {}
        for symbol in symbols:
            result[symbol] = [d for d in self._trade_data[symbol] 
                             if start_time <= d.timestamp <= end_time]
        return result

    async def get_current_market_conditions(self, symbol: str) -> Dict[str, Any]:
        return {
            "price": 45000.0,
            "spread_percent": 0.05,
            "volume_24h": 25000000000,
            "volatility": 0.02
        }


__all__ = [
    "LiquiditySurfaceMapper",
    "MockMarketDataProvider",
    "MarketDataPoint",
    "TradeSizeSlippage",
    "LiquidityModel",
    "CapacityAnalysisResult",
    "PositionSizeRecommendation",
    "MarketDataProvider"
]