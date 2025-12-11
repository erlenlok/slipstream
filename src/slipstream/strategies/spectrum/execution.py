"""
Execution Bridge for Spectrum Strategy: Two-Stage Timing with Beta Hedging

This module implements the conversion of idio-weights to tradeable orders with 
strict beta neutrality and two-stage timing as specified in the Spectrum strategy.

Reference: spectrum_spec.md - Module E: Execution Bridge
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio
import time
from dataclasses import dataclass


@dataclass
class ExecutionTask:
    """Represents a single execution task."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    target_price: Optional[float] = None
    order_type: str = 'MARKET'  # or 'LIMIT'
    fill_threshold: Optional[float] = None  # For limit orders


@dataclass
class PortfolioPosition:
    """Represents a current portfolio position."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    beta_exposure: float


class HedgeManager:
    """
    Manages beta hedging based on confirmed fills, not projected positions.
    """
    
    def __init__(self, hedge_threshold: float = 0.01):
        """
        Initialize the hedge manager.
        
        Args:
            hedge_threshold: Threshold for net beta exposure that triggers hedging (default 0.01 = 1%)
        """
        self.hedge_threshold = hedge_threshold
        self.btc_beta_per_share = 1.0  # By definition
        self.eth_beta_per_share = 1.0  # By definition
        
    def calculate_beta_exposure(
        self, 
        positions: Dict[str, PortfolioPosition], 
        betas: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculate total BTC and ETH beta exposure from current positions.
        
        Args:
            positions: Current portfolio positions
            betas: Beta coefficients for each asset
            
        Returns:
            Tuple of (btc_beta_exposure, eth_beta_exposure)
        """
        total_btc_beta = 0.0
        total_eth_beta = 0.0
        
        for symbol, pos in positions.items():
            if symbol in betas:
                asset_beta_btc = betas[symbol]  # This is the BTC beta component
                # For simplicity, assume the beta represents how much of the systematic risk
                # is related to BTC/ETH. In the real model, we might have separate BTC and ETH betas.
                # For now, assume betas represent the BTC beta, and we hedge with BTC
                total_btc_beta += pos.quantity * asset_beta_btc
                
        return total_btc_beta, total_eth_beta
    
    def generate_hedge_orders(
        self, 
        current_positions: Dict[str, PortfolioPosition], 
        betas: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[ExecutionTask]:
        """
        Generate hedge orders based on confirmed fills.
        
        Args:
            current_positions: Current portfolio positions
            betas: Beta coefficients for each asset
            current_prices: Current market prices
            
        Returns:
            List of ExecutionTask objects for hedging
        """
        btc_beta, eth_beta = self.calculate_beta_exposure(current_positions, betas)
        net_beta = abs(btc_beta)  # Simplified: just BTC beta for now
        
        hedge_orders = []
        
        if net_beta > self.hedge_threshold:
            # Determine hedge direction based on sign of beta exposure
            hedge_direction = 'SELL' if btc_beta > 0 else 'BUY'
            hedge_quantity = abs(btc_beta)  # Hedge with same magnitude as exposure
            
            if hedge_direction == 'SELL':
                hedge_quantity = -hedge_quantity  # Negative for sell
            
            hedge_orders.append(ExecutionTask(
                symbol='BTC',
                side=hedge_direction,
                quantity=abs(hedge_quantity),
                order_type='MARKET'
            ))
        
        return hedge_orders


class SpectrumExecutionBridge:
    """
    Main execution bridge for the Spectrum strategy.
    
    Handles the two-stage execution (23:50 projected, 00:01 correction) and beta hedging.
    """
    
    def __init__(
        self,
        account_equity: float,
        hedge_threshold: float = 0.01,
        twap_duration_minutes: int = 15
    ):
        """
        Initialize the execution bridge.
        
        Args:
            account_equity: Total account equity
            hedge_threshold: Threshold for triggering beta hedging
            twap_duration_minutes: Duration for TWAP orders (default 15 min)
        """
        self.account_equity = account_equity
        self.hedge_manager = HedgeManager(hedge_threshold)
        self.twap_duration_minutes = twap_duration_minutes
        self.current_positions: Dict[str, PortfolioPosition] = {}
        self.pending_orders: List[ExecutionTask] = []
        
    def convert_weights_to_orders(
        self,
        target_weights: pd.Series,
        current_prices: Dict[str, float],
        position_quantities: Optional[Dict[str, float]] = None
    ) -> List[ExecutionTask]:
        """
        Convert target idio-weights to execution tasks.
        
        Args:
            target_weights: Target portfolio weights (idio-weights)
            current_prices: Current market prices for all assets
            position_quantities: Current position quantities (if None, assumes 0)
            
        Returns:
            List of ExecutionTask objects to execute
        """
        if position_quantities is None:
            position_quantities = {symbol: 0.0 for symbol in target_weights.index}
        
        orders = []
        
        for symbol in target_weights.index:
            if symbol not in current_prices:
                continue  # Skip if no price available
                
            target_value = target_weights[symbol] * self.account_equity
            current_value = position_quantities.get(symbol, 0.0) * current_prices[symbol]
            
            # Calculate required quantity change
            required_value_change = target_value - current_value
            required_quantity_change = required_value_change / current_prices[symbol]
            
            if abs(required_quantity_change) > 1e-6:  # Only create orders if meaningful change
                side = 'BUY' if required_quantity_change > 0 else 'SELL'
                quantity = abs(required_quantity_change)
                
                orders.append(ExecutionTask(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    target_price=current_prices[symbol],
                    order_type='MARKET'
                ))
        
        return orders
    
    def handle_asset_universe_changes(
        self,
        current_assets: List[str],
        target_weights: pd.Series,
        betas: pd.DataFrame
    ) -> pd.Series:
        """
        Handle new entrants and dropouts in the universe.

        - New entrants: Include in output with 0 weight
        - Dropouts: Remove from output (since their weight is forced to 0)

        Args:
            current_assets: List of currently active assets
            target_weights: Current target weights
            betas: Current beta coefficients

        Returns:
            Adjusted target weights with universe constraints applied
        """
        # Start with only assets that are currently active
        adjusted_weights = pd.Series(index=current_assets, dtype=float)

        # For active assets, use their target weights if they exist, otherwise 0
        for asset in current_assets:
            if asset in target_weights:
                adjusted_weights[asset] = target_weights[asset]
            else:
                # New entrant, initialize with 0 weight
                adjusted_weights[asset] = 0.0

        return adjusted_weights
    
    async def execute_stage_1_projected(
        self,
        target_weights: pd.Series,
        betas: pd.DataFrame,
        current_prices: Dict[str, float],
        current_positions: Dict[str, PortfolioPosition]
    ) -> Dict[str, Any]:
        """
        Execute Stage 1 (23:50 UTC): Projected phase using live mid prices.
        
        Args:
            target_weights: Target portfolio weights
            betas: Beta coefficients
            current_prices: Current live prices
            current_positions: Current portfolio positions
            
        Returns:
            Execution results
        """
        print("Starting Stage 1 (Projected) execution at 23:50...")
        
        # Convert weights to orders
        orders = self.convert_weights_to_orders(target_weights, current_prices, 
                                               {k: v.quantity for k, v in current_positions.items()})
        
        # Generate TWAP orders for projected execution
        twap_orders = []
        for order in orders:
            # Spread the order execution over the TWAP duration
            twap_orders.append(order)  # In a real system, we'd split these into smaller pieces
        
        self.pending_orders = twap_orders
        
        # Start TWAP execution (simulated here)
        await asyncio.sleep(0.1)  # Simulate some execution time
        
        results = {
            'stage': 'projected',
            'timestamp': datetime.now(),
            'orders_submitted': len(twap_orders),
            'total_value': sum(abs(order.quantity * order.target_price) for order in twap_orders if order.target_price),
            'status': 'started'
        }
        
        print(f"Stage 1: Submitted {len(twap_orders)} orders for projected execution")
        return results
    
    async def execute_stage_2_correction(
        self,
        final_weights: pd.Series,
        betas: pd.DataFrame,
        official_close_prices: Dict[str, float],
        current_positions: Dict[str, PortfolioPosition]
    ) -> Dict[str, Any]:
        """
        Execute Stage 2 (00:01 UTC): Correction phase with official close prices.
        
        Args:
            final_weights: Final target portfolio weights after official close
            betas: Beta coefficients  
            official_close_prices: Official close prices
            current_positions: Current portfolio positions after Stage 1 fills
            
        Returns:
            Execution results
        """
        print("Starting Stage 2 (Correction) execution at 00:01...")
        
        # Calculate correction orders: final - already_filled
        # For this implementation, we assume all stage 1 orders were partially filled
        # In reality, we'd check how much of each order was filled
        
        # For now, just calculate the difference between final weights and current positions
        current_quantities = {k: v.quantity for k, v in current_positions.items()}
        
        correction_orders = self.convert_weights_to_orders(
            final_weights, 
            official_close_prices, 
            current_quantities
        )
        
        # Execute correction orders
        for order in correction_orders:
            # Simulate execution
            await asyncio.sleep(0.01)
        
        results = {
            'stage': 'correction',
            'timestamp': datetime.now(),
            'correction_orders': len(correction_orders),
            'total_correction_value': sum(abs(order.quantity * order.target_price) for order in correction_orders if order.target_price),
            'status': 'completed'
        }
        
        print(f"Stage 2: Executed {len(correction_orders)} correction orders")
        return results
    
    async def execute_beta_hedging(
        self,
        current_positions: Dict[str, PortfolioPosition],
        betas: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[ExecutionTask]:
        """
        Execute beta hedging based on confirmed fills.
        
        Args:
            current_positions: Current portfolio positions (confirmed fills only)
            betas: Beta coefficients
            current_prices: Current market prices
            
        Returns:
            List of hedge execution tasks
        """
        # Generate hedge orders based on confirmed positions
        hedge_orders = self.hedge_manager.generate_hedge_orders(
            current_positions, betas, current_prices
        )
        
        # Execute hedge orders
        for order in hedge_orders:
            # Simulate hedge execution
            await asyncio.sleep(0.01)
        
        print(f"Executed {len(hedge_orders)} hedge orders")
        return hedge_orders
    
    async def run_full_execution_cycle(
        self,
        projected_weights: pd.Series,
        final_weights: pd.Series,
        betas: pd.DataFrame,
        projected_prices: Dict[str, float],
        final_prices: Dict[str, float],
        initial_positions: Optional[Dict[str, PortfolioPosition]] = None
    ) -> Dict[str, Any]:
        """
        Run the full execution cycle: Stage 1 -> wait -> Stage 2 -> hedge.
        
        Args:
            projected_weights: Weights for Stage 1 execution
            final_weights: Weights for Stage 2 execution  
            betas: Beta coefficients
            projected_prices: Projected prices for Stage 1
            final_prices: Final prices for Stage 2
            initial_positions: Initial portfolio positions
            
        Returns:
            Complete execution results
        """
        if initial_positions is None:
            initial_positions = {}
        
        # Stage 1: Projected execution
        stage1_results = await self.execute_stage_1_projected(
            projected_weights, betas, projected_prices, initial_positions
        )
        
        # Simulate waiting until Stage 2
        await asyncio.sleep(0.1)  # In reality, this would be until 00:01 UTC
        
        # For this simulation, assume positions have updated after stage 1
        # In a real system, we'd track actual fills
        current_positions_after_stage1 = initial_positions.copy()
        
        # Stage 2: Correction execution
        stage2_results = await self.execute_stage_2_correction(
            final_weights, betas, final_prices, current_positions_after_stage1
        )
        
        # Execute beta hedging based on final positions
        # For this simulation, assume final positions are based on final weights
        final_positions = {}
        for symbol, weight in final_weights.items():
            if symbol in final_prices and final_prices[symbol] > 0:
                value = weight * self.account_equity
                quantity = value / final_prices[symbol]
                final_positions[symbol] = PortfolioPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=final_prices[symbol],
                    market_value=value,
                    beta_exposure=0.0  # Will be calculated based on betas
                )
        
        hedge_orders = await self.execute_beta_hedging(
            final_positions, betas.to_dict('index')[betas.index[-1]] if not betas.empty else {}, 
            final_prices
        )
        
        return {
            'stage_1': stage1_results,
            'stage_2': stage2_results,
            'hedge_orders': len(hedge_orders),
            'total_execution_cycle': 'completed',
            'timestamp': datetime.now()
        }


def create_execution_schedule() -> Dict[str, datetime]:
    """
    Create the execution schedule for the two-stage timing.
    
    Returns:
        Dictionary with scheduled execution times
    """
    now = datetime.now()
    
    # Today at 23:50 UTC
    projected_time = now.replace(hour=23, minute=50, second=0, microsecond=0)
    if projected_time <= now:
        projected_time = projected_time + timedelta(days=1)  # Tomorrow if already past
    
    # Tomorrow at 00:01 UTC  
    correction_time = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if correction_time <= now:
        correction_time = correction_time + timedelta(days=1)  # Tomorrow if already past
    
    return {
        'stage_1_projected': projected_time,
        'stage_2_correction': correction_time
    }