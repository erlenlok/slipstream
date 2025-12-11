
import time
from typing import Optional, Dict
from dataclasses import dataclass
from .feeds import LocalQuote

@dataclass
class AlphaState:
    fear_side: Optional[str] = None  # 'bid', 'ask', 'both'
    momentum_signal: bool = False
    insider_signal: bool = False

class ReplenishmentTracker:
    """Alpha 2: Tracks order book replenishment velocity to detect MM fear."""
    
    def __init__(self, recovery_threshold: float = 0.8, timeout_seconds: float = 2.0):
        self.recovery_threshold = recovery_threshold
        self.timeout_seconds = timeout_seconds
        
        # Trackers per side: {side: {price, pre_size, start_ts}}
        self.active_consumption: Dict[str, Optional[Dict]] = {
            'bid': None,
            'ask': None
        }
        
        # Active signals per side
        self.active_fear: Dict[str, bool] = {
            'bid': False, # Bid Fear = Support vanished (Bearish)
            'ask': False  # Ask Fear = Resistance vanished (Bullish)
        }
        
        self._prev_quote: Optional[LocalQuote] = None
        self._last_signal_ts: Dict[str, float] = {'bid': 0.0, 'ask': 0.0}
        self._signal_duration: float = 5.0

    def on_quote(self, quote: LocalQuote) -> None:
        now = quote.ts
        
        # 1. Auto-reset signals
        for side in ['bid', 'ask']:
            if self.active_fear[side]:
                if now - self._last_signal_ts[side] > self._signal_duration:
                    self.active_fear[side] = False

        if not self._prev_quote:
            self._prev_quote = quote
            return

        # 2. Process Both Sides
        self._process_side(
            'bid', 
            quote.bid, quote.bid_sz, 
            self._prev_quote.bid, self._prev_quote.bid_sz,
            now
        )
        self._process_side(
            'ask', 
            quote.ask, quote.ask_sz, 
            self._prev_quote.ask, self._prev_quote.ask_sz,
            now
        )

        self._prev_quote = quote

    def _process_side(self, side: str, 
                      curr_px: float, curr_sz: float, 
                      prev_px: float, prev_sz: float, 
                      now: float) -> None:
        
        tracker = self.active_consumption[side]
        
        # A. Already Tracking
        if tracker:
            # 1. Price Check (Must hold level)
            if curr_px != tracker['price']:
                # Level moved -> Cancel tracking
                self.active_consumption[side] = None
                return

            # 2. Size Check (Replenishment)
            target = tracker['pre_size'] * self.recovery_threshold
            if curr_sz >= target:
                # Success!
                self.active_consumption[side] = None
                return
            
            # 3. Timeout Check
            if now - tracker['start_ts'] > self.timeout_seconds:
                # FAILURE -> FEAR on this side
                self.active_fear[side] = True
                self._last_signal_ts[side] = now
                self.active_consumption[side] = None
                return
        
        # B. Not Tracking -> Detect new consumption
        else:
            if curr_px == prev_px:
                # Same level, significant drop?
                if curr_sz < prev_sz * 0.9: 
                    self.active_consumption[side] = {
                        'price': curr_px,
                        'pre_size': prev_sz,
                        'start_ts': now
                    }

class AlphaEngine:
    """Orchestrates multiple alpha signals."""
    def __init__(self, config=None):
        self.trackers: Dict[str, ReplenishmentTracker] = {}
        self.states: Dict[str, AlphaState] = {}

    def on_local_quote(self, quote: LocalQuote) -> AlphaState:
        if quote.symbol not in self.trackers:
            self.trackers[quote.symbol] = ReplenishmentTracker()
            self.states[quote.symbol] = AlphaState()
            
        tracker = self.trackers[quote.symbol]
        tracker.on_quote(quote)
        
        # Update State with Directionality
        state = self.states[quote.symbol]
        bid_fear = tracker.active_fear['bid']
        ask_fear = tracker.active_fear['ask']
        
        if bid_fear and ask_fear:
            state.fear_side = 'both'
        elif bid_fear:
            state.fear_side = 'bid'
        elif ask_fear:
            state.fear_side = 'ask'
        else:
            state.fear_side = None
        
        return state
