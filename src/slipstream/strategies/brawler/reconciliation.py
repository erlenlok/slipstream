
import asyncio
import logging
import time
from typing import Dict, Any, List

from .state import AssetState, OrderSnapshot
from .connectors import HyperliquidInfoClient, HyperliquidOrderSide

logger = logging.getLogger(__name__)

class OrderReconciler:
    """
    Periodically polls the exchange for open orders and enforces consistency
    with the local AssetState.
    
    1. Phantom Orders: Local state has order, Remote does not -> Clear Local.
    2. Orphan Orders: Remote has order, Local does not -> Adopt Local (so we can cancel it).
    """
    def __init__(
        self,
        api: HyperliquidInfoClient,
        wallet: str,
        interval_seconds: float = 15.0,
    ) -> None:
        self.api = api
        self.wallet = wallet
        self.interval = interval_seconds
        
    async def reconcile(self, states: Dict[str, AssetState]) -> None:
        """Main reconciliation pass."""
        try:
            # 1. Fetch Remote State
            raw_orders = await self.api.get_open_orders(self.wallet)
            if raw_orders is None:
                logger.warning("Reconciliation skipped: Open orders fetch failed.")
                return
            
            # Map: symbol -> list of remote orders
            remote_map: Dict[str, List[Dict[str, Any]]] = {}
            for o in raw_orders:
                sym = o.get("coin")
                if sym:
                    if sym not in remote_map:
                        remote_map[sym] = []
                    remote_map[sym].append(o)
            
            # 2. Iterate Local States
            for symbol, state in states.items():
                remote_list = remote_map.get(symbol, [])
                
                # Check IDs of remote orders
                remote_ids = {str(o.get("oid")): o for o in remote_list}
                
                # --- PHANTOM CHECK (Bid) ---
                if state.active_bid:
                    if state.active_bid.order_id not in remote_ids:
                        logger.warning(
                            "👻 Phantom BID detected for %s (id=%s). Clearing state to resume quoting.",
                            symbol, state.active_bid.order_id
                        )
                        state.active_bid = None
                
                # --- PHANTOM CHECK (Ask) ---
                if state.active_ask:
                    if state.active_ask.order_id not in remote_ids:
                        logger.warning(
                            "👻 Phantom ASK detected for %s (id=%s). Clearing state to resume quoting.",
                            symbol, state.active_ask.order_id
                        )
                        state.active_ask = None
                        
                # --- ORPHAN ADOPTION ---
                # If there are open orders on remote that match our Brawler logic (limit orders)
                # but we don't track them, we should adopt them so _ensure_orders can cancel them if needed.
                # However, multiple orders per side isn't fully supported by simple AssetState (active_bid is singular).
                # If we have multiple, Brawler might get confused. 
                # Strategy: If we have NO active_bid, but remote has one, adopt it.
                
                for oid, o_data in remote_ids.items():
                    side_str = o_data.get("side") # "B" or "A"? No, API returns "B"/"A" usually or "buy"/"sell"?
                    # frontendOpenOrders returns "side": "B" (Check script output?)
                    # Wait, Hyperliquid API usually returns "B" or "A" in some endpoints, "buy"/"sell" in others.
                    # Let's assume standard "B"/"A" from frontendOpenOrders or check SDK.
                    # SDK `info.open_orders` returns the raw list.
                    
                    # Assuming 'side' is 'B' or 'A' based on typical HL API.
                    # Let's be robust.
                    is_buy = side_str == "B" or side_str == "buy"
                    
                    if is_buy:
                        if not state.active_bid:
                            logger.info("👶 Adopting orphan BID for %s (id=%s)", symbol, oid)
                            state.active_bid = OrderSnapshot(
                                order_id=oid,
                                price=float(o_data.get("limitPx", 0)),
                                size=float(o_data.get("sz", 0)),
                                side=HyperliquidOrderSide.BUY
                            )
                    else:
                        if not state.active_ask:
                            logger.info("👶 Adopting orphan ASK for %s (id=%s)", symbol, oid)
                            state.active_ask = OrderSnapshot(
                                order_id=oid,
                                price=float(o_data.get("limitPx", 0)),
                                size=float(o_data.get("sz", 0)),
                                side=HyperliquidOrderSide.SELL
                            )

        except Exception as e:
            logger.error("Reconciliation failed: %s", e, exc_info=True)
