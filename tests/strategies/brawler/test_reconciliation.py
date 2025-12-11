
import pytest
from unittest.mock import MagicMock, AsyncMock
from slipstream.strategies.brawler.reconciliation import OrderReconciler
from slipstream.strategies.brawler.state import AssetState, OrderSnapshot
from slipstream.strategies.brawler.connectors import HyperliquidOrderSide

@pytest.fixture
def api_mock():
    return MagicMock()

@pytest.fixture
def wallet():
    return "0xWallet"

@pytest.fixture
def reconciler(api_mock, wallet):
    return OrderReconciler(api_mock, wallet, interval_seconds=0.1)

@pytest.fixture
def mock_state():
    state = AssetState(MagicMock(symbol="FARTCOIN"))
    state.active_bid = OrderSnapshot(
        order_id="1001",
        price=10.0,
        size=100.0,
        side=HyperliquidOrderSide.BUY
    )
    return state

@pytest.mark.asyncio
async def test_reconcile_phantom_order_clearing(reconciler, api_mock, mock_state):
    """Test that a local order NOT in remote list is cleared (Phantom)."""
    # 1. Setup: API returns NO orders
    api_mock.get_open_orders = AsyncMock(return_value=[])
    
    states = {"FARTCOIN": mock_state}
    
    # 2. Run
    await reconciler.reconcile(states)
    
    # 3. Verify
    assert mock_state.active_bid is None, "Phantom bid should have been cleared"

@pytest.mark.asyncio
async def test_reconcile_confirmed_order_stays(reconciler, api_mock, mock_state):
    """Test that a local order FOUND in remote list implies no change."""
    # 1. Setup: API returns matching order
    # Note: Structure depends on what get_open_orders returns (list of dicts)
    api_mock.get_open_orders = AsyncMock(return_value=[{
        "coin": "FARTCOIN",
        "oid": 1001, # Matches
        "limitPx": "10.0",
        "sz": "100.0",
        "side": "B"
    }])
    
    states = {"FARTCOIN": mock_state}
    
    await reconciler.reconcile(states)
    
    assert mock_state.active_bid is not None
    assert mock_state.active_bid.order_id == "1001"

@pytest.mark.asyncio
async def test_reconcile_orphan_adoption(reconciler, api_mock, mock_state):
    """Test that an untracked remote order is adopted."""
    mock_state.active_bid = None # No local order
    
    api_mock.get_open_orders = AsyncMock(return_value=[{
        "coin": "FARTCOIN",
        "oid": 9999,
        "limitPx": "50.0",
        "sz": "5.0",
        "side": "B"
    }])
    
    states = {"FARTCOIN": mock_state}
    
    await reconciler.reconcile(states)
    
    # Should have adopted it
    assert mock_state.active_bid is not None
    assert mock_state.active_bid.order_id == "9999"
    assert mock_state.active_bid.price == 50.0
    assert mock_state.active_bid.size == 5.0
