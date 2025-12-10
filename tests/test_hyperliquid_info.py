
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidInfoClient

class TestHyperliquidInfoClient:
    @pytest.fixture
    def mock_hl_modules(self):
        with patch("slipstream.strategies.brawler.connectors.hyperliquid._load_hyperliquid_modules") as mock_load:
            mock_info_cls = MagicMock()
            mock_info_instance = MagicMock()
            mock_info_cls.return_value = mock_info_instance
            
            # Info, Exchange, Constants, Account
            mock_load.return_value = (mock_info_cls, MagicMock(), MagicMock(), MagicMock())
            yield mock_info_instance

    @pytest.mark.asyncio
    async def test_get_user_rate_limit(self, mock_hl_modules):
        # Setup
        client = HyperliquidInfoClient()
        mock_info = mock_hl_modules
        
        expected_ret = {
            "cum_requests": 100, 
            "cum_volume": 50000.0,
            "user": "0x123"
        }
        mock_info.user_rate_limit.return_value = expected_ret
        
        # Act
        result = await client.get_user_rate_limit("0x123")
        
        # Assert
        mock_info.user_rate_limit.assert_called_once_with("0x123")
        assert result == expected_ret
