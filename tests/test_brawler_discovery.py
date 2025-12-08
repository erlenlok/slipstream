import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerDiscoveryConfig
from slipstream.strategies.brawler.discovery import DiscoveryEngine


@pytest.fixture
def mock_config():
    config = BrawlerConfig()
    config.discovery = BrawlerDiscoveryConfig(
        enabled=True,
        interval_seconds=3600,
        min_volume_ratio=0.5,
        max_funding_rate=0.001,
        benchmarks=["BTC", "ETH"]
    )
    config.assets = {"BTC": MagicMock(), "ETH": MagicMock()}
    return config


@pytest.mark.asyncio
async def test_discovery_engine_scan(mock_config):
    engine = DiscoveryEngine(mock_config)

    mock_hl_universe = [
        {"name": "SOL", "szDecimals": 2},
        {"name": "DOGE", "szDecimals": 0},
        {"name": "BTC", "szDecimals": 5},
        {"name": "ETH", "szDecimals": 4},
    ]
    mock_hl_ctxs = [
        {"dayNtlVlm": "1000000", "funding": "0.00001"},
        {"dayNtlVlm": "500000", "funding": "0.002"},
        {"dayNtlVlm": "10000000", "funding": "0.0"},
        {"dayNtlVlm": "5000000", "funding": "0.0"},
    ]
    
    mock_binance_tickers = [
        {"symbol": "SOLUSDT", "quoteVolume": "10000000"},
        {"symbol": "DOGEUSDT", "quoteVolume": "1000000"},
        {"symbol": "BTCUSDT", "quoteVolume": "100000000"},
        {"symbol": "ETHUSDT", "quoteVolume": "50000000"},
    ]

    # Mock response objects
    mock_hl_resp = MagicMock()
    mock_hl_resp.json.return_value = [{"universe": mock_hl_universe}, mock_hl_ctxs]
    mock_hl_resp.raise_for_status = MagicMock()

    mock_bin_resp = MagicMock()
    mock_bin_resp.json.return_value = mock_binance_tickers
    mock_bin_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        # Setup post/get to return the mock responses
        mock_client.post.return_value = mock_hl_resp
        mock_client.get.return_value = mock_bin_resp

        engine._report_results = MagicMock()
        
        await engine.scan()
        
        candidates = engine._report_results.call_args[0][0]
        assert len(candidates) == 0


@pytest.mark.asyncio
async def test_discovery_engine_scan_finds_candidate(mock_config):
    engine = DiscoveryEngine(mock_config)

    mock_hl_universe = [
        {"name": "SOL", "szDecimals": 2},
        {"name": "BTC", "szDecimals": 5},
    ]
    mock_hl_ctxs = [
        {"dayNtlVlm": "100000", "funding": "0.00001"},
        {"dayNtlVlm": "10000000", "funding": "0.0"},
    ]
    
    mock_binance_tickers = [
        {"symbol": "SOLUSDT", "quoteVolume": "10000000"},
        {"symbol": "BTCUSDT", "quoteVolume": "100000000"},
    ]

    mock_hl_resp = MagicMock()
    mock_hl_resp.json.return_value = [{"universe": mock_hl_universe}, mock_hl_ctxs]
    mock_hl_resp.raise_for_status = MagicMock()

    mock_bin_resp = MagicMock()
    mock_bin_resp.json.return_value = mock_binance_tickers
    mock_bin_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        mock_client.post.return_value = mock_hl_resp
        mock_client.get.return_value = mock_bin_resp

        engine._report_results = MagicMock()
        await engine.scan()
        
        candidates = engine._report_results.call_args[0][0]
        
        assert len(candidates) == 1
        assert candidates[0].symbol == "SOL"
        assert candidates[0].relative_ratio == pytest.approx(0.1)
