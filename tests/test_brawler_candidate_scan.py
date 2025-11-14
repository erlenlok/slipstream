import pandas as pd

from slipstream.strategies.brawler.config import (
    BrawlerAssetConfig,
    BrawlerCandidateScreeningConfig,
)
from slipstream.strategies.brawler.tools.candidate_scan import DepthMetrics, evaluate_candidate


def _make_quotes(mid_start: float, drift: float, spread: float, count: int = 200) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=count, freq="1s", tz="UTC")
    mids = mid_start + drift * pd.RangeIndex(count)
    bids = mids - spread / 2.0
    asks = mids + spread / 2.0
    df = pd.DataFrame({"timestamp": timestamps, "bid": bids, "ask": asks})
    df["mid"] = mids
    df["spread"] = spread
    df["spread_bps"] = (spread / df["mid"]) * 10_000
    return df


def test_candidate_scanner_flags_strong_spread_edge() -> None:
    asset = BrawlerAssetConfig(symbol="BTC", cex_symbol="BTCUSDT", tick_size=0.1)
    screening = BrawlerCandidateScreeningConfig(
        min_samples=50,
        max_mean_basis_ticks=3.0,
        max_basis_std_ticks=2.0,
        min_spread_ratio=2.0,
        sigma_ratio_min=0.8,
        sigma_ratio_max=1.2,
        score_min=-10.0,
    )
    hl_quotes = _make_quotes(100.0, 0.001, spread=1.0, count=120)
    cex_quotes = _make_quotes(99.8, 0.001, spread=0.2, count=120)
    result = evaluate_candidate(
        asset,
        screening,
        hl_quotes,
        cex_quotes,
        tolerance_ms=500,
    )
    assert result.samples == 120
    assert result.spread_ratio > 2.0
    assert result.qualifies


def test_candidate_scanner_rejects_large_basis() -> None:
    asset = BrawlerAssetConfig(symbol="ETH", cex_symbol="ETHUSDT", tick_size=0.05)
    screening = BrawlerCandidateScreeningConfig(
        min_samples=50,
        max_mean_basis_ticks=1.0,
        max_basis_std_ticks=1.0,
        min_spread_ratio=1.5,
        sigma_ratio_min=0.5,
        sigma_ratio_max=1.5,
        score_min=-10.0,
    )
    hl_quotes = _make_quotes(2000.0, 0.002, spread=1.0, count=80)
    # CEX mid sits far away to violate basis limit
    cex_quotes = _make_quotes(1995.0, 0.002, spread=0.4, count=80)
    result = evaluate_candidate(
        asset,
        screening,
        hl_quotes,
        cex_quotes,
        tolerance_ms=500,
    )
    assert not result.qualifies
    assert any("basis" in note.lower() for note in result.notes)


def test_candidate_scanner_handles_missing_overlap() -> None:
    asset = BrawlerAssetConfig(symbol="SOL", cex_symbol="SOLUSDT", tick_size=0.01)
    screening = BrawlerCandidateScreeningConfig(
        min_samples=10,
        max_mean_basis_ticks=5.0,
        max_basis_std_ticks=5.0,
        min_spread_ratio=1.0,
        sigma_ratio_min=0.1,
        sigma_ratio_max=5.0,
        score_min=-10.0,
    )
    hl_quotes = _make_quotes(100.0, 0.001, spread=0.5, count=5)
    cex_quotes = _make_quotes(100.0, 0.001, spread=0.5, count=5)
    # shift CEX timestamps by 5 seconds so merge_asof fails with 1ms tolerance
    cex_quotes["timestamp"] = cex_quotes["timestamp"] + pd.Timedelta(seconds=5)
    result = evaluate_candidate(
        asset,
        screening,
        hl_quotes,
        cex_quotes,
        tolerance_ms=1,
    )
    assert result.samples == 0
    assert not result.qualifies
    assert "overlapping" in " ".join(result.notes).lower()


def test_candidate_scanner_enforces_depth_multiple() -> None:
    asset = BrawlerAssetConfig(symbol="DOGE", cex_symbol="DOGEUSDT", tick_size=0.001, order_size=1.0)
    screening = BrawlerCandidateScreeningConfig(
        min_samples=50,
        max_mean_basis_ticks=5.0,
        max_basis_std_ticks=5.0,
        min_spread_ratio=1.5,
        sigma_ratio_min=0.5,
        sigma_ratio_max=1.5,
        min_depth_multiple=3.0,
        score_min=-10.0,
    )
    hl_quotes = _make_quotes(0.2, 0.0001, spread=0.002, count=60)
    cex_quotes = _make_quotes(0.2, 0.0001, spread=0.001, count=60)
    depth_metrics = DepthMetrics(avg_total_depth=2.0, depth_multiple=2.0)
    result = evaluate_candidate(
        asset,
        screening,
        hl_quotes,
        cex_quotes,
        tolerance_ms=500,
        depth_metrics=depth_metrics,
    )
    assert not result.qualifies
    assert any("depth multiple" in note.lower() for note in result.notes)


def test_candidate_scanner_enforces_funding_std() -> None:
    asset = BrawlerAssetConfig(symbol="OP", cex_symbol="OPUSDT", tick_size=0.01)
    screening = BrawlerCandidateScreeningConfig(
        min_samples=50,
        max_mean_basis_ticks=5.0,
        max_basis_std_ticks=5.0,
        min_spread_ratio=1.5,
        sigma_ratio_min=0.5,
        sigma_ratio_max=1.5,
        max_funding_std=0.01,
        score_min=-10.0,
    )
    hl_quotes = _make_quotes(3.0, 0.0005, spread=0.03, count=80)
    cex_quotes = _make_quotes(3.0, 0.0005, spread=0.01, count=80)
    result = evaluate_candidate(
        asset,
        screening,
        hl_quotes,
        cex_quotes,
        tolerance_ms=500,
        funding_std=0.05,
    )
    assert not result.qualifies
    assert any("funding std" in note.lower() for note in result.notes)
