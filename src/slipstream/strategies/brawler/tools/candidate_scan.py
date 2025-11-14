"""Offline candidate discovery for Brawler."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import (
    BrawlerAssetConfig,
    BrawlerCandidateScreeningConfig,
    BrawlerConfig,
    load_brawler_config,
)

TIMESTAMP_COLUMNS = ("timestamp", "ts", "time", "datetime", "date")
BID_COLUMNS = ("bid", "best_bid", "bid_price", "bbo_bid")
ASK_COLUMNS = ("ask", "best_ask", "ask_price", "bbo_ask")
DEPTH_BID_COLUMNS = ("bid_depth", "depth_bid", "liquidity_bid")
DEPTH_ASK_COLUMNS = ("ask_depth", "depth_ask", "liquidity_ask")
DEPTH_TOTAL_COLUMNS = ("total_depth", "depth_total", "book_depth")
FUNDING_COLUMNS = ("funding_rate", "funding", "rate")


@dataclass
class DepthMetrics:
    avg_total_depth: float
    depth_multiple: float


@dataclass
class CandidateResult:
    symbol: str
    cex_symbol: str
    samples: int
    hl_spread_bps: float
    cex_spread_bps: float
    spread_ratio: float
    basis_mean: float
    basis_std: float
    basis_mean_ticks: float
    basis_std_ticks: float
    sigma_hl: float
    sigma_cex: float
    sigma_ratio: float
    depth_multiple: float
    funding_std: float
    score: float
    qualifies: bool
    notes: List[str]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["notes"] = "; ".join(self.notes)
        return payload


def _infer_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        max_abs = float(numeric.abs().max()) if not numeric.empty else 0.0
        # crude heuristic for units
        if max_abs > 1e12:
            unit = "ns"
        elif max_abs > 1e10:
            unit = "ms"
        elif max_abs > 1e9:
            unit = "s"
        elif max_abs > 1e6:
            unit = "s"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().all():
        raise ValueError("Unable to parse timestamps from provided CSV.")
    return parsed


def _select_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing required columns. Looked for any of: {', '.join(candidates)}")


def _prepare_quotes(path: Path, *, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    ts_col = _select_column(df, TIMESTAMP_COLUMNS)
    bid_col = _select_column(df, BID_COLUMNS)
    ask_col = _select_column(df, ASK_COLUMNS)

    timestamps = _infer_timestamp(df[ts_col])
    bids = pd.to_numeric(df[bid_col], errors="coerce")
    asks = pd.to_numeric(df[ask_col], errors="coerce")

    result = pd.DataFrame({"timestamp": timestamps, "bid": bids, "ask": asks})
    result = result.dropna(subset=["timestamp", "bid", "ask"])
    if start is not None:
        result = result[result["timestamp"] >= start]
    if end is not None:
        result = result[result["timestamp"] <= end]

    if result.empty:
        return result

    result = result.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    result["spread"] = result["ask"] - result["bid"]
    result["spread_bps"] = np.where(
        result["mid"] > 0, (result["spread"] / result["mid"]) * 10_000, np.nan
    )
    return result.dropna(subset=["mid", "spread"])


def _prepare_depth(path: Path, *, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    ts_col = _select_column(df, TIMESTAMP_COLUMNS)
    total_col = next((c for c in DEPTH_TOTAL_COLUMNS if c in df.columns), None)
    bid_col = next((c for c in DEPTH_BID_COLUMNS if c in df.columns), None)
    ask_col = next((c for c in DEPTH_ASK_COLUMNS if c in df.columns), None)

    timestamps = _infer_timestamp(df[ts_col])
    result = {"timestamp": timestamps}
    if total_col:
        result["total_depth"] = pd.to_numeric(df[total_col], errors="coerce")
    elif bid_col and ask_col:
        bid_depth = pd.to_numeric(df[bid_col], errors="coerce")
        ask_depth = pd.to_numeric(df[ask_col], errors="coerce")
        result["total_depth"] = bid_depth.fillna(0.0) + ask_depth.fillna(0.0)
    else:
        raise ValueError(
            f"Depth file {path} missing required columns (need total depth or both bid/ask depth)."
        )

    depth_df = pd.DataFrame(result).dropna(subset=["timestamp", "total_depth"])
    if start is not None:
        depth_df = depth_df[depth_df["timestamp"] >= start]
    if end is not None:
        depth_df = depth_df[depth_df["timestamp"] <= end]
    if depth_df.empty:
        raise ValueError(f"Depth file {path} contains no usable rows in the requested window.")
    return depth_df


def _compute_depth_metrics(depth_df: Optional[pd.DataFrame], order_size: float) -> Optional[DepthMetrics]:
    if depth_df is None or depth_df.empty or order_size <= 0:
        if depth_df is not None and order_size <= 0:
            return DepthMetrics(float("nan"), float("nan"))
        return None
    avg_total = float(depth_df["total_depth"].mean())
    depth_multiple = avg_total / order_size if order_size > 0 else float("nan")
    return DepthMetrics(avg_total_depth=avg_total, depth_multiple=depth_multiple)


def _prepare_funding(path: Path, *, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    ts_col = _select_column(df, TIMESTAMP_COLUMNS)
    funding_col = _select_column(df, FUNDING_COLUMNS)

    timestamps = _infer_timestamp(df[ts_col])
    funding_series = pd.to_numeric(df[funding_col], errors="coerce")
    result = pd.DataFrame({"timestamp": timestamps, "funding": funding_series}).dropna(
        subset=["timestamp", "funding"]
    )
    if start is not None:
        result = result[result["timestamp"] >= start]
    if end is not None:
        result = result[result["timestamp"] <= end]
    if result.empty:
        raise ValueError(f"Funding file {path} contains no usable rows in the requested window.")
    return result


def _compute_funding_std(funding_df: Optional[pd.DataFrame]) -> Optional[float]:
    if funding_df is None or funding_df.empty:
        return None
    return float(funding_df["funding"].std())


def _realized_vol(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    log_returns = np.log(series / series.shift(1)).dropna()
    if log_returns.empty:
        return 0.0
    return float(log_returns.std())


def evaluate_candidate(
    asset_cfg: BrawlerAssetConfig,
    screening_cfg: BrawlerCandidateScreeningConfig,
    hl_quotes: pd.DataFrame,
    cex_quotes: pd.DataFrame,
    *,
    tolerance_ms: int,
    depth_metrics: Optional[DepthMetrics] = None,
    funding_std: Optional[float] = None,
) -> CandidateResult:
    notes: List[str] = []
    if hl_quotes.empty:
        return CandidateResult(
            symbol=asset_cfg.symbol,
            cex_symbol=asset_cfg.cex_symbol,
            samples=0,
            hl_spread_bps=float("nan"),
            cex_spread_bps=float("nan"),
            spread_ratio=float("nan"),
            basis_mean=float("nan"),
            basis_std=float("nan"),
            basis_mean_ticks=float("nan"),
            basis_std_ticks=float("nan"),
            sigma_hl=0.0,
            sigma_cex=0.0,
            sigma_ratio=float("nan"),
            depth_multiple=float("nan"),
            funding_std=float("nan"),
            score=float("-inf"),
            qualifies=False,
            notes=["No Hyperliquid data"],
        )
    if cex_quotes.empty:
        return CandidateResult(
            symbol=asset_cfg.symbol,
            cex_symbol=asset_cfg.cex_symbol,
            samples=0,
            hl_spread_bps=float("nan"),
            cex_spread_bps=float("nan"),
            spread_ratio=float("nan"),
            basis_mean=float("nan"),
            basis_std=float("nan"),
            basis_mean_ticks=float("nan"),
            basis_std_ticks=float("nan"),
            sigma_hl=0.0,
            sigma_cex=0.0,
            sigma_ratio=float("nan"),
            depth_multiple=float("nan"),
            funding_std=float("nan"),
            score=float("-inf"),
            qualifies=False,
            notes=["No CEX data"],
        )

    tolerance = pd.Timedelta(milliseconds=max(1, tolerance_ms))
    merged = pd.merge_asof(
        hl_quotes.sort_values("timestamp"),
        cex_quotes.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tolerance,
        suffixes=("_hl", "_cex"),
    ).dropna(subset=["mid_cex", "mid_hl"])

    samples = len(merged)
    if samples == 0:
        notes.append("No overlapping samples within tolerance")

    hl_spread_bps = float(merged["spread_bps_hl"].mean()) if samples else float("nan")
    cex_spread_bps = float(merged["spread_bps_cex"].mean()) if samples else float("nan")
    basis = merged["mid_hl"] - merged["mid_cex"]
    basis_mean = float(basis.mean()) if samples else float("nan")
    basis_std = float(basis.std()) if samples else float("nan")

    tick_size = asset_cfg.tick_size or math.nan
    basis_mean_ticks = basis_mean / tick_size if tick_size and not math.isnan(basis_mean) else math.nan
    basis_std_ticks = basis_std / tick_size if tick_size and not math.isnan(basis_std) else math.nan

    sigma_hl = _realized_vol(merged["mid_hl"])
    sigma_cex = _realized_vol(merged["mid_cex"])
    sigma_ratio = (
        sigma_hl / sigma_cex if sigma_cex > 0 and sigma_hl > 0 else (float("inf") if sigma_cex == 0 else float("nan"))
    )

    spread_ratio = (
        hl_spread_bps / max(cex_spread_bps, asset_cfg.tick_size or 1e-9)
        if hl_spread_bps and cex_spread_bps
        else float("nan")
    )

    basis_penalty = 0.0
    if not math.isnan(basis_mean_ticks) and screening_cfg.max_mean_basis_ticks > 0:
        drift_over = abs(basis_mean_ticks) - screening_cfg.max_mean_basis_ticks
        if drift_over > 0:
            basis_penalty += drift_over / screening_cfg.max_mean_basis_ticks
            notes.append(
                f"Mean basis {basis_mean_ticks:.2f} ticks exceeds {screening_cfg.max_mean_basis_ticks}"
            )
    else:
        notes.append("Basis mean unavailable (missing tick size)" if math.isnan(basis_mean_ticks) else "")

    if not math.isnan(basis_std_ticks) and screening_cfg.max_basis_std_ticks > 0:
        std_over = basis_std_ticks - screening_cfg.max_basis_std_ticks
        if std_over > 0:
            basis_penalty += std_over / screening_cfg.max_basis_std_ticks
            notes.append(
                f"Basis std {basis_std_ticks:.2f} ticks exceeds {screening_cfg.max_basis_std_ticks}"
            )
    elif math.isnan(basis_std_ticks):
        notes.append("Basis std unavailable (missing tick size)")

    if not math.isnan(sigma_ratio):
        if sigma_ratio < screening_cfg.sigma_ratio_min or sigma_ratio > screening_cfg.sigma_ratio_max:
            notes.append(
                f"Sigma ratio {sigma_ratio:.2f} outside [{screening_cfg.sigma_ratio_min}, {screening_cfg.sigma_ratio_max}]"
            )
        vol_penalty = abs(math.log(sigma_ratio)) if sigma_ratio > 0 else 0.0
    else:
        notes.append("Sigma ratio unavailable")
        vol_penalty = 1.0

    depth_multiple = depth_metrics.depth_multiple if depth_metrics else math.nan
    if depth_metrics and math.isnan(depth_metrics.depth_multiple):
        notes.append("Depth metrics missing (order_size <= 0 or total depth unavailable)")

    funding_std_val = funding_std if funding_std is not None else math.nan
    if funding_std is not None and math.isnan(funding_std_val):
        notes.append("Funding data invalid (non-numeric)")

    if math.isnan(spread_ratio):
        notes.append("Spread ratio unavailable")
        spread_edge = 0.0
    else:
        spread_edge = spread_ratio
        if spread_ratio < screening_cfg.min_spread_ratio:
            notes.append(
                f"Spread ratio {spread_ratio:.2f} below target {screening_cfg.min_spread_ratio}"
            )

    depth_penalty = 0.0
    if not math.isnan(depth_multiple) and screening_cfg.min_depth_multiple > 0:
        if depth_multiple < screening_cfg.min_depth_multiple:
            depth_penalty = (screening_cfg.min_depth_multiple - depth_multiple) / screening_cfg.min_depth_multiple
            notes.append(
                f"Depth multiple {depth_multiple:.2f} below target {screening_cfg.min_depth_multiple}"
            )

    funding_penalty = 0.0
    if not math.isnan(funding_std_val) and screening_cfg.max_funding_std > 0:
        if funding_std_val > screening_cfg.max_funding_std:
            funding_penalty = (funding_std_val - screening_cfg.max_funding_std) / screening_cfg.max_funding_std
            notes.append(
                f"Funding std {funding_std_val:.5f} exceeds {screening_cfg.max_funding_std}"
            )

    score = (
        screening_cfg.weight_spread_edge * spread_edge
        - screening_cfg.weight_basis_penalty * basis_penalty
        - screening_cfg.weight_vol_penalty * vol_penalty
        - screening_cfg.weight_depth_penalty * depth_penalty
        - screening_cfg.weight_funding_penalty * funding_penalty
    )

    qualifies = (
        samples >= screening_cfg.min_samples
        and not math.isnan(spread_ratio)
        and spread_ratio >= screening_cfg.min_spread_ratio
        and not math.isnan(basis_mean_ticks)
        and abs(basis_mean_ticks) <= screening_cfg.max_mean_basis_ticks
        and not math.isnan(basis_std_ticks)
        and basis_std_ticks <= screening_cfg.max_basis_std_ticks
        and not math.isnan(sigma_ratio)
        and screening_cfg.sigma_ratio_min <= sigma_ratio <= screening_cfg.sigma_ratio_max
        and (math.isnan(depth_multiple) or depth_multiple >= screening_cfg.min_depth_multiple)
        and (math.isnan(funding_std_val) or funding_std_val <= screening_cfg.max_funding_std)
        and score >= screening_cfg.score_min
    )
    if samples < screening_cfg.min_samples:
        notes.append(f"Only {samples} overlapping samples (min {screening_cfg.min_samples})")

    return CandidateResult(
        symbol=asset_cfg.symbol,
        cex_symbol=asset_cfg.cex_symbol,
        samples=samples,
        hl_spread_bps=hl_spread_bps,
        cex_spread_bps=cex_spread_bps,
        spread_ratio=spread_ratio,
        basis_mean=basis_mean,
        basis_std=basis_std,
        basis_mean_ticks=basis_mean_ticks,
        basis_std_ticks=basis_std_ticks,
        sigma_hl=sigma_hl,
        sigma_cex=sigma_cex,
        sigma_ratio=sigma_ratio,
        depth_multiple=depth_multiple,
        funding_std=funding_std_val,
        score=score,
        qualifies=qualifies,
        notes=[note for note in notes if note],
    )


def _format_pattern(pattern: str, asset_cfg: BrawlerAssetConfig) -> Path:
    template = pattern.format(
        symbol=asset_cfg.symbol,
        lower=asset_cfg.symbol.lower(),
        cex_symbol=asset_cfg.cex_symbol,
        cex_lower=asset_cfg.cex_symbol.lower(),
    )
    return Path(template)


def _render_table(results: List[CandidateResult]) -> str:
    if not results:
        return "No candidates evaluated."
    df = pd.DataFrame([r.as_dict() for r in results])
    cols = [
        "symbol",
        "cex_symbol",
        "samples",
        "spread_ratio",
        "basis_mean_ticks",
        "basis_std_ticks",
        "sigma_ratio",
        "depth_multiple",
        "funding_std",
        "score",
        "qualifies",
        "notes",
    ]
    df_subset = df[cols]
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        return df_subset.to_string(index=False, float_format=lambda x: f"{x:0.3f}")


def scan_candidates(
    config: BrawlerConfig,
    symbols: Iterable[str],
    hl_pattern: str,
    cex_pattern: str,
    *,
    hl_depth_pattern: Optional[str] = None,
    funding_pattern: Optional[str] = None,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    align_ms: Optional[int],
) -> List[CandidateResult]:
    results: List[CandidateResult] = []
    tolerance_ms = align_ms or config.candidate_screening.align_tolerance_ms
    for symbol in symbols:
        asset_cfg = config.assets.get(symbol)
        if not asset_cfg:
            raise ValueError(f"Symbol '{symbol}' not found in config assets.")
        hl_path = _format_pattern(hl_pattern, asset_cfg)
        cex_path = _format_pattern(cex_pattern, asset_cfg)
        try:
            hl_quotes = _prepare_quotes(hl_path, start=start, end=end)
        except FileNotFoundError:
            results.append(
                CandidateResult(
                    symbol=asset_cfg.symbol,
                    cex_symbol=asset_cfg.cex_symbol,
                    samples=0,
                    hl_spread_bps=float("nan"),
                    cex_spread_bps=float("nan"),
                    spread_ratio=float("nan"),
                    basis_mean=float("nan"),
                    basis_std=float("nan"),
                    basis_mean_ticks=float("nan"),
                    basis_std_ticks=float("nan"),
                    sigma_hl=0.0,
                    sigma_cex=0.0,
            sigma_ratio=float("nan"),
            depth_multiple=float("nan"),
            funding_std=float("nan"),
            score=float("-inf"),
            qualifies=False,
            notes=[f"Missing Hyperliquid file: {hl_path}"],
                )
            )
            continue
        except ValueError as exc:
            results.append(
                CandidateResult(
                    symbol=asset_cfg.symbol,
                    cex_symbol=asset_cfg.cex_symbol,
                    samples=0,
                    hl_spread_bps=float("nan"),
                    cex_spread_bps=float("nan"),
                    spread_ratio=float("nan"),
                    basis_mean=float("nan"),
                    basis_std=float("nan"),
                    basis_mean_ticks=float("nan"),
                    basis_std_ticks=float("nan"),
                    sigma_hl=0.0,
                    sigma_cex=0.0,
                    sigma_ratio=float("nan"),
                    depth_multiple=float("nan"),
                    funding_std=float("nan"),
                    score=float("-inf"),
                    qualifies=False,
                    notes=[f"Hyperliquid file error ({hl_path}): {exc}"],
                )
            )
            continue

        try:
            cex_quotes = _prepare_quotes(cex_path, start=start, end=end)
        except FileNotFoundError:
            results.append(
                CandidateResult(
                    symbol=asset_cfg.symbol,
                    cex_symbol=asset_cfg.cex_symbol,
                    samples=0,
                    hl_spread_bps=float("nan"),
                    cex_spread_bps=float("nan"),
                    spread_ratio=float("nan"),
                    basis_mean=float("nan"),
                    basis_std=float("nan"),
                    basis_mean_ticks=float("nan"),
                    basis_std_ticks=float("nan"),
                    sigma_hl=0.0,
                    sigma_cex=0.0,
                    sigma_ratio=float("nan"),
                    depth_multiple=float("nan"),
                    funding_std=float("nan"),
                    score=float("-inf"),
                    qualifies=False,
                    notes=[f"Missing CEX file: {cex_path}"],
                )
            )
            continue
        except ValueError as exc:
            results.append(
                CandidateResult(
                    symbol=asset_cfg.symbol,
                    cex_symbol=asset_cfg.cex_symbol,
                    samples=0,
                    hl_spread_bps=float("nan"),
                    cex_spread_bps=float("nan"),
                    spread_ratio=float("nan"),
                    basis_mean=float("nan"),
                    basis_std=float("nan"),
                    basis_mean_ticks=float("nan"),
                    basis_std_ticks=float("nan"),
                    sigma_hl=0.0,
                    sigma_cex=0.0,
                    sigma_ratio=float("nan"),
                    depth_multiple=float("nan"),
                    funding_std=float("nan"),
                    score=float("-inf"),
                    qualifies=False,
                    notes=[f"CEX file error ({cex_path}): {exc}"],
                )
            )
            continue

        depth_metrics = None
        extra_notes: List[str] = []
        if hl_depth_pattern:
            depth_path = _format_pattern(hl_depth_pattern, asset_cfg)
            try:
                depth_df = _prepare_depth(depth_path, start=start, end=end)
                depth_metrics = _compute_depth_metrics(depth_df, asset_cfg.order_size)
            except FileNotFoundError:
                extra_notes.append(f"Missing depth file: {depth_path}")
            except ValueError as exc:
                extra_notes.append(f"Depth file error ({depth_path}): {exc}")

        funding_std = None
        if funding_pattern:
            funding_path = _format_pattern(funding_pattern, asset_cfg)
            try:
                funding_df = _prepare_funding(funding_path, start=start, end=end)
                funding_std = _compute_funding_std(funding_df)
            except FileNotFoundError:
                extra_notes.append(f"Missing funding file: {funding_path}")
            except ValueError as exc:
                extra_notes.append(f"Funding file error ({funding_path}): {exc}")

        result = evaluate_candidate(
            asset_cfg,
            config.candidate_screening,
            hl_quotes,
            cex_quotes,
            tolerance_ms=tolerance_ms,
            depth_metrics=depth_metrics,
            funding_std=funding_std,
        )
        result.notes.extend(extra_notes)
        results.append(result)
    return results


def _parse_timestamp_arg(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp '{value}'. Expected ISO-8601 or YYYY-MM-DD format.")
    return ts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank Hyperliquid perps for Brawler by comparing HL vs Binance microstructure."
    )
    parser.add_argument("--config", help="Brawler YAML/JSON config (defaults to single-asset template).")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Subset of Hyperliquid symbols to evaluate (defaults to every asset in config).",
    )
    parser.add_argument(
        "--hl-pattern",
        required=True,
        help="Path pattern for Hyperliquid BBO CSVs. Use {symbol} / {lower}. Example: data/bbo/hl_{symbol}.csv",
    )
    parser.add_argument(
        "--cex-pattern",
        required=True,
        help="Path pattern for Binance (or other CEX) BBO CSVs. Supports {cex_symbol} / {cex_lower}.",
    )
    parser.add_argument(
        "--hl-depth-pattern",
        help="Optional path pattern for Hyperliquid depth metrics (total depth CSV).",
    )
    parser.add_argument(
        "--funding-pattern",
        help="Optional path pattern for funding-rate CSVs (used to cap funding volatility).",
    )
    parser.add_argument(
        "--start",
        help="Filter data starting at this timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--end",
        help="Filter data ending at this timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--align-ms",
        type=int,
        help="Maximum time delta (ms) when pairing HL + CEX quotes (overrides config.candidate_screening.align_tolerance_ms).",
    )
    parser.add_argument(
        "--output",
        choices=("table", "json"),
        default="table",
        help="Render results as a human-readable table or JSON.",
    )
    parser.add_argument(
        "--save-csv",
        help="Optional path to write the raw results table as CSV.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        start = _parse_timestamp_arg(args.start)
        end = _parse_timestamp_arg(args.end)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    config = load_brawler_config(args.config)
    symbols = args.symbols or list(config.assets.keys())
    results = scan_candidates(
        config,
        symbols,
        args.hl_pattern,
        args.cex_pattern,
        start=start,
        end=end,
        align_ms=args.align_ms,
        hl_depth_pattern=args.hl_depth_pattern,
        funding_pattern=args.funding_pattern,
    )

    if args.output == "json":
        print(json.dumps([r.as_dict() for r in results], indent=2, default=str))
    else:
        print(_render_table(results))

    if args.save_csv:
        df = pd.DataFrame([r.as_dict() for r in results])
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
