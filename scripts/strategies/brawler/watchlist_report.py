#!/usr/bin/env python3
"""
Generate a rolling Brawler watchlist by replaying recorded HL/CEX microstructure data.

Intended to be scheduled (cron/GH action) so operators always have a fresh ranked list
of assets plus timestamped CSV/JSON artifacts under `logs/brawler_watchlist/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence
import shutil

import pandas as pd

from slipstream.strategies.brawler.config import BrawlerConfig, load_brawler_config
from slipstream.strategies.brawler.tools.candidate_scan import (
    CandidateResult,
    scan_candidates,
)


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise argparse.ArgumentTypeError(f"Invalid timestamp '{value}' (expected ISO-8601 or YYYY-MM-DD).")
    return ts


def _summarize(results: List[CandidateResult], top_k: int) -> str:
    if not results:
        return "No candidates evaluated."

    passing = sum(1 for r in results if r.qualifies)
    lines = [f"{passing}/{len(results)} assets satisfy the configured screening thresholds."]

    sorted_results = sorted(results, key=lambda r: (r.qualifies, r.score), reverse=True)
    header = f"{'Rank':<4} {'Symbol':<8} {'Score':>7} {'Spread':>7} {'Sigma':>7} {'Basis':>7} {'Depth':>7} {'Funding':>8} Notes"
    lines.append(header)
    lines.append("-" * len(header))

    for idx, result in enumerate(sorted_results[: max(1, top_k)]):
        notes = "; ".join(result.notes)
        lines.append(
            f"{idx + 1:<4} {result.symbol:<8} {result.score:7.2f} "
            f"{result.spread_ratio:7.2f} {result.sigma_ratio:7.2f} "
            f"{result.basis_mean_ticks:7.2f} {result.depth_multiple:7.2f} "
            f"{result.funding_std:8.4f} {notes}"
        )
    return "\n".join(lines)


def _write_artifacts(
    results: List[CandidateResult],
    output_dir: Path,
    tag: Optional[str],
) -> dict[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = tag or "watchlist"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = [r.as_dict() for r in results]
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    markdown_path = output_dir / f"{prefix}_{timestamp}.md"

    df = pd.DataFrame(payload)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2))

    markdown_lines = ["| Rank | Symbol | Score | Spread Ratio | Sigma Ratio | Basis (ticks) | Depth Multiple | Funding Std | Notes |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    sorted_results = sorted(results, key=lambda r: (r.qualifies, r.score), reverse=True)
    for idx, res in enumerate(sorted_results, start=1):
        markdown_lines.append(
            f"| {idx} | {res.symbol} | {res.score:.2f} | {res.spread_ratio:.2f} | {res.sigma_ratio:.2f} | "
            f"{res.basis_mean_ticks:.2f} | {res.depth_multiple:.2f} | {res.funding_std:.4f} | {'; '.join(res.notes)} |"
        )
    markdown_path.write_text("\n".join(markdown_lines))

    latest_csv = output_dir / f"{prefix}_latest.csv"
    latest_json = output_dir / f"{prefix}_latest.json"
    latest_md = output_dir / f"{prefix}_latest.md"
    shutil.copyfile(csv_path, latest_csv)
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(markdown_path, latest_md)

    return {
        "csv": csv_path,
        "json": json_path,
        "md": markdown_path,
        "latest_csv": latest_csv,
        "latest_json": latest_json,
        "latest_md": latest_md,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce a rolling Brawler watchlist report.")
    parser.add_argument("--config", help="Path to Brawler YAML/JSON config.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Restrict evaluation to this subset of Hyperliquid symbols (defaults to all config assets).",
    )
    parser.add_argument(
        "--hl-pattern",
        required=True,
        help="Hyperliquid BBO CSV pattern with {symbol}/{lower} tokens.",
    )
    parser.add_argument(
        "--cex-pattern",
        required=True,
        help="Binance (or other CEX) BBO CSV pattern with {cex_symbol}/{cex_lower} tokens.",
    )
    parser.add_argument(
        "--hl-depth-pattern",
        help="Optional Hyperliquid depth CSV pattern (total depth or bid/ask depth columns).",
    )
    parser.add_argument(
        "--funding-pattern",
        help="Optional funding-rate CSV pattern used to penalize volatile carry.",
    )
    parser.add_argument("--start", help="Only process samples after this timestamp (ISO-8601).")
    parser.add_argument("--end", help="Only process samples before this timestamp (ISO-8601).")
    parser.add_argument(
        "--align-ms",
        type=int,
        help="Maximum pairing delta (ms) between HL and CEX quotes (overrides config).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many entries to show in the console summary (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/brawler_watchlist"),
        help="Directory for timestamped CSV/JSON/Markdown artifacts.",
    )
    parser.add_argument(
        "--tag",
        help="Optional prefix for generated artifact filenames (e.g., 'nightly').",
    )
    return parser


def run_watchlist(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_brawler_config(args.config)
    symbols = args.symbols or list(config.assets.keys())
    start = _parse_timestamp(args.start)
    end = _parse_timestamp(args.end)

    results = scan_candidates(
        config,
        symbols,
        hl_pattern=args.hl_pattern,
        cex_pattern=args.cex_pattern,
        hl_depth_pattern=args.hl_depth_pattern,
        funding_pattern=args.funding_pattern,
        start=start,
        end=end,
        align_ms=args.align_ms,
    )

    print(_summarize(results, top_k=args.top_k))
    artifact_paths = _write_artifacts(results, args.output_dir, args.tag)
    artifacts_str = ", ".join(f"{key}={path}" for key, path in artifact_paths.items())
    print(f"\nArtifacts written: {artifacts_str}")


if __name__ == "__main__":  # pragma: no cover
    run_watchlist()
