"""
Wall of Shame — Episode History Analyzer.

Reads one or more episode_history_*.csv files produced by the training loop and
ranks events by their mean reward so you can identify data-poisoners.

Usage:
    uv run wall-of-shame                          # auto-discover latest CSV in reports_dir
    uv run wall-of-shame --csv path/to/file.csv   # explicit CSV path
    uv run wall-of-shame --top 20                 # show N worst events (default 10)
    uv run wall-of-shame --threshold -50          # flag events below this reward
    uv run wall-of-shame --quarantine             # write quarantine.txt with flagged IDs
    uv run wall-of-shame --all                    # show all events, not just bottom N
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from config import cfg

log = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "episode", "global_step", "event_id", "episode_reward", "episode_steps",
    "stage", "rotation_tier", "length_mi", "n_timesteps", "max_rotation_score",
]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _find_latest_csv(reports_dir: Path) -> Optional[Path]:
    """Return the most recently modified episode_history_*.csv in reports_dir."""
    pattern = str(reports_dir / "episode_history_*.csv")
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(matches[0]) if matches else None


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate an episode history CSV."""
    df = pd.read_csv(csv_path, dtype={"event_id": str, "rotation_tier": str})
    missing = [c for c in ("event_id", "episode_reward") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["episode_reward"] = pd.to_numeric(df["episode_reward"], errors="coerce")
    df["episode_steps"] = pd.to_numeric(df.get("episode_steps", pd.Series(dtype=float)), errors="coerce")
    log.info("Loaded %d episode rows from %s", len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _build_shame_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode history by event_id and compute shame metrics."""
    agg = df.groupby("event_id").agg(
        appearances=("episode_reward", "count"),
        mean_reward=("episode_reward", "mean"),
        min_reward=("episode_reward", "min"),
        max_reward=("episode_reward", "max"),
        std_reward=("episode_reward", "std"),
        mean_steps=("episode_steps", "mean"),
    ).reset_index()

    # Carry through static metadata columns (use most-common value per event)
    for col in ("rotation_tier", "length_mi", "n_timesteps", "max_rotation_score", "stage"):
        if col in df.columns:
            meta = df.groupby("event_id")[col].agg(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None)
            agg = agg.merge(meta.rename(col), on="event_id", how="left")

    return agg.sort_values("mean_reward")


def _auto_threshold(shame: pd.DataFrame) -> float:
    """Compute a default flagging threshold: mean - 2 * std of per-event mean rewards."""
    return float(shame["mean_reward"].mean() - 2 * shame["mean_reward"].std())


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_report(shame: pd.DataFrame, threshold: float, top: int, show_all: bool) -> None:
    display = shame if show_all else shame.head(top)

    print("\n" + "=" * 90)
    print("  💀  WALL OF SHAME — Worst-Performing Training Events")
    print("=" * 90)
    print(f"\n  Total unique events : {len(shame)}")
    print(f"  Total episodes      : {shame['appearances'].sum():,}")
    print(f"  Flagging threshold  : {threshold:.2f}  (events below this are 'poisoners')")
    flagged = shame[shame["mean_reward"] < threshold]
    print(f"  Flagged events      : {len(flagged)}")

    has_tier = "rotation_tier" in display.columns
    has_len = "length_mi" in display.columns
    has_steps = "n_timesteps" in display.columns

    hdr_event = f"{'Event ID':<42}"
    hdr_fixed = f"{'N':>5}  {'MeanRew':>9}  {'MinRew':>9}  {'MaxRew':>9}  {'StdRew':>7}"
    hdr_meta = (
        (f"  {'Tier':<10}" if has_tier else "")
        + (f"  {'LenMi':>6}" if has_len else "")
        + (f"  {'Steps':>6}" if has_steps else "")
    )

    print(f"\n  {hdr_event} {hdr_fixed}{hdr_meta}")
    print(f"  {'-'*42} {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}"
          + ("  " + "-"*10 if has_tier else "")
          + ("  " + "-"*6 if has_len else "")
          + ("  " + "-"*6 if has_steps else ""))

    for _, row in display.iterrows():
        flag = " 💀" if row["mean_reward"] < threshold else "   "
        line = (
            f"  {str(row['event_id']):<42} "
            f"{int(row['appearances']):>5}  "
            f"{row['mean_reward']:>9.2f}  "
            f"{row['min_reward']:>9.2f}  "
            f"{row['max_reward']:>9.2f}  "
            f"{row['std_reward']:>7.2f}"
            f"{flag}"
        )
        if has_tier:
            tier = str(row.get("rotation_tier") or "?")
            line += f"  {tier:<10}"
        if has_len:
            lmi = row.get("length_mi")
            line += f"  {lmi:>6.1f}" if lmi is not None and not (isinstance(lmi, float) and np.isnan(lmi)) else f"  {'N/A':>6}"
        if has_steps:
            nts = row.get("n_timesteps")
            line += f"  {int(nts):>6}" if nts is not None and not (isinstance(nts, float) and np.isnan(nts)) else f"  {'N/A':>6}"
        print(line)

    print("\n  Tip: Run with --quarantine to write quarantine.txt for these events.")
    print("  Then remove them from index.parquet and re-run `uv run scan-events`.")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--csv", "csv_path",
    default=None,
    help="Path to episode_history CSV. Auto-discovers latest in reports_dir if not set.",
)
@click.option(
    "--top",
    default=10,
    type=int,
    show_default=True,
    help="Number of worst events to display.",
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Flag events with mean reward below this value. Default: mean - 2*std.",
)
@click.option(
    "--quarantine",
    is_flag=True,
    default=False,
    help="Write a quarantine.txt file listing flagged event IDs.",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Show all events ranked, not just --top N.",
)
def main(
    csv_path: Optional[str],
    top: int,
    threshold: Optional[float],
    quarantine: bool,
    show_all: bool,
) -> None:
    """Analyze episode history CSV and surface worst-performing training events."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reports_dir = Path(cfg.data.reports_dir)

    if csv_path:
        path = Path(csv_path)
    else:
        path = _find_latest_csv(reports_dir)
        if path is None:
            click.echo(
                f"No episode_history_*.csv found in {reports_dir}. "
                "Run a training stage first.",
                err=True,
            )
            raise SystemExit(1)
        log.info("Auto-discovered CSV: %s", path)

    df = _load_csv(path)
    if df.empty:
        click.echo("CSV is empty — no episodes recorded yet.", err=True)
        raise SystemExit(1)

    shame = _build_shame_table(df)
    thr = threshold if threshold is not None else _auto_threshold(shame)

    _print_report(shame, thr, top, show_all)

    if quarantine:
        flagged_ids = shame[shame["mean_reward"] < thr]["event_id"].tolist()
        if not flagged_ids:
            click.echo("No events below threshold — quarantine.txt not written.")
        else:
            q_path = reports_dir / "quarantine.txt"
            q_path.write_text("\n".join(flagged_ids) + "\n", encoding="utf-8")
            click.echo(f"Quarantine list → {q_path}  ({len(flagged_ids)} events)")
            click.echo(
                "\nTo remove from index.parquet, run:\n"
                "  uv run python -c \"\n"
                "  import geopandas as gpd\n"
                "  q = open('quarantine.txt').read().splitlines()\n"
                "  idx = gpd.read_parquet('path/to/index.parquet')\n"
                "  idx = idx[~idx['event_id'].isin(q)]\n"
                "  idx.to_parquet('path/to/index.parquet', index=False)\n"
                "  print(f'Removed {len(q)} events. {len(idx)} remain.')\n"
                "  \""
            )


if __name__ == "__main__":
    main()
