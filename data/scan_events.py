"""
Event Quality Scanner — builds and updates the GeoParquet event index.

This is the sole owner of index.parquet. It must be run after `uv run zarr-build`
to create or update the index.

For each completed zarr event (detected via metadata.json sidecars) it:
  1. Reads bbox_wgs84 / start_time / end_time / n_timesteps from metadata.json
  2. Joins with dat_tracks.parquet to enrich with length_mi, ef_rating, width_yd, wfo
  3. Builds a shapely Box polygon from the MRMS bounding box as the geometry
  4. Computes a Signal-to-Noise Ratio (SNR) score for each event based on the
     peak RotationTrack60min value and classifies it into curriculum tiers:

       Tier 1 "Monster"  : max_rotation_score > monster_threshold
                           AND n_timesteps >= min_monster_steps
                           AND length_mi >= min_monster_length_mi
       Tier 2 "Moderate" : weak_threshold < max_rotation_score <= monster_threshold
                           (or Monster rotation but fails steps/length criteria)
       Tier 3 "Weak"     : max_rotation_score <= weak_threshold

  5. Writes the result as a GeoParquet file (EPSG:4326) so downstream tools
     can perform spatial queries (e.g. filter by region, check overlap).

The scores and tiers are written to index.parquet so that training scripts can
filter events by tier via the --tier flag.

Usage:
    uv run scan-events                                   # scan all events, update index
    uv run scan-events --report-only                     # print report without writing
    uv run scan-events --monster-threshold 0.020         # custom tier boundary
    uv run scan-events --weak-threshold 0.003
    uv run scan-events --events-dir "D:\\my-data\\events"
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box as shapely_box
from tqdm import tqdm

from config import cfg

log = logging.getLogger(__name__)

# Column names added/updated by the zarr quality scan
_SCORE_COLS = [
    "max_rotation_score",
    "mean_rotation_core",
    "active_pixel_count",
    "n_timesteps",
    "data_completeness",
    "rotation_tier",
    "curriculum_stage",
]

# Warn if an event has unusually many timesteps
_TIMESTEP_OUTLIER = 500

_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Train / val / test split by year (no temporal leakage)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Train / val / test split assignment
# ---------------------------------------------------------------------------

def _assign_splits_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test splits to every row of the event index.

    Primary strategy: year-based (no temporal leakage — older years → train).

    Fallback (when year-based produces empty val or test buckets, e.g. a
    single-year dataset): sort events by descending quality —
        1. length_mi  (longer tracks first)
        2. n_timesteps (more steps second)
        3. max_rotation_score (strongest rotation third)
    — then assign by the configured ratios so the highest-quality events
    are placed in the training split.
    """
    ratios = cfg.training.train_val_test_split  # [0.70, 0.15, 0.15]
    n = len(df)
    if n == 0:
        return df

    # --- Year-based assignment ---
    valid_years = sorted(set(int(y) for y in df["year"].dropna()))
    nu = len(valid_years)
    train_end = max(1, int(nu * ratios[0]))
    val_end = train_end + int(nu * ratios[1])

    def _year_to_split(year) -> str:
        if year is None or (isinstance(year, float) and np.isnan(year)):
            return "train"
        idx = valid_years.index(int(year)) if int(year) in valid_years else 0
        if idx < train_end:
            return "train"
        if idx < val_end:
            return "val"
        return "test"

    year_splits = df["year"].apply(_year_to_split)

    # Check whether year-based produces at least one event in val AND test
    needs_fallback = (year_splits == "val").sum() == 0 or (year_splits == "test").sum() == 0

    if not needs_fallback:
        df = df.copy()
        df["split"] = year_splits
        log.info(
            "Split assignment (year-based): train=%d  val=%d  test=%d",
            (year_splits == "train").sum(),
            (year_splits == "val").sum(),
            (year_splits == "test").sum(),
        )
        return df

    # --- Quality-sorted fallback ---
    log.info(
        "Year-based split produces empty val/test buckets (%d unique year(s)). "
        "Falling back to quality-sorted split: length_mi → n_timesteps → max_rotation_score.",
        nu,
    )

    # Sort best-quality events first so they land in "train"
    sort_cols = {
        "length_mi": df["length_mi"].fillna(0) if "length_mi" in df.columns else pd.Series(0, index=df.index),
        "n_timesteps": df["n_timesteps"].fillna(0) if "n_timesteps" in df.columns else pd.Series(0, index=df.index),
        "max_rotation_score": df["max_rotation_score"].fillna(0) if "max_rotation_score" in df.columns else pd.Series(0, index=df.index),
    }
    sort_df = pd.DataFrame(sort_cols)
    sorted_positions = sort_df.sort_values(
        ["length_mi", "n_timesteps", "max_rotation_score"], ascending=False
    ).index  # original DataFrame index labels, best-quality first

    # Determine bucket sizes (minimum 1 for train; val/test only if enough events)
    train_count = max(1, round(n * ratios[0]))
    val_count = max(1, round(n * ratios[1])) if n >= 3 else 0
    # test gets the remainder

    splits = pd.Series("test", index=df.index)
    for rank, orig_idx in enumerate(sorted_positions):
        if rank < train_count:
            splits[orig_idx] = "train"
        elif rank < train_count + val_count:
            splits[orig_idx] = "val"
        # else stays "test"

    df = df.copy()
    df["split"] = splits
    log.info(
        "Split assignment (quality-sorted fallback): train=%d  val=%d  test=%d",
        (splits == "train").sum(),
        (splits == "val").sum(),
        (splits == "test").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Build base event rows from metadata.json sidecars + DAT tracks join
# ---------------------------------------------------------------------------

def _build_event_rows(
    events_root: Path,
    dat_tracks_path: Path,
) -> gpd.GeoDataFrame:
    """
    Discover completed events and build a base GeoDataFrame.

    Each event directory must contain a metadata.json with at minimum:
        event_id, start_time, end_time, bbox_wgs84

    Columns are enriched via a left-join with dat_tracks.parquet on event_id
    to add: length_mi, ef_rating, width_yd, wfo.
    """
    # Load DAT tracks for property enrichment
    tracks_lookup: dict[str, dict] = {}
    if dat_tracks_path.exists():
        try:
            tracks_gdf = gpd.read_parquet(dat_tracks_path)
            for _, tr in tracks_gdf.iterrows():
                eid = str(tr.get("event_id", ""))
                if eid:
                    tracks_lookup[eid] = {
                        "ef_rating": tr.get("ef_rating"),
                        "length_mi": tr.get("length_mi"),
                        "width_yd": tr.get("width_yd"),
                        "wfo": tr.get("wfo"),
                    }
            log.info("Loaded %d DAT track records for enrichment", len(tracks_lookup))
        except Exception as exc:
            log.warning("Could not load dat_tracks.parquet (%s) — ef_rating/length_mi will be NaN", exc)
    else:
        log.warning("dat_tracks.parquet not found at %s — ef_rating/length_mi will be NaN", dat_tracks_path)

    rows = []
    for event_dir in sorted(events_root.iterdir()):
        if not event_dir.is_dir():
            continue
        meta_path = event_dir / "metadata.json"
        zarr_path = event_dir / "data.zarr"
        if not meta_path.exists() or not zarr_path.exists():
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as exc:
            log.warning("Could not read %s: %s", meta_path, exc)
            continue

        event_id = str(meta.get("event_id", event_dir.name))
        bbox = meta.get("bbox_wgs84")
        if not bbox or len(bbox) != 4:
            log.warning("Event %s missing valid bbox_wgs84 — skipping", event_id)
            continue

        start_time = pd.to_datetime(meta.get("start_time"), utc=True)
        end_time = pd.to_datetime(meta.get("end_time"), utc=True)
        n_timesteps = meta.get("n_timesteps", 0)

        track = tracks_lookup.get(event_id, {})

        rows.append({
            "event_id": event_id,
            "zarr_path": str(zarr_path),
            "start_time": start_time,
            "end_time": end_time,
            "year": start_time.year if start_time is not pd.NaT else None,
            "n_timesteps": n_timesteps,
            "ef_rating": track.get("ef_rating"),
            "length_mi": track.get("length_mi"),
            "width_yd": track.get("width_yd"),
            "wfo": track.get("wfo"),
            "geometry": shapely_box(*bbox),  # (minx, miny, maxx, maxy) → Box polygon
        })

    if not rows:
        log.warning("No completed events found in %s", events_root)
        return gpd.GeoDataFrame(columns=[
            "event_id", "zarr_path", "start_time", "end_time", "year",
            "n_timesteps", "ef_rating", "length_mi", "width_yd", "wfo", "geometry",
        ], geometry="geometry").set_crs(_CRS)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=_CRS)
    log.info("Discovered %d completed events in %s", len(gdf), events_root)
    return gdf


# ---------------------------------------------------------------------------
# Per-event metric computation
# ---------------------------------------------------------------------------

def _scan_event(
    zarr_path: Path,
    monster_threshold: float,
    weak_threshold: float,
    min_monster_steps: int,
    min_monster_length_mi: float,
    length_mi: Optional[float],
) -> dict:
    """
    Open one zarr store and compute quality metrics from RotationTrack60min.

    Monster (Tier 1) requires all three conditions:
      - max_rotation_score > monster_threshold
      - n_timesteps >= min_monster_steps
      - length_mi >= min_monster_length_mi (from DAT tracks, passed in)

    Returns a dict with all _SCORE_COLS values plus basic sanity flags.
    """
    result: dict = {
        "max_rotation_score": np.nan,
        "mean_rotation_core": np.nan,
        "active_pixel_count": 0,
        "n_timesteps": 0,
        "data_completeness": 0.0,
        "rotation_tier": "unknown",
        "curriculum_stage": 3,
        "scan_error": None,
        "timestep_outlier": False,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_zarr(zarr_path, consolidated=True)

        # Data completeness: fraction of expected variables present
        expected_vars = set(cfg.mrms.variables)
        present_vars = set(ds.data_vars)
        result["data_completeness"] = len(present_vars & expected_vars) / max(len(expected_vars), 1)

        # Timestep count
        t_coord = ds.coords.get("time", ds.coords.get("t", None))
        if t_coord is not None:
            result["n_timesteps"] = int(len(t_coord))
        elif ds.data_vars:
            first_var = next(iter(ds.data_vars))
            dims = ds[first_var].dims
            t_dim = next((d for d in dims if d in ("time", "t")), dims[0] if dims else None)
            if t_dim is not None:
                result["n_timesteps"] = int(ds[first_var].sizes[t_dim])

        if result["n_timesteps"] > _TIMESTEP_OUTLIER:
            result["timestep_outlier"] = True

        # RotationTrack60min SNR metrics
        rot_var = "RotationTrack60min"
        if rot_var in ds.data_vars:
            rot = ds[rot_var].values  # (T, H, W) or (H, W)
            rot = np.nan_to_num(rot, nan=0.0)

            max_score = float(np.max(rot))
            result["max_rotation_score"] = round(max_score, 6)

            # Mean of pixels that exceed the weak threshold (the "core" signal)
            core_mask = rot > weak_threshold
            result["active_pixel_count"] = int(np.sum(core_mask))
            if result["active_pixel_count"] > 0:
                result["mean_rotation_core"] = round(float(np.mean(rot[core_mask])), 6)
            else:
                result["mean_rotation_core"] = 0.0

            # Tier classification — all three conditions required for Monster
            length_ok = (
                length_mi is not None
                and not (isinstance(length_mi, float) and np.isnan(length_mi))
                and length_mi >= min_monster_length_mi
            )
            if (
                max_score > monster_threshold
                and result["n_timesteps"] >= min_monster_steps
                and length_ok
            ):
                result["rotation_tier"] = "monster"
                result["curriculum_stage"] = 1
            elif max_score > weak_threshold:
                result["rotation_tier"] = "moderate"
                result["curriculum_stage"] = 2
            else:
                result["rotation_tier"] = "weak"
                result["curriculum_stage"] = 3
        else:
            result["rotation_tier"] = "no_data"
            result["scan_error"] = f"{rot_var} not found in zarr"

    except Exception as exc:
        result["scan_error"] = str(exc)
        log.debug("Error scanning %s: %s", zarr_path, exc)

    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_report(df: pd.DataFrame) -> None:
    """Print a ranked human-readable quality report to stdout."""
    scored = df[df["max_rotation_score"].notna()].copy()
    scored = scored.sort_values("max_rotation_score", ascending=False)

    # Tier counts
    tier_counts = scored["rotation_tier"].value_counts()
    monster_n = tier_counts.get("monster", 0)
    moderate_n = tier_counts.get("moderate", 0)
    weak_n = tier_counts.get("weak", 0)
    no_data_n = tier_counts.get("no_data", 0)
    unscored_n = len(df) - len(scored)

    print("\n" + "=" * 72)
    print("  🌪  TORNADO EVENT QUALITY SCAN — Kankakee Curriculum Report")
    print("=" * 72)
    print(f"\n  Total events  : {len(df)}")
    print(f"  Scanned       : {len(scored)}")
    print(f"  Tier 1 Monster  (>{_fmt(scored['max_rotation_score'].max() if monster_n else 0)} s⁻¹) : {monster_n}")
    print(f"  Tier 2 Moderate                                    : {moderate_n}")
    print(f"  Tier 3 Weak                                        : {weak_n}")
    if no_data_n:
        print(f"  No rotation data                                   : {no_data_n}")
    if unscored_n:
        print(f"  Scan errors / skipped                              : {unscored_n}")

    print(f"\n  {'Rank':<5} {'Event ID':<42} {'Tier':<10} {'MaxRot':>8} {'MeanCore':>10} {'Steps':>7} {'LenMi':>7} {'Split':<6} {'Flags'}")
    print(f"  {'-'*5} {'-'*42} {'-'*10} {'-'*8} {'-'*10} {'-'*7} {'-'*7} {'-'*6} {'-'*10}")

    for rank, (_, row) in enumerate(scored.iterrows(), 1):
        tier = row.get("rotation_tier", "?")
        tier_icon = {"monster": "🔴", "moderate": "🟡", "weak": "⚪", "no_data": "❌"}.get(tier, "?")
        flags = []
        if row.get("timestep_outlier"):
            flags.append("⚠ steps")
        if row.get("data_completeness", 1.0) < 1.0:
            flags.append(f"⚠ {row['data_completeness']:.0%} vars")

        lmi = row.get("length_mi")
        lmi_str = f"{lmi:.1f}" if lmi is not None and not (isinstance(lmi, float) and np.isnan(lmi)) else "N/A"

        print(
            f"  {rank:<5} {str(row.get('event_id', '?')):<42} "
            f"{tier_icon} {tier:<8} {_fmt(row.get('max_rotation_score', float('nan'))):>8} "
            f"{_fmt(row.get('mean_rotation_core', float('nan'))):>10} "
            f"{int(row.get('n_timesteps', 0)):>7} "
            f"{lmi_str:>7} "
            f"{str(row.get('split', '?')):<6} "
            f"{'  '.join(flags)}"
        )

    print("\n  Curriculum recommendation:")
    print(f"  • Stage 1 (Follower)  → train on {monster_n} Monster events (Tier 1) first")
    print(f"  • Stage 2 (Hunter)    → expand to Monster + Moderate ({monster_n + moderate_n} events)")
    print(f"  • Stage 3 (Surveyor)  → use all {len(scored)} events")
    print("\n  Run `uv run train-stage1 --tier 1` to enforce the Monster-only filter.")
    print("=" * 72 + "\n")


def _fmt(v: float) -> str:
    """Format a float for report display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.5f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--events-dir",
    default=None,
    help="Override path to events directory (default from config).",
)
@click.option(
    "--index-path",
    default=None,
    help="Override path to index.parquet (default from config).",
)
@click.option(
    "--monster-threshold",
    default=None,
    type=float,
    help="max_rotation_score (s⁻¹) above which an event is 'Monster' (default from config).",
)
@click.option(
    "--weak-threshold",
    default=None,
    type=float,
    help="max_rotation_score (s⁻¹) below which an event is 'Weak' (default from config).",
)
@click.option(
    "--min-monster-steps",
    default=None,
    type=int,
    help="Minimum n_timesteps required for Tier 1 'Monster' (default from config).",
)
@click.option(
    "--min-monster-length-mi",
    default=None,
    type=float,
    help="Minimum DAT track length in miles required for Tier 1 'Monster' (default from config).",
)
@click.option(
    "--report-only",
    is_flag=True,
    default=False,
    help="Print the quality report without updating index.parquet.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-scan events that already have a max_rotation_score in the index.",
)
def main(
    events_dir: Optional[str],
    index_path: Optional[str],
    monster_threshold: Optional[float],
    weak_threshold: Optional[float],
    min_monster_steps: Optional[int],
    min_monster_length_mi: Optional[float],
    report_only: bool,
    force: bool,
) -> None:
    """Build/update the GeoParquet event index and score events for the Kankakee Curriculum."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    events_root = Path(events_dir or cfg.data.events_dir)
    idx_path = Path(index_path or cfg.data.index_path)
    dat_tracks_path = Path(cfg.data.dat_dir) / "dat_tracks.parquet"
    monster_thr = monster_threshold if monster_threshold is not None else cfg.curriculum.monster_threshold
    weak_thr = weak_threshold if weak_threshold is not None else cfg.curriculum.weak_threshold
    min_monster_steps_val = min_monster_steps if min_monster_steps is not None else cfg.curriculum.min_monster_steps
    min_monster_length_mi_val = min_monster_length_mi if min_monster_length_mi is not None else cfg.curriculum.min_monster_length_mi

    log.info("Scanning events in: %s", events_root)
    log.info(
        "Monster threshold : %.4f s⁻¹  |  Weak threshold: %.4f s⁻¹  |  "
        "Min monster steps: %d  |  Min monster length: %.1f mi",
        monster_thr, weak_thr, min_monster_steps_val, min_monster_length_mi_val,
    )

    # Build base GeoDataFrame from metadata.json sidecars + DAT tracks join
    base_gdf = _build_event_rows(events_root, dat_tracks_path)

    if base_gdf.empty:
        log.error("No completed events found in %s — run `uv run zarr-build` first.", events_root)
        raise SystemExit(1)

    # Load existing scores from prior index (if present) so we can skip re-scanning
    existing_scores: dict[str, dict] = {}
    if idx_path.exists() and not force:
        try:
            prev = gpd.read_parquet(idx_path)
            score_cols_present = [c for c in _SCORE_COLS + ["timestep_outlier", "scan_error"] if c in prev.columns]
            if "max_rotation_score" in prev.columns:
                scored_rows = prev[prev["max_rotation_score"].notna()]
                for _, row in scored_rows.iterrows():
                    eid = str(row["event_id"])
                    existing_scores[eid] = {c: row[c] for c in score_cols_present if c in row.index}
                log.info(
                    "Loaded %d existing scores from %s (use --force to re-scan all)",
                    len(existing_scores), idx_path,
                )
        except Exception as exc:
            log.warning("Could not load existing index for score preservation: %s", exc)

    # Determine which events need a fresh zarr scan
    to_scan = [
        eid for eid in base_gdf["event_id"].tolist()
        if eid not in existing_scores
    ]
    log.info("Events to scan: %d  |  Already scored: %d", len(to_scan), len(existing_scores))

    # Build event_id → zarr_path and event_id → length_mi lookups from base_gdf
    event_to_zarr: dict[str, Path] = {
        str(r["event_id"]): Path(r["zarr_path"])
        for _, r in base_gdf.iterrows()
    }
    event_to_length_mi: dict[str, Optional[float]] = {
        str(r["event_id"]): r.get("length_mi")
        for _, r in base_gdf.iterrows()
    }

    # Scan new events
    scan_results: dict[str, dict] = dict(existing_scores)
    errors = 0
    with tqdm(to_scan, desc="Scanning", unit="event") as bar:
        for event_id in bar:
            bar.set_postfix(event=event_id[:30])
            result = _scan_event(
                event_to_zarr[event_id],
                monster_thr,
                weak_thr,
                min_monster_steps_val,
                min_monster_length_mi_val,
                event_to_length_mi.get(event_id),
            )
            scan_results[event_id] = result
            if result["scan_error"]:
                errors += 1
                log.warning("Scan error for %s: %s", event_id, result["scan_error"])

    if errors:
        log.warning("%d event(s) had scan errors (check zarr integrity)", errors)

    # Merge scan results into the base GeoDataFrame
    scan_df = pd.DataFrame.from_dict(scan_results, orient="index")
    scan_df.index.name = "event_id"
    scan_df = scan_df.reset_index()

    index = base_gdf.set_index("event_id")
    scan_df = scan_df.set_index("event_id")

    # Use update() so only rows present in scan_df are overwritten (NaN-safe)
    for col in _SCORE_COLS + ["timestep_outlier"]:
        if col in scan_df.columns:
            if col not in index.columns:
                index[col] = np.nan
            index[col].update(scan_df[col])

    index = index.reset_index()

    # Always re-classify tiers so cached events pick up threshold/criteria changes
    if "max_rotation_score" in index.columns:
        def _tier(row: pd.Series) -> tuple[str, int]:
            score = row.get("max_rotation_score")
            if pd.isna(score):
                return ("unknown", 3)
            steps = int(row.get("n_timesteps", 0) or 0)
            lmi = row.get("length_mi")
            length_ok = (
                lmi is not None
                and not (isinstance(lmi, float) and np.isnan(lmi))
                and lmi >= min_monster_length_mi_val
            )
            if score > monster_thr and steps >= min_monster_steps_val and length_ok:
                return ("monster", 1)
            if score > weak_thr:
                return ("moderate", 2)
            return ("weak", 3)

        tiers = index.apply(_tier, axis=1)
        index["rotation_tier"] = tiers.apply(lambda t: t[0])
        index["curriculum_stage"] = tiers.apply(lambda t: t[1])

    # Always re-assign splits post-merge so max_rotation_score is available for
    # the quality-sorted fallback and cached events get fresh split assignments
    index = _assign_splits_df(index)

    # Print report
    _print_report(index)

    # Summary stats
    if "rotation_tier" in index.columns:
        tier_summary = index["rotation_tier"].value_counts()
        log.info("Tier summary: %s", tier_summary.to_dict())

    if report_only:
        log.info("--report-only: index.parquet NOT updated.")
        return

    # Write as GeoParquet
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(idx_path, index=False)
    log.info("Wrote GeoParquet index → %s  (%d events)", idx_path, len(index))
    log.info(
        "Columns: %s",
        ", ".join(index.columns.tolist()),
    )


if __name__ == "__main__":
    main()
