"""
Fast Zarr store builder — streams MRMS GRIB2 directly from S3 via s3fs.

For each tornado event:
  1. Lists S3 files for each MRMS product/day using s3fs (cached in-memory)
  2. Streams .grib2.gz from S3, decompresses in memory
  3. Decodes with eccodes (35× faster than cfgrib), clips to event bbox
  4. Normalizes lon to WGS84 –180/180, regrids to common grid
  5. Writes per-event Zarr store

Output Zarr is directly readable by xarray for training:
  ds = xr.open_zarr("events/{event_id}/data.zarr")
  data = ds.to_array(dim="channel").values  # (C, T, H, W)

Usage:
    uv run zarr-build                       # 100 most recent unprocessed events
    uv run zarr-build --batch-size 50       # 50 at a time
    uv run zarr-build --batch-size 0        # ALL events
    uv run zarr-build --workers 16          # concurrent S3 fetches per event
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import click
import eccodes
import geopandas as gpd
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from tqdm import tqdm

from config import cfg
from data.mrms_download import _buffer_bbox, _S3_PRODUCT_MAP

log = logging.getLogger(__name__)

_BUCKET = cfg.mrms.bucket
_VARIABLES = cfg.mrms.variables
_TIME_WINDOW = timedelta(minutes=cfg.mrms.time_window_minutes)
_BUFFER_KM = cfg.mrms.spatial_buffer_km
_GRID_SIZE = cfg.zarr.grid_size
_CHUNK_T = cfg.zarr.chunk_t
_CHUNK_H = cfg.zarr.chunk_h
_CHUNK_W = cfg.zarr.chunk_w

# In-memory cache: s3fs listings per (product, date_str) — avoids re-listing
_listing_cache: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# S3 file discovery (s3fs, no index file needed)
# ---------------------------------------------------------------------------

def _list_day(fs: s3fs.S3FileSystem, product: str, date_str: str) -> list[str]:
    """List S3 paths for one product on one UTC day (cached per run)."""
    key = f"{product}/{date_str}"
    if key not in _listing_cache:
        prefix = f"{_BUCKET}/CONUS/{product}/{date_str}/"
        try:
            _listing_cache[key] = fs.ls(prefix, detail=False)
        except Exception:
            _listing_cache[key] = []
    return _listing_cache[key]


def _parse_time(s3_path: str) -> datetime | None:
    """Extract UTC timestamp from an MRMS S3 filename."""
    try:
        fname = s3_path.rsplit("/", 1)[-1]
        ts = fname.split("_")[-1].replace(".grib2.gz", "")
        return datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def _files_in_window(
    fs: s3fs.S3FileSystem,
    product: str,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[str, datetime]]:
    """Collect all S3 paths for a product within [window_start, window_end]."""
    result: list[tuple[str, datetime]] = []
    day = window_start.date()
    while day <= window_end.date():
        for path in _list_day(fs, product, day.strftime("%Y%m%d")):
            dt = _parse_time(path)
            if dt and window_start <= dt <= window_end:
                result.append((path, dt))
        day += timedelta(days=1)
    return sorted(result, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Single-file fetch → DataArray
# ---------------------------------------------------------------------------

def _fetch_raw(
    fs: s3fs.S3FileSystem,
    s3_path: str,
) -> bytes | None:
    """Download and decompress one .grib2.gz from S3 (thread-safe)."""
    try:
        with fs.open(s3_path, "rb") as fh:
            return gzip.decompress(fh.read())
    except Exception as exc:
        log.debug("Fetch failed %s: %s", s3_path, exc)
        return None


def _decode_grib(
    raw: bytes,
    var_name: str,
    bbox: tuple[float, float, float, float],
    timestamp: datetime,
) -> xr.DataArray | None:
    """Decode raw GRIB2 bytes via eccodes — thread-safe, no temp files.

    ~0.23s per file vs ~8s with cfgrib (35× faster).
    """
    try:
        msgid = eccodes.codes_new_from_message(raw)
        try:
            Ni = eccodes.codes_get(msgid, "Ni")
            Nj = eccodes.codes_get(msgid, "Nj")
            lat1 = eccodes.codes_get(msgid, "latitudeOfFirstGridPointInDegrees")
            lon1 = eccodes.codes_get(msgid, "longitudeOfFirstGridPointInDegrees")
            dlat = eccodes.codes_get(msgid, "jDirectionIncrementInDegrees")
            dlon = eccodes.codes_get(msgid, "iDirectionIncrementInDegrees")
            miss = eccodes.codes_get(msgid, "missingValue")
            vals = eccodes.codes_get_values(msgid)
        finally:
            eccodes.codes_release(msgid)

        data = vals.reshape(Nj, Ni).astype(np.float32)
        data[data == miss] = np.nan

        # Clip to bbox (convert WGS84 -180/180 → 0-360 for MRMS grid indexing)
        minx, miny, maxx, maxy = bbox
        minx_360 = minx % 360
        maxx_360 = maxx % 360

        j_start = max(0, int(round((lat1 - maxy) / dlat)))
        j_end = min(Nj - 1, int(round((lat1 - miny) / dlat)))
        i_start = max(0, int(round((minx_360 - lon1) / dlon)))
        i_end = min(Ni - 1, int(round((maxx_360 - lon1) / dlon)))

        subset = data[j_start : j_end + 1, i_start : i_end + 1].copy()

        # Build coordinates in WGS84 -180/180
        lats = lat1 - np.arange(j_start, j_end + 1) * dlat
        lons_raw = lon1 + np.arange(i_start, i_end + 1) * dlon
        lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)

        ts_np = np.datetime64(timestamp.replace(tzinfo=None), "ns")

        return xr.DataArray(
            subset[np.newaxis, :, :],
            dims=["time", "latitude", "longitude"],
            coords={
                "time": [ts_np],
                "latitude": lats,
                "longitude": lons,
            },
            name=var_name,
        )
    except Exception as exc:
        log.debug("Decode failed: %s", exc)
        return None


def _fetch_and_decode(
    fs: s3fs.S3FileSystem,
    s3_path: str,
    var_name: str,
    bbox: tuple[float, float, float, float],
    timestamp: datetime,
) -> xr.DataArray | None:
    """Fetch .grib2.gz from S3 and decode with eccodes (thread-safe)."""
    raw = _fetch_raw(fs, s3_path)
    if raw is None:
        return None
    return _decode_grib(raw, var_name, bbox, timestamp)


# ---------------------------------------------------------------------------
# Regrid all DataArrays to a common grid
# ---------------------------------------------------------------------------

def _regrid(das: list[xr.DataArray], grid_size: int) -> list[xr.DataArray]:
    """Interpolate all DataArrays onto the same grid_size × grid_size lat/lon grid."""
    lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []
    for da in das:
        lat_c = next((c for c in ("latitude", "lat", "y") if c in da.coords), None)
        lon_c = next((c for c in ("longitude", "lon", "x") if c in da.coords), None)
        if lat_c and lon_c:
            lat_mins.append(float(da[lat_c].min()))
            lat_maxs.append(float(da[lat_c].max()))
            lon_mins.append(float(da[lon_c].min()))
            lon_maxs.append(float(da[lon_c].max()))

    if not lat_mins:
        return das

    target_lat = np.linspace(min(lat_mins), max(lat_maxs), grid_size)
    target_lon = np.linspace(min(lon_mins), max(lon_maxs), grid_size)

    result = []
    for da in das:
        lat_c = next((c for c in ("latitude", "lat", "y") if c in da.coords), None)
        lon_c = next((c for c in ("longitude", "lon", "x") if c in da.coords), None)
        if lat_c and lon_c:
            da = da.interp({lat_c: target_lat, lon_c: target_lon}, method="linear")
            rename = {}
            if lat_c != "y":
                rename[lat_c] = "y"
            if lon_c != "x":
                rename[lon_c] = "x"
            if rename:
                da = da.rename(rename)
        result.append(da)
    return result


# ---------------------------------------------------------------------------
# Process one event → Zarr
# ---------------------------------------------------------------------------

def _process_event(
    fs: s3fs.S3FileSystem,
    event_id: str,
    dt_start: datetime,
    dt_end: datetime,
    bbox: tuple[float, float, float, float],
    events_dir: Path,
    workers: int,
) -> xr.Dataset | None:
    """
    Fetch all MRMS files for one event from S3, build xr.Dataset, write Zarr.
    Returns the Dataset on success (for stats), None on failure.
    """
    event_dir = events_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = event_dir / "data.zarr"

    window_start = dt_start - _TIME_WINDOW
    window_end = dt_end + _TIME_WINDOW

    # Collect all (var, path, dt) triples across all variables
    all_tasks: list[tuple[str, str, datetime]] = []
    for var in _VARIABLES:
        product = _S3_PRODUCT_MAP.get(var)
        if not product:
            continue
        files = _files_in_window(fs, product, window_start, window_end)
        if not files:
            log.warning("No S3 files for var=%s event=%s", var, event_id)
        for path, dt in files:
            all_tasks.append((var, path, dt))

    if not all_tasks:
        log.error("No data for event %s — skipping", event_id)
        return None

    # Fetch + decode ALL files concurrently (eccodes is thread-safe)
    var_das: dict[str, list[xr.DataArray]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_and_decode, fs, path, var, bbox, dt): (var, path)
            for var, path, dt in all_tasks
        }
        for fut in as_completed(futures):
            da = fut.result()
            if da is not None:
                var_das.setdefault(da.name, []).append(da)

    # Build per-variable merged arrays
    channel_arrays: list[xr.DataArray] = []
    for var in _VARIABLES:
        if var not in var_das:
            log.warning("All fetches failed for var=%s event=%s", var, event_id)
            continue
        try:
            merged = xr.concat(var_das[var], dim="time").sortby("time")
            channel_arrays.append(merged)
        except Exception as exc:
            log.warning("Concat failed for %s event=%s: %s", var, event_id, exc)

    if not channel_arrays:
        log.error("No data for event %s — skipping", event_id)
        return None

    # Regrid to common grid_size × grid_size
    channel_arrays = _regrid(channel_arrays, _GRID_SIZE)

    # Align time dimension (union of all timestamps, NaN-fill gaps)
    all_times = sorted(set(t for da in channel_arrays for t in da.time.values))
    aligned = [
        da.reindex(time=all_times, method=None, fill_value=np.nan)
        for da in channel_arrays
    ]

    # Build Dataset with one variable per channel — env reads with
    # ds.to_array(dim="channel") → (C, T, H, W)
    ds = xr.Dataset({da.name: da for da in aligned})
    ds = ds.chunk({"time": _CHUNK_T, "y": _CHUNK_H, "x": _CHUNK_W})
    ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    # Metadata sidecar
    meta = {
        "event_id": event_id,
        "start_time": str(dt_start),
        "end_time": str(dt_end),
        "bbox_wgs84": bbox,
        "n_timesteps": len(all_times),
        "n_variables": len(aligned),
        "variables": [da.name for da in aligned],
    }
    with open(event_dir / "metadata.json", "w") as f:
        json.dump(meta, f, default=str, indent=2)

    log.info(
        "✓ %s — %d vars × %d timesteps × %d×%d",
        event_id, len(aligned), len(all_times), _GRID_SIZE, _GRID_SIZE,
    )
    return ds


# ---------------------------------------------------------------------------
# Train / val / test split by year (no temporal leakage)
# ---------------------------------------------------------------------------

def _assign_split(year: int, years: list[int]) -> str:
    sorted_years = sorted(set(years))
    n = len(sorted_years)
    train_end = int(n * cfg.training.train_val_test_split[0])
    val_end = train_end + int(n * cfg.training.train_val_test_split[1])
    idx = sorted_years.index(year) if year in sorted_years else 0
    if idx < train_end:
        return "train"
    elif idx < val_end:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Running normalization statistics (accumulate across batches)
# ---------------------------------------------------------------------------

def _load_running_stats(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _update_stats(running: dict, ds: xr.Dataset) -> None:
    for var in _VARIABLES:
        if var not in ds:
            continue
        vals = ds[var].values.astype(np.float32)
        finite = vals[np.isfinite(vals)]
        if not finite.size:
            continue
        if var not in running:
            running[var] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
        running[var]["sum"] += float(finite.sum())
        running[var]["sum_sq"] += float((finite ** 2).sum())
        running[var]["count"] += int(finite.size)


def _finalize_stats(running: dict) -> dict:
    stats = {}
    for var, acc in running.items():
        n = acc["count"]
        if n == 0:
            continue
        mean = acc["sum"] / n
        variance = acc["sum_sq"] / n - mean ** 2
        std = float(np.sqrt(max(variance, 1e-8)))
        stats[var] = {"mean": float(mean), "std": std}
    return stats


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    batch_size: int = 100,
    workers: int = 16,
    resume: bool = True,
) -> None:
    tracks_path = Path(cfg.data.dat_dir) / "dat_tracks.parquet"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Run dat-ingest first: {tracks_path}")

    events_dir = Path(cfg.data.events_dir)
    events_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_parquet(tracks_path)
    log.info("Loaded %d tornado events", len(gdf))

    # Filter valid timestamps
    valid = gdf.dropna(subset=["start_time", "end_time"]).copy()

    # Fix midnight-crossing events
    mask = valid["end_time"] < valid["start_time"]
    if mask.any():
        valid.loc[mask, "end_time"] += pd.Timedelta(days=1)
        log.info("Fixed %d midnight-crossing events (end < start)", mask.sum())

    # Skip empty / NaN event IDs
    valid = valid[valid["event_id"].notna() & (valid["event_id"].astype(str) != "")]

    # Sort most recent first
    valid = valid.sort_values("start_time", ascending=False).reset_index(drop=True)

    # Skip events with existing zarr (resume mode)
    if resume:
        done_mask = valid.apply(
            lambda r: (events_dir / str(r["event_id"]) / "data.zarr").exists(),
            axis=1,
        )
        n_done = done_mask.sum()
        if n_done:
            valid = valid[~done_mask].reset_index(drop=True)
            log.info("Skipping %d already-processed events", n_done)

    # Apply batch limit
    if batch_size > 0:
        valid = valid.head(batch_size)

    if valid.empty:
        log.info("Nothing to process.")
        return

    log.info(
        "Processing %d events (batch_size=%s, workers=%d)",
        len(valid), batch_size or "ALL", workers,
    )

    fs = s3fs.S3FileSystem(anon=True)

    all_years = [t.year for t in gdf["start_time"].dt.to_pydatetime()]
    index_rows: list[dict] = []

    # Load running stats (accumulate across batches)
    running_stats_path = Path(cfg.data.stats_path).with_suffix(".running.json")
    running_stats = _load_running_stats(running_stats_path) if resume else {}

    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="Building Zarr"):
        event_id = str(row["event_id"])
        dt_start = row["start_time"].to_pydatetime()
        dt_end = row["end_time"].to_pydatetime()
        bounds = row.geometry.bounds
        bbox = _buffer_bbox(*bounds, km=_BUFFER_KM)

        ds = _process_event(fs, event_id, dt_start, dt_end, bbox, events_dir, workers)
        if ds is None:
            continue

        _update_stats(running_stats, ds)

        index_rows.append({
            "event_id": event_id,
            "zarr_path": str(events_dir / event_id / "data.zarr"),
            "ef_rating": row.get("ef_rating"),
            "start_time": dt_start,
            "end_time": dt_end,
            "year": dt_start.year,
            "split": _assign_split(dt_start.year, all_years),
            "state": row.get("state"),
        })

    # Persist running stats (accumulates across batches)
    if running_stats:
        with open(running_stats_path, "w") as f:
            json.dump(running_stats, f, indent=2)

    # Write / merge index
    index_path = Path(cfg.data.index_path)
    if index_rows:
        new_df = pd.DataFrame(index_rows)
        if resume and index_path.exists():
            existing = pd.read_parquet(index_path)
            combined = pd.concat([existing, new_df]).drop_duplicates(
                subset="event_id", keep="last"
            )
            combined.to_parquet(index_path, index=False)
            log.info("Updated index → %s (%d total events)", index_path, len(combined))
        else:
            new_df.to_parquet(index_path, index=False)
            log.info("Wrote index → %s (%d events)", index_path, len(new_df))

    # Write finalized stats (mean/std)
    stats = _finalize_stats(running_stats)
    if stats:
        stats_path = Path(cfg.data.stats_path)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("Wrote normalization stats → %s", stats_path)

    log.info("Done. %d events processed this batch.", len(index_rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--batch-size", default=100, show_default=True,
              help="Events to process per run (0 = all). Most recent first.")
@click.option("--workers", default=16, show_default=True,
              help="Concurrent S3 file fetches per event")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip events with existing Zarr stores")
def main(batch_size: int, workers: int, resume: bool) -> None:
    """Build Zarr training stores by streaming MRMS GRIB2 directly from S3."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    build(batch_size=batch_size, workers=workers, resume=resume)


if __name__ == "__main__":
    main()
