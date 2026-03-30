"""
MRMS GRIB2 downloader from AWS Open Data (s3://noaa-mrms-pds).

For each tornado event in dat_tracks.parquet, computes a ±90-minute time window
and 50 km spatial buffer, then downloads the 8 required MRMS GRIB2.gz files.

Files are cached to a local temp directory so re-runs skip already-downloaded files.

Usage:
    python -m data.mrms_download
    python -m data.mrms_download --tracks-path E:\\projects\\tornado-track\\dat\\dat_tracks.parquet
"""
from __future__ import annotations

import gzip
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import boto3
import click
import geopandas as gpd
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from config import cfg

log = logging.getLogger(__name__)

_BUCKET = cfg.mrms.bucket
_VARIABLES = cfg.mrms.variables
_TIME_WINDOW = timedelta(minutes=cfg.mrms.time_window_minutes)
_BUFFER_KM = cfg.mrms.spatial_buffer_km
_KM_PER_DEG_LAT = 111.0

# Maps our logical variable names → actual S3 product folder names (verified live)
# Pattern: CONUS/{S3_PRODUCT}/{YYYYMMDD}/MRMS_{S3_PRODUCT}_{YYYYMMDD}-{HHMMSS}.grib2.gz
_S3_PRODUCT_MAP: dict[str, str] = {
    "ReflectivityQC":       "MergedReflectivityQC_00.50",
    "AzShear_0-2kmAGL":     "MergedAzShear_0-2kmAGL_00.50",
    "AzShear_3-6kmAGL":     "MergedAzShear_3-6kmAGL_00.50",
    "MESH":                 "MESH_00.50",
    "RotationTrack30min":   "RotationTrack30min_00.50",
    "RotationTrack60min":   "RotationTrack60min_00.50",
    "RotationTrackML30min": "RotationTrackML30min_00.50",
    "RotationTrackML60min": "RotationTrackML60min_00.50",
}


def _s3_client():
    """Create anonymous S3 client for public NOAA bucket."""
    return boto3.client(
        "s3",
        region_name=cfg.mrms.region,
        config=Config(signature_version=UNSIGNED),
    )


def _buffer_bbox(minx: float, miny: float, maxx: float, maxy: float, km: float) -> tuple:
    """Expand a bbox by `km` kilometers in all directions."""
    deg_lat = km / _KM_PER_DEG_LAT
    # Longitude degrees depend on latitude; use midpoint latitude
    mid_lat = (miny + maxy) / 2.0
    deg_lon = km / (_KM_PER_DEG_LAT * np.cos(np.radians(mid_lat)))
    return (minx - deg_lon, miny - deg_lat, maxx + deg_lon, maxy + deg_lat)


def _time_range(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Expand start/end by TIME_WINDOW on each side."""
    return start - _TIME_WINDOW, end + _TIME_WINDOW


def _s3_prefixes_for_variable(var: str, dt_start: datetime, dt_end: datetime) -> list[str]:
    """
    Generate S3 key prefixes for a given variable covering the time window.
    MRMS bucket layout:
      CONUS/{product}/{YYYYMMDD}/MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz
    We generate one prefix per day in the window.
    """
    product = _S3_PRODUCT_MAP.get(var)
    if not product:
        log.warning("No S3 product mapping for variable '%s' — skipping", var)
        return []
    prefixes = []
    day = dt_start.date()
    end_day = dt_end.date()
    while day <= end_day:
        prefixes.append(f"CONUS/{product}/{day.strftime('%Y%m%d')}/")
        day += timedelta(days=1)
    return prefixes


def _list_keys_in_window(
    s3, var: str, dt_start: datetime, dt_end: datetime
) -> list[str]:
    """List all S3 keys for a variable that fall within [dt_start, dt_end]."""
    keys = []
    for prefix in _s3_prefixes_for_variable(var, dt_start, dt_end):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Filename: MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz
                # Timestamp is always the last underscore-delimited token before .grib2.gz
                try:
                    fname = Path(key).name
                    ts_str = fname.split("_")[-1].replace(".grib2.gz", "")
                    file_dt = datetime.strptime(ts_str, "%Y%m%d-%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                    if dt_start <= file_dt <= dt_end:
                        keys.append(key)
                except (ValueError, IndexError):
                    continue
    return keys


def _download_key(s3, key: str, dest: Path) -> Path:
    """Download a single S3 key to dest (skip if already exists)."""
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / Path(key).name
    if local.exists():
        return local
    s3.download_file(_BUCKET, key, str(local))
    return local


def _decompress_grib(gz_path: Path) -> Path:
    """Decompress .grib2.gz → .grib2, return path to decompressed file."""
    out = gz_path.with_suffix("")  # removes .gz
    if out.exists():
        return out
    with gzip.open(gz_path, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out


def download_event(
    s3,
    event_id: str,
    dt_start: datetime,
    dt_end: datetime,
    bbox: tuple,
    cache_dir: Path,
) -> dict[str, list[Path]]:
    """
    Download all MRMS GRIB2 files for one tornado event.

    Returns a dict mapping variable name → list of local .grib2 file paths.
    """
    window_start, window_end = _time_range(dt_start, dt_end)
    event_cache = cache_dir / event_id
    event_cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[Path]] = {}
    for var in _VARIABLES:
        keys = _list_keys_in_window(s3, var, window_start, window_end)
        if not keys:
            log.warning("No MRMS keys found for var=%s event=%s", var, event_id)
            result[var] = []
            continue

        var_dir = event_cache / var
        paths = []
        for key in keys:
            gz_path = _download_key(s3, key, var_dir)
            grib_path = _decompress_grib(gz_path)
            paths.append(grib_path)

        result[var] = sorted(paths)
        log.debug("var=%s event=%s → %d files", var, event_id, len(paths))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(tracks_path: Path | None = None, cache_dir: Path | None = None) -> None:
    tracks_path = tracks_path or Path(cfg.data.dat_dir) / "dat_tracks.parquet"
    cache_dir = cache_dir or Path(cfg.data.root) / "mrms_cache"

    if not tracks_path.exists():
        raise FileNotFoundError(f"DAT tracks not found: {tracks_path}. Run dat_ingest first.")

    gdf = gpd.read_parquet(tracks_path)
    log.info("Loaded %d tornado tracks", len(gdf))

    s3 = _s3_client()

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading MRMS events"):
        event_id = str(row["event_id"])
        start_time: datetime = row["start_time"].to_pydatetime()
        end_time: datetime = row["end_time"].to_pydatetime()
        # Fix midnight-crossing events where end_time date wasn't incremented
        if end_time < start_time:
            end_time += timedelta(days=1)
            log.info("Adjusted end_time +1 day for midnight-crossing event %s", event_id)
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        bbox = _buffer_bbox(*bounds, km=_BUFFER_KM)

        try:
            download_event(s3, event_id, start_time, end_time, bbox, cache_dir)
        except Exception as exc:
            log.error("Failed event %s: %s", event_id, exc)

    log.info("MRMS download complete. Cache: %s", cache_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--tracks-path", default=None, help="Path to dat_tracks.parquet")
@click.option("--cache-dir", default=None, help="Directory to cache downloaded GRIB files")
def main(tracks_path: str | None, cache_dir: str | None) -> None:
    """Download MRMS GRIB2 files from AWS for all DAT tornado events."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        tracks_path=Path(tracks_path) if tracks_path else None,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )


if __name__ == "__main__":
    main()
