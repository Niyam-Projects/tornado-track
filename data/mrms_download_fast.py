"""
Fast parallel MRMS GRIB2 downloader using async HTTP + S3 direct HTTPS.

Same output as mrms_download.py but uses asyncio + httpx to download many
files concurrently.  The bottleneck on a home connection is latency × file
count, not bandwidth, so parallelism matters more than raw speed.

Key improvements over the boto3 version:
  - Async HTTP: --workers concurrent downloads at once (default 16)
  - Day-level key index cached on disk: avoids re-listing S3 every run
  - --step-minutes: thin the 2-min MRMS cadence (e.g. 4 keeps every other
    file, cutting file count in half with little impact on training)
  - Skips events that are already fully cached

Usage:
    uv run mrms-download-fast
    uv run mrms-download-fast --workers 32 --step-minutes 4
    uv run mrms-download-fast --tracks-path E:\\...\\dat_tracks.parquet
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import click
import geopandas as gpd
import httpx
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from config import cfg
from data.mrms_download import _BUFFER_KM, _S3_PRODUCT_MAP, _buffer_bbox, _KM_PER_DEG_LAT

log = logging.getLogger(__name__)

_BUCKET      = cfg.mrms.bucket
_VARIABLES   = cfg.mrms.variables
_TIME_WINDOW = timedelta(minutes=cfg.mrms.time_window_minutes)
_S3_HTTPS    = f"https://{_BUCKET}.s3.amazonaws.com"
_INDEX_FILE  = "mrms_key_index.json"   # cached per-day key listings


# ---------------------------------------------------------------------------
# Day-level key index (avoids re-listing S3 on resume)
# ---------------------------------------------------------------------------

def _load_index(cache_dir: Path) -> dict[str, list[str]]:
    p = cache_dir / _INDEX_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_index(cache_dir: Path, index: dict[str, list[str]]) -> None:
    (cache_dir / _INDEX_FILE).write_text(json.dumps(index))


def _build_day_prefix(product: str, date_str: str) -> str:
    return f"CONUS/{product}/{date_str}/"


def _list_day_keys(s3, product: str, date_str: str) -> list[str]:
    """List all S3 keys for one product on one UTC day."""
    prefix = _build_day_prefix(product, date_str)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _parse_key_dt(key: str) -> datetime | None:
    try:
        ts = Path(key).name.split("_")[-1].replace(".grib2.gz", "")
        return datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def _keys_for_window(
    all_day_keys: list[str],
    dt_start: datetime,
    dt_end: datetime,
    step_minutes: int,
) -> list[str]:
    """Filter day keys to those within [dt_start, dt_end], thinned by step_minutes."""
    result = []
    last_kept: datetime | None = None
    step = timedelta(minutes=step_minutes)
    for key in sorted(all_day_keys):
        dt = _parse_key_dt(key)
        if dt is None or not (dt_start <= dt <= dt_end):
            continue
        if last_kept is None or (dt - last_kept) >= step:
            result.append(key)
            last_kept = dt
    return result


# ---------------------------------------------------------------------------
# Async download helpers
# ---------------------------------------------------------------------------

async def _download_one(
    client: httpx.AsyncClient,
    key: str,
    dest_dir: Path,
    sem: asyncio.Semaphore,
) -> Path | None:
    """Download one S3 key via direct HTTPS, decompress .gz, return .grib2 path."""
    fname = Path(key).name
    gz_path = dest_dir / fname
    grib_path = gz_path.with_suffix("")  # strip .gz

    if grib_path.exists():
        return grib_path

    async with sem:
        if not gz_path.exists():
            url = f"{_S3_HTTPS}/{key}"
            try:
                async with client.stream("GET", url, timeout=60) as resp:
                    resp.raise_for_status()
                    gz_path.parent.mkdir(parents=True, exist_ok=True)
                    with gz_path.open("wb") as f:
                        async for chunk in resp.aiter_bytes(65536):
                            f.write(chunk)
            except Exception as exc:
                log.warning("Download failed %s: %s", key, exc)
                if gz_path.exists():
                    gz_path.unlink(missing_ok=True)
                return None

    # Decompress
    try:
        with gzip.open(gz_path, "rb") as f_in, grib_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)  # free disk space
    except Exception as exc:
        log.warning("Decompress failed %s: %s", gz_path, exc)
        return None

    return grib_path


async def _download_event_async(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    event_id: str,
    dt_start: datetime,
    dt_end: datetime,
    cache_dir: Path,
    key_index: dict[str, list[str]],
    step_minutes: int,
) -> dict[str, list[Path]]:
    """Download all variables for one event concurrently."""
    window_start = dt_start - _TIME_WINDOW
    window_end   = dt_end   + _TIME_WINDOW
    event_cache  = cache_dir / event_id
    result: dict[str, list[Path]] = {}

    tasks = []
    var_key_pairs: list[tuple[str, str]] = []

    for var in _VARIABLES:
        product = _S3_PRODUCT_MAP.get(var)
        if not product:
            result[var] = []
            continue

        var_dir = event_cache / var
        var_dir.mkdir(parents=True, exist_ok=True)

        # Collect keys across all days in the window
        keys: list[str] = []
        day = window_start.date()
        while day <= window_end.date():
            date_str = day.strftime("%Y%m%d")
            index_key = f"{product}/{date_str}"
            if index_key not in key_index:
                # Will be populated by the caller in the index-building phase
                day += timedelta(days=1)
                continue
            keys.extend(key_index[index_key])
            day += timedelta(days=1)

        filtered = _keys_for_window(keys, window_start, window_end, step_minutes)
        if not filtered:
            log.warning("No MRMS keys for var=%s event=%s", var, event_id)
            result[var] = []
            continue

        for key in filtered:
            var_key_pairs.append((var, key))
            tasks.append(_download_one(client, key, var_dir, sem))

    downloaded = await asyncio.gather(*tasks, return_exceptions=True)

    # Re-associate paths with variable names
    for (var, _key), path_or_exc in zip(var_key_pairs, downloaded):
        if isinstance(path_or_exc, Path) and path_or_exc is not None:
            result.setdefault(var, []).append(path_or_exc)
        else:
            result.setdefault(var, [])

    for var in result:
        result[var] = sorted(result[var])

    return result


# ---------------------------------------------------------------------------
# Index builder (one S3 list call per product per day)
# ---------------------------------------------------------------------------

def _ensure_index(
    s3,
    cache_dir: Path,
    dt_starts: list[datetime],
    dt_ends: list[datetime],
) -> dict[str, list[str]]:
    """
    Build (and cache) a day-level key index for all product/day combos
    needed by the given set of events.
    """
    index = _load_index(cache_dir)
    needed: set[tuple[str, str]] = set()

    for dt_start, dt_end in zip(dt_starts, dt_ends):
        ws = dt_start - _TIME_WINDOW
        we = dt_end   + _TIME_WINDOW
        day = ws.date()
        while day <= we.date():
            date_str = day.strftime("%Y%m%d")
            for var in _VARIABLES:
                product = _S3_PRODUCT_MAP.get(var)
                if product:
                    needed.add((product, date_str))
            day += timedelta(days=1)

    missing = [(p, d) for (p, d) in needed if f"{p}/{d}" not in index]

    if missing:
        log.info("Building S3 key index for %d product/day combos…", len(missing))
        for product, date_str in tqdm(missing, desc="Indexing S3 keys"):
            keys = _list_day_keys(s3, product, date_str)
            index[f"{product}/{date_str}"] = keys

        _save_index(cache_dir, index)
        log.info("Index saved → %s", cache_dir / _INDEX_FILE)

    return index


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def _run_async(
    gdf: gpd.GeoDataFrame,
    cache_dir: Path,
    workers: int,
    step_minutes: int,
) -> None:
    s3 = boto3.client("s3", region_name=cfg.mrms.region, config=Config(signature_version=UNSIGNED))

    # Filter rows with valid timestamps
    valid = gdf.dropna(subset=["start_time", "end_time"])
    if len(valid) < len(gdf):
        log.warning("Skipping %d events with missing start/end times", len(gdf) - len(valid))

    dt_starts = [r.to_pydatetime() for r in valid["start_time"]]
    dt_ends   = [r.to_pydatetime() for r in valid["end_time"]]

    # Fix midnight-crossing events where end_time date wasn't incremented
    for i in range(len(dt_ends)):
        if dt_ends[i] < dt_starts[i]:
            dt_ends[i] += timedelta(days=1)
            log.info("Adjusted end_time +1 day for midnight-crossing event (index %d)", i)

    # Build S3 key index up front (one list call per product/day, cached)
    key_index = _ensure_index(s3, cache_dir, dt_starts, dt_ends)

    sem = asyncio.Semaphore(workers)
    limits = httpx.Limits(max_connections=workers + 4, max_keepalive_connections=workers)

    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = []
        for (_, row), dt_start, dt_end in zip(valid.iterrows(), dt_starts, dt_ends):
            event_id = str(row["event_id"])
            tasks.append(
                _download_event_async(
                    client, sem, event_id, dt_start, dt_end,
                    cache_dir, key_index, step_minutes,
                )
            )

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading events"):
            results.append(await coro)

    total_files = sum(len(paths) for r in results for paths in r.values())
    log.info("Done. %d files across %d events. Cache: %s", total_files, len(valid), cache_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(
    tracks_path: Path | None = None,
    cache_dir: Path | None = None,
    workers: int = 16,
    step_minutes: int = 2,
    rebuild_index: bool = False,
) -> None:
    tracks_path = tracks_path or Path(cfg.data.dat_dir) / "dat_tracks.parquet"
    cache_dir   = cache_dir   or Path(cfg.data.root) / "mrms_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if rebuild_index:
        idx_path = cache_dir / _INDEX_FILE
        if idx_path.exists():
            idx_path.unlink()
            log.info("Deleted existing index: %s", idx_path)

    if not tracks_path.exists():
        raise FileNotFoundError(f"DAT tracks not found: {tracks_path}. Run dat-ingest first.")

    gdf = gpd.read_parquet(tracks_path)
    log.info("Loaded %d tornado tracks", len(gdf))
    log.info("Workers: %d  |  Step: %d min  |  Time window: ±%d min",
             workers, step_minutes, cfg.mrms.time_window_minutes)

    asyncio.run(_run_async(gdf, cache_dir, workers, step_minutes))


@click.command()
@click.option("--tracks-path",  default=None,  help="Path to dat_tracks.parquet")
@click.option("--cache-dir",    default=None,  help="Directory to cache downloaded GRIB files")
@click.option("--workers",      default=16,    show_default=True, help="Concurrent downloads")
@click.option("--step-minutes", default=2,     show_default=True,
              help="Keep one file per N minutes (2=all, 4=every-other, 6=every-third)")
@click.option("--rebuild-index", is_flag=True, default=False,
              help="Delete and rebuild the S3 key index from scratch")
def main(
    tracks_path: str | None,
    cache_dir: str | None,
    workers: int,
    step_minutes: int,
    rebuild_index: bool,
) -> None:
    """Fast parallel MRMS GRIB2 downloader (async HTTP, cached S3 key index)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        tracks_path   = Path(tracks_path) if tracks_path else None,
        cache_dir     = Path(cache_dir)   if cache_dir   else None,
        workers       = workers,
        step_minutes  = step_minutes,
        rebuild_index = rebuild_index,
    )


if __name__ == "__main__":
    main()
