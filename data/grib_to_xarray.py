"""
GRIB2 → xarray converter with 0–360° longitude normalization.

For each tornado event, reads all 8 MRMS GRIB2 files, normalizes longitudes from
0–360° to standard WGS84 –180–180°, clips to the buffered event bounding box,
aligns all variables to a common grid, and returns an xr.Dataset ready for Zarr.

The output Dataset has dimensions (channel, time, y, x) where:
  - channel = 8 (one per MRMS variable, in config order)
  - time    = sorted UTC timestamps within the event window
  - y / x   = latitude / longitude grid cells in WGS84
"""
from __future__ import annotations

import logging
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr

from config import cfg

log = logging.getLogger(__name__)

_VARIABLES = cfg.mrms.variables
_GRID_SIZE = cfg.zarr.grid_size


# ---------------------------------------------------------------------------
# Longitude normalization
# ---------------------------------------------------------------------------
def normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """Convert 0–360° longitude to –180–180° in-place."""
    if not cfg.grib.normalize_lon:
        return ds
    lon_coord = None
    for cname in ("longitude", "lon", "x"):
        if cname in ds.coords:
            lon_coord = cname
            break
    if lon_coord is None:
        return ds
    lon = ds[lon_coord].values
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords({lon_coord: lon})
    return ds


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------
def _load_grib(path: Path, var_name: str) -> xr.DataArray | None:
    """
    Load one GRIB2 file, normalize lon, return a DataArray named `var_name`.
    Returns None on failure.
    """
    try:
        datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexing_time": "valid_time"})
    except Exception as exc:
        log.warning("cfgrib failed on %s: %s", path, exc)
        return None

    if not datasets:
        return None

    ds = datasets[0]
    ds = normalize_lon(ds)

    # Take the first data variable (MRMS files have exactly one variable per file)
    data_vars = list(ds.data_vars)
    if not data_vars:
        return None

    da = ds[data_vars[0]].rename(var_name)
    # Ensure there is a 'time' dimension
    if "valid_time" in da.coords and "time" not in da.dims:
        da = da.expand_dims("time").assign_coords(time=("time", [da.valid_time.values]))
    elif "time" not in da.dims:
        da = da.expand_dims("time")

    return da


# ---------------------------------------------------------------------------
# Bounding box clip
# ---------------------------------------------------------------------------
def _clip_to_bbox(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """Clip a DataArray to (minx, miny, maxx, maxy) in WGS84."""
    minx, miny, maxx, maxy = bbox
    lon_coord = next((c for c in ("longitude", "lon", "x") if c in da.coords), None)
    lat_coord = next((c for c in ("latitude", "lat", "y") if c in da.coords), None)
    if lon_coord and lat_coord:
        da = da.where(
            (da[lon_coord] >= minx) & (da[lon_coord] <= maxx) &
            (da[lat_coord] >= miny) & (da[lat_coord] <= maxy),
            drop=True,
        )
    return da


# ---------------------------------------------------------------------------
# Regrid to common grid
# ---------------------------------------------------------------------------
def _regrid_to_common(das: list[xr.DataArray], grid_size: int) -> list[xr.DataArray]:
    """
    Interpolate all DataArrays to the same (grid_size × grid_size) lat/lon grid
    defined by the union of their extents.
    """
    # Find common lat/lon range
    lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []
    for da in das:
        lat_coord = next((c for c in ("latitude", "lat", "y") if c in da.coords), None)
        lon_coord = next((c for c in ("longitude", "lon", "x") if c in da.coords), None)
        if lat_coord and lon_coord:
            lat_mins.append(float(da[lat_coord].min()))
            lat_maxs.append(float(da[lat_coord].max()))
            lon_mins.append(float(da[lon_coord].min()))
            lon_maxs.append(float(da[lon_coord].max()))

    if not lat_mins:
        return das

    target_lat = np.linspace(min(lat_mins), max(lat_maxs), grid_size)
    target_lon = np.linspace(min(lon_mins), max(lon_maxs), grid_size)

    result = []
    for da in das:
        lat_coord = next((c for c in ("latitude", "lat", "y") if c in da.coords), None)
        lon_coord = next((c for c in ("longitude", "lon", "x") if c in da.coords), None)
        if lat_coord and lon_coord:
            da = da.interp(
                {lat_coord: target_lat, lon_coord: target_lon},
                method="linear",
            )
        result.append(da)
    return result


# ---------------------------------------------------------------------------
# Build event Dataset
# ---------------------------------------------------------------------------
def build_event_dataset(
    grib_files: dict[str, list[Path]],
    bbox: tuple[float, float, float, float],
) -> xr.Dataset | None:
    """
    Given a mapping {variable_name → [grib2_path, ...]}, build and return an
    xr.Dataset with all 8 channels merged, clipped to bbox, on a common grid.

    Returns None if no data could be loaded.
    """
    channel_arrays: list[xr.DataArray] = []

    for var_name in _VARIABLES:
        paths = grib_files.get(var_name, [])
        if not paths:
            log.warning("No files for variable %s", var_name)
            continue

        timestep_das = []
        for p in sorted(paths):
            da = _load_grib(p, var_name)
            if da is not None:
                da = _clip_to_bbox(da, bbox)
                timestep_das.append(da)

        if not timestep_das:
            log.warning("All files failed for variable %s", var_name)
            continue

        try:
            merged = xr.concat(timestep_das, dim="time")
            merged = merged.sortby("time")
        except Exception as exc:
            log.warning("Concat failed for %s: %s", var_name, exc)
            continue

        channel_arrays.append(merged)

    if not channel_arrays:
        return None

    # Regrid to common grid_size × grid_size
    channel_arrays = _regrid_to_common(channel_arrays, _GRID_SIZE)

    # Align on time dimension (take union, fill gaps with NaN)
    all_times = sorted(
        set(t for da in channel_arrays for t in da.time.values)
    )
    aligned = []
    for da in channel_arrays:
        da = da.reindex(time=all_times, method=None, fill_value=np.nan)
        aligned.append(da)

    # Combine into Dataset
    ds = xr.Dataset({da.name: da for da in aligned})
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def convert_event(
    event_id: str,
    grib_files: dict[str, list[Path]],
    bbox: tuple[float, float, float, float],
) -> xr.Dataset | None:
    """
    Convert MRMS GRIB files for one event to an xr.Dataset.

    Args:
        event_id: Unique tornado event identifier.
        grib_files: Dict mapping variable name → list of local .grib2 paths.
        bbox: (minx, miny, maxx, maxy) bounding box in WGS84 degrees.

    Returns:
        xr.Dataset with variables for each MRMS channel, or None on failure.
    """
    log.info("Converting event %s (%d variable groups)…", event_id, len(grib_files))
    ds = build_event_dataset(grib_files, bbox)
    if ds is None:
        log.error("Could not build dataset for event %s", event_id)
    return ds
