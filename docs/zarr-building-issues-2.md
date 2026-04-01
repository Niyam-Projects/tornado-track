# Zarr Building Issue #2 — Incomplete Base Reflectivity Coverage

## Summary

The base reflectivity layer showed incomplete spatial coverage in the event viewer — a
rectangular band of radar data rather than full coverage around the tornado track. Two
bugs were identified and fixed: a silent spatial clipping error in the GRIB decoder, and
use of the wrong MRMS reflectivity product.

---

## Root Cause 1 — Bbox Index Clipping (`round()` instead of `floor`/`ceil`)

`_decode_grib()` in `data/build_zarr_store.py` clips the full MRMS CONUS GRIB grid down
to the event bounding box using integer index arithmetic:

```python
# BEFORE (buggy)
j_start = max(0, int(round((lat1 - maxy) / dlat)))
j_end   = min(Nj - 1, int(round((lat1 - miny) / dlat)))
i_start = max(0, int(round((minx_360 - lon1) / dlon)))
i_end   = min(Ni - 1, int(round((maxx_360 - lon1) / dlon)))
```

`round()` rounds toward the nearest integer in either direction. When used on both the
start **and** end index, it can round the start *inward* and the end *inward* at the
same time, silently shrinking the spatial window by 1–2 grid cells on every edge. At the
MRMS native resolution of ~0.01°/cell (~1 km), this trimmed up to 2 km off each side of
the intended ±50 km spatial buffer.

### Fix

Use `floor()` for start indices (expand outward) and `ceil()` for end indices (expand
outward), ensuring the clipped window is always inclusive of the full buffer:

```python
# AFTER (fixed)
j_start = max(0, int(np.floor((lat1 - maxy) / dlat)))
j_end   = min(Nj - 1, int(np.ceil((lat1 - miny) / dlat)))
i_start = max(0, int(np.floor((minx_360 - lon1) / dlon)))
i_end   = min(Ni - 1, int(np.ceil((maxx_360 - lon1) / dlon)))
```

---

## Root Cause 2 — Wrong MRMS Reflectivity Product

The pipeline was fetching `MergedReflectivityQC_00.50`, the quality-controlled composite.
This product applies aggressive QC processing — AP clutter removal, beam-blockage
correction, anomalous propagation filters — which can delete real precipitation returns,
creating spatial holes in the coverage that looked like a processing artifact.

MRMS also publishes `ReflectivityAtLowestAltitude_00.50` (RALA). RALA composites the
**lowest available elevation scan** from every contributing radar, maximizing spatial
coverage while applying only minimal filtering. It is the better product for storm
structure analysis and tornado tracking.

| Property | `MergedReflectivityQC_00.50` | `ReflectivityAtLowestAltitude_00.50` |
|---|---|---|
| QC filtering | Aggressive (AP, blockage, etc.) | Minimal |
| Spatial coverage | Gaps where QC removes data | Full coverage where any radar exists |
| S3 bucket | `noaa-mrms-pds` | `noaa-mrms-pds` (same) |
| Update cadence | 2 minutes | 2 minutes (same) |
| No-data sentinel | `-999.0` | `-999.0` (same) |
| GRIB format | GRIB2 `.gz` | GRIB2 `.gz` (same) |

### Fix

Replaced `ReflectivityQC` (`MergedReflectivityQC_00.50`) with
`ReflectivityAtLowestAltitude` (`ReflectivityAtLowestAltitude_00.50`) as a drop-in
replacement. Channel count remains 8; no model architecture change is needed.

---

## Files Changed

| File | Change |
|---|---|
| `data/build_zarr_store.py` | `floor`/`ceil` index fix; `_MRMS_NODATA` key renamed |
| `data/mrms_download.py` | `_S3_PRODUCT_MAP` entry swapped to RALA |
| `config/config.yaml` | `mrms.variables[0]` updated |
| `config/__init__.py` | `MRMSConfig.variables` default updated |
| `viz/event_viewer.py` | Colormap dict key updated |

### `_MRMS_NODATA` update (`data/build_zarr_store.py`)

```python
# BEFORE
_MRMS_NODATA: dict[str, float] = {
    "ReflectivityQC": -999.0,
    "MESH":             -1.0,
}

# AFTER
_MRMS_NODATA: dict[str, float] = {
    "ReflectivityAtLowestAltitude": -999.0,
    "MESH":                           -1.0,
}
```

### `_S3_PRODUCT_MAP` update (`data/mrms_download.py`)

```python
# BEFORE
"ReflectivityQC": "MergedReflectivityQC_00.50",

# AFTER
"ReflectivityAtLowestAltitude": "ReflectivityAtLowestAltitude_00.50",
```

---

## Note on Residual NaN Coverage

After both fixes, some NaN areas in the reflectivity layer are **expected and correct**.
RALA only returns values where at least one radar contributes a scan — clear-air regions
(no precipitation) and areas beyond the range of all radars legitimately return the
`-999.0` sentinel, which is masked to NaN. The storm system itself should now show
complete, uninterrupted coverage.

---

## Required Action

All previously built Zarr stores must be **rebuilt** — channel 0 data was produced with
`MergedReflectivityQC` and is incompatible with the renamed variable.

```bash
uv run zarr-build --no-resume   # re-processes all events from scratch
```
