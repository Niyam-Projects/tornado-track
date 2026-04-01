# Zarr Building Issue #1 — MRMS Product Timestamp Fragmentation

## Summary

Each MRMS product has a different processing latency and arrives at S3 with a unique
sub-minute timestamp offset from the nominal 2-minute observation window. The Zarr
builder was naively unioning all raw timestamps into a single time axis, which meant
**every variable was NaN at every other variable's timestamps**. The result was a
`(C=8, T, H, W)` tensor where at any given timestep only one channel had data — making
the multi-channel tensor essentially useless for training.

---

## Root Cause

MRMS products are all derived from the same 2-minute radar scan cycle, but each product
has a different processing pipeline with a different latency before being posted to S3:

| Product group | Typical S3 timestamp offset |
|---|---|
| `RotationTrack*` | `:00` — arrives at the exact 2-minute mark |
| `AzShear_*` | ~20–25 s after the 2-minute mark |
| `ReflectivityQC` | ~40–42 s after the 2-minute mark |
| `MESH` | ~40–43 s after the 2-minute mark |

A snippet of the raw KnoxTor time axis shows the interleaving clearly:

```
2026-03-26T23:26:00   .   Y   .   .    ← RotationTrack60min only
2026-03-26T23:26:25   .   .   Y   .    ← AzShear_0-2kmAGL only
2026-03-26T23:26:40   .   .   .   .    ← all NaN
2026-03-26T23:26:42   Y   .   .   .    ← ReflectivityQC only
2026-03-26T23:28:00   .   Y   .   .
2026-03-26T23:28:18   .   .   Y   .
2026-03-26T23:28:37   Y   .   .   .
...
```

Because `_process_event()` used `reindex(method=None)` (exact timestamp match) against
the union axis, each variable populated only its own native timestamps and filled
`NaN` everywhere else.

### Observed numbers for KnoxTor (`data.zarr`, pre-fix)

| Metric | Value |
|---|---|
| Total union timestamps | **285** |
| Timestamps with data per variable | **91** (of 285) |
| Timestamps with ALL 8 variables present | **0** |
| Timestamps with ANY variable present | 281 |
| ReflectivityQC + RotationTrack60min overlap | **0** |

All three built events (KnoxTor, GileadTor, 032726-Kenton) showed the same 285-timestamp
fragmentation — this was a systematic bug affecting every built event.

---

## Fix

**Snap all DataArray timestamps to the nearest 2-minute boundary** before building
the union time axis. Since MRMS's native cadence is exactly 2 minutes, rounding to
`"2min"` groups co-temporal observations from all products into the same time slot.

### Code change — `data/build_zarr_store.py`

Added `_snap_times()` helper and called it in `_process_event()` after regridding:

```python
_SNAP_FREQ = "2min"

def _snap_times(channel_arrays: list[xr.DataArray]) -> list[xr.DataArray]:
    """Round each DataArray's time coordinate to the nearest 2-minute boundary."""
    snapped = []
    for da in channel_arrays:
        new_times = pd.DatetimeIndex(da.time.values).round(_SNAP_FREQ)
        da = da.assign_coords(time=new_times)
        # Drop duplicates within this variable (keep first if two snap to same slot)
        _, unique_idx = np.unique(da.time.values, return_index=True)
        if len(unique_idx) < len(da.time):
            da = da.isel(time=sorted(unique_idx))
        snapped.append(da)
    return snapped
```

In `_process_event()`, inserted after `_regrid()`:

```python
# Snap timestamps to nearest 2-minute boundary before building union time axis.
channel_arrays = _snap_times(channel_arrays)
```

---

## Result (post-fix, verified on existing Zarr content)

After snapping, all products that belong to the same 2-minute observation window share
one timestamp:

| Metric | Before fix | After fix |
|---|---|---|
| Total timestamps | 285 | **91** |
| Timestamps with ALL 8 vars | 0 | **59** |
| Timestamps with 7/8 vars (MESH absent) | 0 | **32** |
| Timestamps with ReflectivityQC | 91 / 285 (32%) | **91 / 91 (100%)** |

MESH has fewer total timestamps (59 vs 91 for other variables) because it is a
derived hail-detection product with higher latency and not generated during
non-convective periods. The 32 slots where MESH is absent are expected and benign —
the training env handles per-channel NaN via `nan_to_num`.

---

## Required Action

All previously built Zarr stores must be **rebuilt** — the fix is purely in the build
pipeline and does not affect the store schema.

```bash
# Delete existing stores
uv run zarr-build --no-resume   # re-processes everything

# Or manually clear then rebuild:
# rm -rf E:\projects\tornado-track\events\*\data.zarr
# uv run zarr-build --batch-size 0
```
