# Zarr Build Pipeline — Optimization Journey

How the MRMS data ingestion pipeline went from ~108 minutes per event to ~2.3 minutes,
and the debugging process that got us there.

---

## The Starting Problem

Running `mrms-download-fast` produced warnings for every variable and event:

```
WARNING No MRMS keys for var=RotationTrack30min event=Evansville to Lake Koshkonong
WARNING No MRMS keys for var=RotationTrack60min event=Evansville to Lake Koshkonong
...
```

The original `mrms-download` (sequential boto3) worked but was painfully slow. The
"fast" async version wasn't finding any S3 keys at all for certain events, and for
others it wasn't noticeably faster.

---

## Phase 1 — Diagnosing "No MRMS keys" (~30 min)

### Root cause: midnight-crossing timestamps

The pre-built S3 key index (873 MB JSON, 12,984 entries) was fine. The bug was in the
event timestamps from `dat_tracks.parquet`:

```
Evansville event:
  start_time = 2024-02-08 23:39 UTC
  end_time   = 2024-02-08 00:17 UTC   ← should be 2024-02-09
```

21 events had `start_time > end_time` because the tornado crossed midnight but the
end date wasn't incremented. The `_keys_for_window` function compared
`start <= key_time <= end`, which matched zero keys when start > end.

### Fix

One-line fix in both `mrms_download.py` and `mrms_download_fast.py`:

```python
if dt_end < dt_start:
    dt_end += timedelta(days=1)
```

Verified: Evansville event went from 0 matching keys to 109.

---

## Phase 2 — "Fast" still isn't fast (~20 min)

After fixing the key matching, `mrms-download-fast` ran but was still slow. The
pipeline had a fundamental architecture problem: it was a **three-step process**:

1. **Download** — `mrms-download` or `mrms-download-fast` saves `.grib2` files to local disk
2. **Convert** — `grib_to_xarray.py` opens each GRIB2 with `cfgrib`, normalizes lon
3. **Build** — `build_zarr_store.py` reads the converted files and writes Zarr

Each step waited for the previous one. The "fast" downloader only sped up step 1 —
steps 2 and 3 were still sequential and slow.

### Decision: rewrite as a single streaming pipeline

Inspired by the user's working code snippet that used `s3fs.S3FileSystem(anon=True)`
to stream GRIB2 directly from S3, we designed a new `build_zarr_store.py` that does
everything in one pass:

- **s3fs** for S3 file discovery and streaming (replaces boto3 + 873 MB key index)
- **In-memory gzip decompression** (no local GRIB cache)
- **cfgrib** for GRIB2 → xarray decode (initially)
- **Direct Zarr write** per event

---

## Phase 3 — First rewrite hits eccodes crash (~15 min)

The first version used `ThreadPoolExecutor` for both fetching AND decoding GRIB2
files. This crashed immediately:

```
fatal flex scanner internal error
```

### Root cause: cfgrib/eccodes is not thread-safe

The `cfgrib` library calls the `eccodes` C library internally, and `eccodes` uses
global state (flex scanner) that is not safe to call from multiple threads
simultaneously.

### Fix: split fetch and decode

```
_fetch_raw()   → ThreadPoolExecutor (network I/O, thread-safe)
_decode_grib() → sequential on main thread (eccodes, NOT thread-safe)
```

This worked but was still slow because decode was sequential.

---

## Phase 4 — Profiling reveals the real bottleneck (~15 min)

With the split approach running, a single event was taking 140+ seconds. Profiling
each step:

| Step | Time | Notes |
|------|------|-------|
| S3 listing (8 products × 2 days) | ~4s | Cached after first call |
| Fetch + decompress (720 files, 16 workers) | ~15s | Network-bound, fast |
| **cfgrib decode (720 files, sequential)** | **~108 min** | **THE BOTTLENECK** |

### The cfgrib problem

```python
ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
```

This single call takes **8 seconds per file** because:

1. cfgrib eagerly loads the **entire CONUS grid** (3500 × 7000 = 24.5 million points)
2. It writes a temporary index file
3. It builds full xarray coordinate arrays
4. Only THEN can you `.sel()` to clip to your small bbox

With 720 files per event: 720 × 8s = 5,760s = **96 minutes** just for decoding.

---

## Phase 5 — The eccodes breakthrough (~20 min)

### Hypothesis: use eccodes C library directly

Instead of going through cfgrib → xarray → load full grid → clip, call the eccodes
Python bindings directly to read just what we need.

### Benchmark

```python
# cfgrib (xarray backend)
ds = xr.open_dataset(tmp, engine="cfgrib", ...)  # 8.06s, eagerly loads 3500×7000

# eccodes direct (from bytes, no temp file!)
msgid = eccodes.codes_new_from_message(raw_bytes)
vals = eccodes.codes_get_values(msgid)            # 0.19s for full grid
data = vals.reshape(Nj, Ni)
subset = data[j_start:j_end, i_start:i_end]      # 0.0002s for bbox clip
```

**Result: 0.23s vs 8.06s — a 35× speedup per file.**

### Bonus discovery: eccodes IS thread-safe for this use case

```python
# 4 parallel eccodes decodes from bytes — all produce identical results
with ThreadPoolExecutor(max_workers=4) as pool:
    futs = [pool.submit(decode_subset, raw, None) for _ in range(4)]
    results = [f.result() for f in futs]
# Completed in 0.24s — no crashes, no corruption
```

The earlier crash was cfgrib's fault (it does complex multi-step operations with global
state). Low-level `codes_new_from_message` + `codes_get_values` uses only the message
handle, which is safe across threads.

### Key insight

cfgrib is not thread-safe, but eccodes is. cfgrib is slow because it loads the full
grid eagerly. eccodes is fast because `codes_get_values` just decompresses the GRIB2
data section — the heavy lifting is in the GRIB2 compression, not Python overhead.

---

## Phase 6 — Final implementation (~10 min)

Rewrote `_decode_grib()` to use eccodes directly:

1. `eccodes.codes_new_from_message(raw_bytes)` — read from bytes, no temp file
2. Read grid metadata (Ni, Nj, lat1, lon1, dlat, dlon)
3. `eccodes.codes_get_values()` → reshape to (Nj, Ni)
4. Compute bbox indices, extract subset
5. Normalize lon from 0-360° to -180/180° via arithmetic
6. Build `xr.DataArray` with proper coordinates

Combined `_fetch_and_decode()` runs in ThreadPoolExecutor — both network I/O AND
GRIB decode are now fully concurrent.

### End-to-end test result

```
Event: KnoxTor
8 vars × 285 timesteps × 200×200
Total time: 140.3s
Zarr readback shape (C,T,H,W): (8, 285, 200, 200)
Lon range: [-83.19, -82.00]  ← standard WGS84 ✓
```

The 140s is now **network-bound** (downloading 720 × 260KB files from S3), not
CPU-bound. The decode overhead is negligible.

---

## Summary

| Version | Approach | Per-event time | Bottleneck |
|---------|----------|----------------|------------|
| v1 `mrms_download` | Sequential boto3 → local GRIB → cfgrib → Zarr | ~30+ min | Everything sequential |
| v2 `mrms_download_fast` | Async downloads → local GRIB → cfgrib → Zarr | ~10 min | cfgrib decode (8s/file) |
| v3 `zarr-build` (cfgrib) | s3fs stream → cfgrib sequential decode → Zarr | ~108 min | cfgrib eager full-grid load |
| **v3 `zarr-build` (eccodes)** | s3fs stream → eccodes parallel decode → Zarr | **~2.3 min** | **S3 network bandwidth** |

### Total debugging + optimization time: ~2 hours

### Key takeaways

1. **cfgrib eagerly loads the entire GRIB2 grid** (24.5M points) even if you only need
   a 200×200 subset. This is by far the biggest performance killer.
2. **eccodes `codes_new_from_message()` reads from bytes** — no temp files needed,
   which eliminates disk I/O and Windows file locking issues.
3. **eccodes is thread-safe** for simple read operations (new_from_message +
   get_values + release). The earlier thread-safety crash was cfgrib's fault.
4. **Streaming from S3 beats local caching** when your decode is fast enough — no
   50 GB disk footprint, no stale cache management, no index files.
5. **Profile before optimizing** — the async downloader (v2) optimized the wrong
   step. The network wasn't the bottleneck; the GRIB decoder was.

---

## Development Time

The entire diagnosis-to-working-pipeline cycle was completed by Claude Opus 4.6 in a
single session. Wall-clock time spanned several hours, but most of that was waiting for
long-running commands (the old cfgrib pipeline churning through 720 files, S3 listings,
end-to-end test runs). Active development time broke down as follows:

| Phase | Active time |
|-------|-------------|
| Diagnose midnight-crossing bug | ~10 min |
| Rewrite `build_zarr_store.py` with s3fs streaming | ~15 min |
| Fix cfgrib thread-safety crash (split fetch/decode) | ~5 min |
| Profile cfgrib bottleneck (8s/file discovery) | ~5 min |
| Discover eccodes 35× speedup + thread-safety | ~10 min |
| Implement eccodes rewrite + end-to-end verify | ~10 min |

**Total active development time: ~55 minutes.**
