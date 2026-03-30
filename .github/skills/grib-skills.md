# GRIB2 File Handling — Lessons & Best Practices

Hard-won knowledge from building the MRMS tornado-track pipeline. Use this as a
reference any time you need to read, decode, subset, or stream GRIB2 files.

---

## 1. Never use cfgrib for performance-sensitive work

`cfgrib` (the xarray backend) is convenient but catastrophically slow:

```python
# ❌ cfgrib: 8 seconds per file — loads ENTIRE grid into memory
ds = xr.open_dataset("file.grib2", engine="cfgrib", backend_kwargs={"indexpath": ""})
# At this point, the full 3500×7000 CONUS grid (24.5M floats) is already in RAM
# .sel() after this is fast, but the damage is done
```

```python
# ✅ eccodes direct: 0.23 seconds per file — decode + subset
msgid = eccodes.codes_new_from_message(raw_bytes)
vals = eccodes.codes_get_values(msgid)  # 0.19s — full grid as flat float64 array
data = vals.reshape(Nj, Ni)
subset = data[j_start:j_end+1, i_start:i_end+1]  # 0.0002s — numpy slice
eccodes.codes_release(msgid)
```

**Why:** cfgrib parses every key in the GRIB message, builds coordinate arrays,
creates an xarray Dataset with full metadata, and eagerly loads all values. eccodes
just decompresses the data section and hands you a flat array.

---

## 2. Read GRIB2 from bytes — no temp files needed

```python
# ❌ Temp file dance (slow, Windows file-locking issues)
fd, tmp = tempfile.mkstemp(suffix=".grib2")
os.write(fd, raw_bytes)
os.close(fd)
ds = xr.open_dataset(tmp, engine="cfgrib", ...)
# ... use ds ...
os.unlink(tmp)  # may fail on Windows if handle still open

# ✅ Direct from bytes (fast, no disk I/O, no locking)
msgid = eccodes.codes_new_from_message(raw_bytes)
# ... read what you need ...
eccodes.codes_release(msgid)
```

`eccodes.codes_new_from_message()` accepts a `bytes` object directly. This is the
single biggest win for streaming pipelines — you never touch the filesystem.

---

## 3. Spatial subsetting by array index, not coordinate query

GRIB2 files store grid metadata as scalar keys, not coordinate arrays. Use the grid
parameters to compute array indices for your bounding box:

```python
msgid = eccodes.codes_new_from_message(raw_bytes)

# Grid parameters (read once, same for all files in a product)
Ni = eccodes.codes_get(msgid, "Ni")          # columns (longitude)
Nj = eccodes.codes_get(msgid, "Nj")          # rows (latitude)
lat1 = eccodes.codes_get(msgid, "latitudeOfFirstGridPointInDegrees")   # top-left lat
lon1 = eccodes.codes_get(msgid, "longitudeOfFirstGridPointInDegrees")  # top-left lon
dlat = eccodes.codes_get(msgid, "jDirectionIncrementInDegrees")        # lat step
dlon = eccodes.codes_get(msgid, "iDirectionIncrementInDegrees")        # lon step

vals = eccodes.codes_get_values(msgid)
eccodes.codes_release(msgid)

data = vals.reshape(Nj, Ni).astype(np.float32)

# Compute row/col indices for your bounding box
# Note: MRMS latitude is DESCENDING (55° at top → 20° at bottom)
j_start = max(0, int(round((lat1 - bbox_max_lat) / dlat)))
j_end   = min(Nj - 1, int(round((lat1 - bbox_min_lat) / dlat)))
i_start = max(0, int(round((bbox_min_lon_360 - lon1) / dlon)))
i_end   = min(Ni - 1, int(round((bbox_max_lon_360 - lon1) / dlon)))

subset = data[j_start:j_end+1, i_start:i_end+1].copy()
```

**Key points:**
- `codes_get_values()` returns a flat 1-D array — reshape with `(Nj, Ni)` (rows, cols)
- Latitude is typically **descending** in MRMS (first grid point is the NORTH edge)
- Clipping by index is O(1) — no coordinate search, no xarray overhead
- Always `.copy()` the subset to release the large parent array for GC

---

## 4. MRMS longitude is 0–360° — convert for WGS84

MRMS CONUS data uses 0–360° longitude (e.g., 230°–300° for the US). Most GIS tools
and training pipelines expect standard WGS84 −180°/180°.

```python
# Convert bbox from WGS84 to 0-360 for index computation
minx_360 = bbox_minx % 360  # e.g., -90° → 270°
maxx_360 = bbox_maxx % 360  # e.g., -88° → 272°

# Convert output coordinates back to WGS84
lons_raw = lon1 + np.arange(i_start, i_end + 1) * dlon  # 0-360 range
lons_wgs84 = np.where(lons_raw > 180, lons_raw - 360, lons_raw)  # -180/180

# For latitudes (descending in MRMS)
lats = lat1 - np.arange(j_start, j_end + 1) * dlat
```

**Do NOT** use xarray's `assign_coords` + `sortby` approach — it's slow and
re-indexes the entire array. The arithmetic approach is instant.

---

## 5. eccodes IS thread-safe for read-only operations

Despite common belief, eccodes works fine across threads when each thread uses its
own message handle:

```python
# ✅ Thread-safe — each thread gets its own msgid
def decode_one(raw_bytes, bbox):
    msgid = eccodes.codes_new_from_message(raw_bytes)
    vals = eccodes.codes_get_values(msgid)
    eccodes.codes_release(msgid)
    return clip_to_bbox(vals, bbox)

with ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(decode_one, raw_bytes_list, bboxes))
```

```python
# ❌ NOT thread-safe — cfgrib uses global state (flex scanner)
# This WILL crash with "fatal flex scanner internal error"
with ThreadPoolExecutor(max_workers=4) as pool:
    pool.map(lambda f: xr.open_dataset(f, engine="cfgrib"), files)
```

**The rule:** `codes_new_from_message` + `codes_get` + `codes_get_values` +
`codes_release` are safe per-handle. Do NOT share a `msgid` across threads.

cfgrib's thread-unsafety comes from its use of the eccodes file-based API and the
flex-based index parser, not from `codes_get_values` itself.

---

## 6. Handle missing values explicitly

GRIB2 files use a configurable missing value indicator (often `9999` or `9.999e20`):

```python
miss = eccodes.codes_get(msgid, "missingValue")
data = vals.reshape(Nj, Ni).astype(np.float32)
data[data == miss] = np.nan
```

**Always** read `missingValue` from the message — don't hardcode it. Different GRIB
producers use different sentinel values.

---

## 7. Streaming from S3 beats local caching

When your per-file decode is fast enough (~0.2s with eccodes), streaming beats caching:

```python
import s3fs, gzip

fs = s3fs.S3FileSystem(anon=True)

# Stream + decompress + decode in one shot — no local files
with fs.open("noaa-mrms-pds/CONUS/MergedReflectivityQC_00.50/20240208/file.grib2.gz", "rb") as fh:
    raw = gzip.decompress(fh.read())  # ~0.1s fetch + decompress

msgid = eccodes.codes_new_from_message(raw)  # ~0.2s decode
# ... extract subset ...
```

**Advantages over local cache:**
- No 50+ GB disk footprint
- No stale cache management or cache invalidation
- No index files (873 MB JSON index eliminated)
- Works from any machine with internet access
- Resume is trivial (just check if output Zarr exists)

**Use `s3fs.ls()` with in-memory caching** for file discovery:

```python
_cache = {}

def list_day(fs, product, date_str):
    key = f"{product}/{date_str}"
    if key not in _cache:
        _cache[key] = fs.ls(f"noaa-mrms-pds/CONUS/{product}/{date_str}/", detail=False)
    return _cache[key]
```

`fs.ls()` takes ~0.2–1.1s per call. Cache aggressively within a run.

---

## 8. Parse timestamps from filenames, not GRIB metadata

MRMS filenames contain the exact timestamp:

```
MRMS_MergedReflectivityQC_00.50_20240208-153900.grib2.gz
                                 ^^^^^^^^ ^^^^^^
                                 YYYYMMDD HHMMSS
```

The GRIB `dataTime` key truncates seconds and prints noisy warnings:

```
ECCODES ERROR: Key dataTime (unpack_long): Truncating time: non-zero seconds(39) ignored
```

**Parse from the filename instead:**

```python
def parse_time(s3_path: str) -> datetime:
    fname = s3_path.rsplit("/", 1)[-1]
    ts = fname.split("_")[-1].replace(".grib2.gz", "")
    return datetime.strptime(ts, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
```

This avoids the eccodes warnings entirely and is more reliable.

---

## 9. MRMS grid constants (for reference)

All MRMS CONUS products share the same grid:

| Parameter | Value |
|-----------|-------|
| Ni (columns) | 7000 |
| Nj (rows) | 3500 |
| Resolution | 0.01° (~1.1 km) |
| Latitude range | 54.995°N → 20.005°N (descending) |
| Longitude range | 230.005° → 299.995° (0-360 system) |
| Longitude in WGS84 | -129.995° → -60.005° |
| Total grid points | 24,500,000 |
| File size (compressed) | ~250–300 KB (.grib2.gz) |
| Temporal cadence | 2 minutes |

S3 bucket: `s3://noaa-mrms-pds` (public, anonymous access)
Path pattern: `CONUS/{product}/{YYYYMMDD}/MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz`

---

## 10. Complete optimized decode pattern

Here is the full recommended pattern for reading a GRIB2 file from S3, clipping to a
bounding box, and returning a numpy array with WGS84 coordinates:

```python
import gzip
import eccodes
import numpy as np
import s3fs

def fetch_and_decode(
    fs: s3fs.S3FileSystem,
    s3_path: str,
    bbox: tuple[float, float, float, float],  # (minx, miny, maxx, maxy) in WGS84
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Stream one GRIB2.gz from S3, decode with eccodes, clip to bbox.
    Returns (subset, lats, lons) or None on failure.
    Thread-safe — can run in ThreadPoolExecutor.
    """
    # Fetch + decompress
    try:
        with fs.open(s3_path, "rb") as fh:
            raw = gzip.decompress(fh.read())
    except Exception:
        return None

    # Decode with eccodes (no temp file, thread-safe)
    try:
        msgid = eccodes.codes_new_from_message(raw)
        try:
            Ni   = eccodes.codes_get(msgid, "Ni")
            Nj   = eccodes.codes_get(msgid, "Nj")
            lat1 = eccodes.codes_get(msgid, "latitudeOfFirstGridPointInDegrees")
            lon1 = eccodes.codes_get(msgid, "longitudeOfFirstGridPointInDegrees")
            dlat = eccodes.codes_get(msgid, "jDirectionIncrementInDegrees")
            dlon = eccodes.codes_get(msgid, "iDirectionIncrementInDegrees")
            miss = eccodes.codes_get(msgid, "missingValue")
            vals = eccodes.codes_get_values(msgid)
        finally:
            eccodes.codes_release(msgid)
    except Exception:
        return None

    # Reshape + handle missing values
    data = vals.reshape(Nj, Ni).astype(np.float32)
    data[data == miss] = np.nan

    # Compute bbox indices (WGS84 → 0-360 for MRMS grid)
    minx, miny, maxx, maxy = bbox
    minx_360 = minx % 360
    maxx_360 = maxx % 360

    j_start = max(0, int(round((lat1 - maxy) / dlat)))
    j_end   = min(Nj - 1, int(round((lat1 - miny) / dlat)))
    i_start = max(0, int(round((minx_360 - lon1) / dlon)))
    i_end   = min(Ni - 1, int(round((maxx_360 - lon1) / dlon)))

    subset = data[j_start:j_end+1, i_start:i_end+1].copy()

    # Build WGS84 coordinates
    lats = lat1 - np.arange(j_start, j_end + 1) * dlat
    lons_raw = lon1 + np.arange(i_start, i_end + 1) * dlon
    lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)

    return subset, lats, lons
```

**Performance:** ~0.3s per file (0.1s network + 0.2s decode) with 16 concurrent
workers processing ~720 files in ~14 seconds total.
