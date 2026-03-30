# Tornado Track Model — Implementation Plan

## Problem Statement

Build a self-contained Python repository that:
1. **Acquires and prepares training data** — downloads MRMS GRIB files from AWS Open Data, fetches known tornado paths from NOAA DAT (2014–present), converts GRIB to xarray, buffers around each track, and stores everything in a Zarr store on `E:\`.
2. **Trains a Reinforcement Learning model** in 3 curriculum stages using CleanRL + PPO.
3. **Evaluates and reports** on training progress and match quality.
4. **Runs local inference** — given a time range and bounding box, outputs a GeoJSON with the predicted track path and confidence-interval polygons.

No REST API is in scope at this stage.

---

## Architecture Overview

### Data Tensor
4D tensor `(C, T, H, W)` with 8 MRMS channels:
- `ReflectivityQC` — base reflectivity
- `AzShear_0-2kmAGL` — low-level azimuthal shear
- `AzShear_3-6kmAGL` — mid-level azimuthal shear
- `MESH` — maximum estimated hail size (proxy for intense updraft)
- `RotationTrack30min` — 0–2 km rotation track (30-min window)
- `RotationTrack60min` — 0–2 km rotation track (60-min window)
- `RotationTrackML30min` — 3–6 km rotation track (30-min window)
- `RotationTrackML60min` — 3–6 km rotation track (60-min window)

### Model Heads
- **Actor Head** → `[dx, dy, dr]` (center displacement + buffer radius)
- **Lifecycle Head** → sigmoid probability P(tornado on ground)
  - Touchdown: P > 0.7
  - Lift: P < 0.3
- **Confidence Output** → Gaussian distribution → inner polygon (mean) + outer polygon (95% CI)
- **EF Classification Head** (optional, add if DAT data includes EF rating) → 6-class softmax (EF0–EF5)

### RL Policy Network
CNN encoder (spatial feature extraction from each timestep) → LSTM (temporal context) → Actor + Lifecycle heads. Trained with PPO via CleanRL.

---

## Repository Structure

```
tornado-track/
├── config/
│   └── config.yaml              # Paths, hyperparams, AWS bucket names
├── data/
│   ├── dat_ingest.py            # Fetch + parse NOAA DAT tornado tracks (2014+)
│   ├── mrms_download.py         # Download MRMS GRIB2 files from AWS per event
│   ├── grib_to_xarray.py        # Convert GRIB2 → xarray with spatial buffer
│   └── build_zarr_store.py      # Assemble 8-channel Zarr store on E:\
├── env/
│   └── tornado_env.py           # Gymnasium environment (TornadoTrackEnv)
├── model/
│   ├── policy.py                # CNN+LSTM actor-critic, Actor + Lifecycle heads
│   └── reward.py                # Reward function (RotationTrack anchoring + lifecycle)
├── training/
│   ├── stage1_follower.py       # Stage 1: Imitation (spawn on known DAT track)
│   ├── stage2_hunter.py         # Stage 2: Discovery (spawn 15 min pre-touchdown)
│   └── stage3_surveyor.py       # Stage 3: Full lifecycle (clear-sky to post-storm)
├── evaluation/
│   └── evaluate.py              # Per-stage reporting, track match quality metrics
├── inference/
│   └── predict.py               # Local inference → GeoJSON output
├── requirements.txt
└── README.md
```

---

## Data Pipeline Detail

### NOAA DAT Source
- URL: `https://apps.dat.noaa.gov/StormDamage/DamageViewer/`
- API/download: `https://apps.dat.noaa.gov/StormDamage/DamageViewer/docs/dat_api_docs.html`
- Filter: last 5 years (approx 2020–present), event type = Tornado
- **Three DAT feature types** stored as separate GeoParquet files:

| File | Geometry | Key Fields |
|------|----------|------------|
| `dat_tracks.parquet` | LineString (storm track polyline) | event_id, ef_rating, start_time, end_time, start_point, end_point (derived from polyline endpoints) |
| `dat_ef_polygons.parquet` | Polygon (EF rating zones) | event_id, ef_scale, area_m2 |
| `dat_damage_points.parquet` | Point (damage survey pts) | event_id, ef_scale, windspeed_mph, lat, lon |

- **Derived fields on `dat_tracks`**: `start_point` (Point, first vertex), `end_point` (Point, last vertex) extracted from the track polyline for direct reference in RL spawn logic
- Storage: `E:\projects\tornado-track\dat\`

### AWS MRMS Source
- Bucket: `s3://noaa-mrms-pds`
- Prefix pattern: `CONUS/{variable}/{YYYYMMDD}/{variable}_00.00_{YYYYMMDD}-{HHMMSS}.grib2.gz`
- Key variables (exact S3 prefixes):
  - `ReflectivityQC`
  - `AzShear_0-2kmAGL`
  - `AzShear_3-6kmAGL`
  - `MESH`
  - `RotationTrack30min`
  - `RotationTrack60min`
  - `RotationTrackML30min`
  - `RotationTrackML60min`
- Time window: ±90 minutes around each tornado event
- Spatial buffer: 50 km around the DAT track bounding box
- **Scope: last 5 years only** (~200 GB total storage target)

### GRIB → xarray (Coordinate Normalization)
- Tool: `cfgrib` + `xarray`
- **MRMS GRIB files use 0–360° longitude** (not standard −180–180). During conversion, longitudes are normalized: `lon = lon - 360 if lon > 180 else lon`
- Output: `xr.Dataset` per event in standard WGS84 (lat: −90–90, lon: −180–180), all 8 variables on common grid, clipped to buffered bbox
- Saved as Zarr chunks on `E:\projects\tornado-track\`

### Zarr Store Layout
```
E:\projects\tornado-track\
├── dat/
│   ├── dat_tracks.parquet          # Track polylines + start/end points + EF rating
│   ├── dat_ef_polygons.parquet     # EF-rated damage polygons
│   └── dat_damage_points.parquet   # Individual damage survey points (efscale, windspeed)
├── events/
│   ├── {event_id}/
│   │   ├── data.zarr               # (C=8, T, H, W) tensor, WGS84 coords
│   │   └── metadata.json           # DAT metadata, EF rating, start/end times
├── index.parquet                   # All events: paths, EF, start/end, split (train/val/test)
└── stats.json                      # Per-channel mean/std for normalization
```

> **Data path is hardcoded** to `E:\projects\tornado-track` throughout all scripts.
> **Storage budget: ~200 GB** achieved by limiting to last 5 years of tornado events.

---

## Training Detail

### Curriculum Stages (5,000 episodes each)

| Stage | Name | Spawn Point | Reward Signal |
|-------|------|-------------|---------------|
| 1 | Follower | Directly on DAT track (t=0) | +high for matching RotationTrack30min + reflectivity |
| 2 | Hunter | 15 min before touchdown | +for approaching RotationTrackML (mid-level) signatures |
| 3 | Surveyor | Clear-sky through post-storm | +for correct lifecycle timing; heavy penalty for false active claims |

### Reward Function (core)
```python
r = w1 * overlap(pred_polygon, dat_polygon)     # Spatial accuracy
  + w2 * rotation_track_mean(pred_center)       # Physics anchoring
  - w3 * false_active_penalty                   # Cost of effort (Stage 3)
  + w4 * lifecycle_accuracy                     # Correct touchdown/lift timing
```

### PPO Hyperparameters (starting point)
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `n_epochs`: 10
- `gamma`: 0.99
- `clip_coef`: 0.2
- `total_timesteps`: 5000 episodes × avg episode length

---

## Evaluation Metrics
- **Track Hausdorff Distance** — max deviation between predicted and DAT path
- **Polygon IoU** — intersection-over-union of predicted swath vs DAT damage polygon
- **Lifecycle F1** — precision/recall on touchdown/lift detection
- **EF Classification Accuracy** (if enabled) — per-class and weighted
- **Episode Reward Curve** — per-stage learning curves with smoothed mean ± std

---

## Implementation Todos

1. `setup` — Create project structure, `requirements.txt`, `config.yaml`; **copy plan PDF and this plan.md to repo root**
2. `dat-ingest` — NOAA DAT fetch (last 5 yrs, tornado only); save three GeoParquet files to `E:\projects\tornado-track\dat\`: `dat_tracks.parquet` (polylines + derived start/end Points + EF), `dat_ef_polygons.parquet`, `dat_damage_points.parquet` (efscale, windspeed)
3. `mrms-download` — Per-event AWS S3 MRMS GRIB downloader with time/space filter (5-yr scope)
4. `grib-to-xarray` — Convert GRIB (0–360° lon → WGS84 −180–180°) + clip + align 8 channels to xarray Dataset
5. `zarr-builder` — Write Zarr store to `E:\projects\tornado-track\events\`, build `index.parquet` + `stats.json`
6. `gym-env` — `TornadoTrackEnv` Gymnasium environment (obs, action, reward, done)
7. `policy-net` — CNN+LSTM policy with Actor + Lifecycle (+ optional EF) heads
8. `reward-fn` — Reward function with RotationTrack anchoring
9. `stage1-train` — Follower training script (CleanRL PPO, 5000 episodes), checkpoints to `E:\projects\tornado-track\checkpoints\stage1\`
10. `stage2-train` — Hunter training script (CleanRL PPO, 5000 episodes), checkpoints to `E:\projects\tornado-track\checkpoints\stage2\`
11. `stage3-train` — Surveyor training script (CleanRL PPO, 5000 episodes), checkpoints to `E:\projects\tornado-track\checkpoints\stage3\`
12. `evaluate` — Evaluation script with all metrics + plots, reports to `E:\projects\tornado-track\reports\`
13. `inference` — Local prediction script → GeoJSON (track + confidence polygons)
14. `readme` — README with setup, data download, training, and inference instructions

---

## Key Technology Choices
| Concern | Choice |
|---------|--------|
| GRIB parsing | `cfgrib` + `eccodes` |
| Array/data | `xarray`, `dask`, `numpy` |
| Zarr storage | `zarr`, `fsspec` |
| Geospatial storage | `geopandas`, `pyarrow` (GeoParquet) |
| RL framework | CleanRL (PPO) |
| Neural network | PyTorch |
| Geospatial ops | `geopandas`, `shapely` |
| DAT API | `httpx` (async) |
| AWS data | `boto3` / `s3fs` (anonymous access) |
| Visualization | `matplotlib`, `cartopy` |
| Config | `pydantic-settings` + YAML |

---

## Notes & Considerations
- **Hardcoded data root:** `E:\projects\tornado-track` — all scripts reference `DATA_ROOT` from `config/config.yaml` which defaults to this path
- **Plan copy:** During `setup`, the original PDF (`Storm Tracking Model Implementation Plan.pdf`) and this `plan.md` are copied to the repository root for reference
- **GRIB longitude normalization:** MRMS GRIB files store longitude in 0–360° range; all conversion scripts remap to standard −180–180° WGS84 before writing to Zarr
- **DAT geometry types:** Three separate GeoParquet files — track polylines (with derived start/end Points), EF polygons, and damage survey points (efscale + windspeed)
- **5-year scope:** Only last 5 years of tornado events to stay within 200 GB storage budget
- MRMS 2-min temporal resolution = ~45 timesteps per 90-min window (T=45 typical)
- Spatial grid: MRMS native ~0.01° (~1 km), crop to ~200×200 cells around event
- EF rating: DAT includes it — add the classification head but make it optional via config flag
- Train/val/test split: 70/15/15 by year (not random) to avoid temporal leakage
- GPU recommended for training; CPU-only inference supported
- Storage estimate: ~200 MB per event × ~1,000 events (5 yrs) ≈ 200 GB on `E:\projects\tornado-track\`
