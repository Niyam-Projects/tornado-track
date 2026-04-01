# 🌪️ tornado-track

Reinforcement-learning model that tracks and identifies tornado storm paths from MRMS radar data.

See [`PLAN.md`](PLAN.md) and [`Storm Tracking Model Implementation Plan.pdf`](Storm%20Tracking%20Model%20Implementation%20Plan.pdf) for full design rationale.

---

## Quick start (all commands in order)

```powershell
# 1. Install dependencies
uv sync

# 2. Ingest tornado track metadata from NOAA DAT (5 years, ~5–20 min)
uv run dat-ingest

# 3. Build Zarr training store (streams MRMS GRIB2 directly from S3 — no separate download step)
uv run zarr-build                      # 100 most recent unprocessed events per batch
uv run zarr-build --batch-size 0       # all events (~6.4 days, ~200 GB)
uv run zarr-build --workers 32         # more parallelism on fast connections

# 4. Scan events for signal quality — scores every event by RotationTrack60min peak intensity,
#    classifies into Monster / Moderate / Weak tiers, and updates index.parquet.
uv run scan-events                     # scan all events
uv run scan-events --report-only       # print quality report without writing to index

# 4a. (Optional) Visualize events locally in a browser to inspect radar signal vs DAT track
uv run viz-events                      # opens http://localhost:8501

# 5. Train Stage 1 — Follower (~6 hrs on GPU)
#    By default, Stage 1 trains only on Tier 1 "Monster" events (Kankakee Curriculum)
uv run train-stage1                    # Monster events only (--tier 1 default)
uv run train-stage1 --tier 2           # expand to Monster + Moderate

# 6. Train Stage 2 — Hunter (~6 hrs on GPU)
uv run train-stage2                    # Monster + Moderate events (--tier 2 default)

# 7. Train Stage 3 — Surveyor (~8 hrs on GPU)
uv run train-stage3                    # all events (--tier 3 default)

# 8. Evaluate the trained model against the held-out test split
uv run evaluate

# 9. Run inference on a specific time window and bounding box
uv run predict --start "2023-05-06T22:00:00Z" --end "2023-05-07T00:30:00Z" --bbox "-99.5,35.0,-98.0,36.5" --output my_track.geojson
```

Monitor training in real time:
```powershell
uv run tensorboard --logdir "E:\projects\tornado-track\reports\tensorboard"
# open http://localhost:6006
```

---

## System requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- GPU strongly recommended for training; CPU-only works for inference
- ~200 GB free on `E:\` (Zarr training store)
- No AWS credentials required (MRMS data is on public S3)

> **GRIB2 support:** `zarr-build` uses the `eccodes` C library directly for fast GRIB2
> decoding (~35× faster than cfgrib). On Windows the easiest install is
> `conda install -c conda-forge eccodes`. Without it, the Zarr build step will fail.
> Steps 1, 4–9 do not require it.

---

## Configuration

All paths and hyperparameters live in [`config/config.yaml`](config/config.yaml).
Data is written to `E:\projects\tornado-track`. Change every `E:\\projects\\tornado-track`
in that file if you need a different root.

Key settings:

```yaml
mrms:
  time_window_minutes: 90    # ± minutes around each tornado event
  spatial_buffer_km: 50      # buffer around the DAT track bounding box

training:
  episodes_per_stage: 5000   # episodes per curriculum stage

model:
  ef_classification: true    # set false to skip EF rating head

# Kankakee Curriculum — SNR tier thresholds (populated by scan-events)
curriculum:
  monster_threshold: 0.015   # peak RotationTrack60min s⁻¹ → Tier 1 "Monster"
  weak_threshold: 0.005      # peak RotationTrack60min s⁻¹ → Tier 3 "Weak"
  stage1_tier: 1             # Stage 1 trains only on Monster events
  stage2_tier: 2             # Stage 2 adds Moderate events
  stage3_tier: 3             # Stage 3 uses all events

# Rotation channel normalization — Clipped Robust Scaler
normalization:
  rotation_clip_low: 0.010   # s⁻¹ below this → 0.0 (background suppressed)
  rotation_clip_high: 0.040  # s⁻¹ at or above this → 1.0 (Kankakee-scale peak)
```

---

## Step-by-step reference

Every `uv run <name>` command below maps to a script entrypoint defined in `pyproject.toml`.
You can also call the underlying module directly with `uv run python -m <module>` if you need
to pass extra arguments not exposed on the short form.

---

### Step 1 — Ingest NOAA DAT tornado tracks

```powershell
uv run dat-ingest
```

Fetches tornado events from the NOAA Damage Assessment Toolkit REST API (last 5 years, no
login required) and saves three GeoParquet files:

| File | Contents |
|------|----------|
| `E:\projects\tornado-track\dat\dat_tracks.parquet` | Track polylines + EF rating + derived start/end points |
| `E:\projects\tornado-track\dat\dat_ef_polygons.parquet` | EF-rated damage zone polygons |
| `E:\projects\tornado-track\dat\dat_damage_points.parquet` | Survey points with EF scale and windspeed |

Options:
```powershell
uv run dat-ingest --years-back 3
uv run dat-ingest --out-dir "D:\my-data\dat"
```

**Runtime:** 5–20 min.

---

### Step 2 — Build the Zarr training store

```powershell
uv run zarr-build
```

Streams MRMS GRIB2 radar data directly from AWS S3 via `s3fs`, decodes each file in
memory with `eccodes` (35× faster than the cfgrib library), normalizes longitudes to
standard WGS84 −180/180°, clips and regrids to a 200×200 grid, and writes per-event
Zarr stores. No separate download step is needed — there is no local GRIB cache.

> **Previous iterations:** Two earlier MRMS downloaders (`mrms-download` and
> `mrms-download-fast`) still exist in the codebase. They are not part of the current
> workflow. See [Legacy MRMS Downloaders](#legacy-mrms-downloaders) for details.

For each event it covers ±90 minutes and a 50 km spatial buffer around the DAT track,
fetching all 8 MRMS radar variables:

- `ReflectivityQC` — base reflectivity
- `AzShear_0-2kmAGL` / `AzShear_3-6kmAGL` — low- and mid-level azimuthal shear
- `MESH` — max estimated hail size (updraft proxy)
- `RotationTrack30min` / `RotationTrack60min` — low-level rotation tracks
- `RotationTrackML30min` / `RotationTrackML60min` — mid-level rotation tracks

Output layout:
```
E:\projects\tornado-track\
├── events\{event_id}\data.zarr
├── events\{event_id}\metadata.json
├── index.parquet
└── stats.json
```

Options:
```powershell
uv run zarr-build --batch-size 50    # 50 events per run (default: 100)
uv run zarr-build --batch-size 0     # process ALL events
uv run zarr-build --workers 32       # more concurrent S3 fetches (default: 16)
uv run zarr-build --no-resume        # rebuild all events from scratch
```

Resume mode is on by default — re-runs skip events that already have a Zarr store.

**Runtime:** ~2.3 min/event (network-bound). 100 events ≈ 4 hours. All ~4,000 events ≈ 6.4 days.
**Storage:** ~200 GB for the full Zarr store.

---

### Step 3 — Scan events for signal quality

```powershell
uv run scan-events
```

Reads each downloaded Zarr store, extracts the peak `RotationTrack60min` value, and
classifies events into **three quality tiers** for the Kankakee Curriculum:

| Tier | Name | Threshold | Training use |
|------|------|-----------|--------------|
| 1 | **Monster** | `max_rotation_score > 0.015 s⁻¹` | Stage 1 core training (high-SNR "type specimens") |
| 2 | **Moderate** | `0.005–0.015 s⁻¹` | Introduced in Stage 2 curriculum |
| 3 | **Weak** | `< 0.005 s⁻¹` | Stage 3 (all events, including messy/low-signal) |

New columns added to `index.parquet`:

| Column | Description |
|--------|-------------|
| `max_rotation_score` | Peak `RotationTrack60min` value (s⁻¹) across all timesteps |
| `mean_rotation_core` | Mean of pixels exceeding the weak threshold |
| `active_pixel_count` | Count of above-threshold pixels |
| `n_timesteps` | T dimension of the Zarr store |
| `data_completeness` | Fraction of the 8 expected MRMS variables present |
| `rotation_tier` | `"monster"` / `"moderate"` / `"weak"` |
| `curriculum_stage` | `1` / `2` / `3` — matches the `--tier` flag on train scripts |

Options:
```powershell
uv run scan-events --report-only                     # print report, don't write index
uv run scan-events --monster-threshold 0.020         # custom tier boundary
uv run scan-events --force                           # re-scan already-scored events
```

**Runtime:** ~5–30 seconds for 100 events (reads only one channel per zarr, lazy I/O).

---

### Step 3a — Visualize events (optional)

```powershell
uv run viz-events
# opens http://localhost:8501
```

Launches a local Streamlit web app that lets you visually inspect downloaded events
before training. Designed for the "Type Specimen" workflow — see a MRMS frame,
toggle layers, and compare the radar signal against the NOAA DAT ground-truth track.

**Features:**
- Event selector dropdown (sorted by `max_rotation_score` if scan-events has been run)
- Timestep slider — step through the MRMS radar frame by frame
- Per-layer toggles + opacity sliders for all 8 MRMS channels
- Normalization range sliders (min/max clamp) per layer to tune contrast
- Color scheme selector per layer (viridis, plasma, turbo, RdBu, etc.)
- Folium interactive map with MRMS rendered as image overlays
- **DAT Track overlay** — the known tornado path as a colored polyline (EF-rated)
- **EF Damage Polygon overlay** — surveyed damage zones by EF rating
- **Damage Survey Points** — individual field survey locations with EF scale and windspeed
- Event metadata panel: EF rating, start/end time, quality tier, max rotation score

Options:
```powershell
uv run viz-events --port 8502    # use a different port
```

---

### Step 4 — Train Stage 1: Follower

```powershell
uv run train-stage1
```

Agent spawns directly on the known DAT track at t=0 and learns that high
`RotationTrack30min` + high reflectivity = reward. Teaches the physics of following
an active tornado.

By default, Stage 1 trains only on **Tier 1 "Monster" events** (see [Kankakee Curriculum](#kankakee-curriculum--signal-quality-tiers)). This prevents the model from being confused by low-SNR events during the critical first phase of learning.

- Episodes: **5,000**
- Checkpoint: `E:\projects\tornado-track\checkpoints\stage1\checkpoint_final.pt`
- TensorBoard logs: `E:\projects\tornado-track\reports\tensorboard\stage1_follower_*\`

Options:
```powershell
uv run train-stage1 --episodes 100    # quick smoke test
uv run train-stage1 --tier 2          # expand to Monster + Moderate once Stage 1 converges
uv run train-stage1 --tier 3          # use all events (no SNR filter)
```

**Runtime:** ~6 hrs on GPU.

---

### Step 5 — Train Stage 2: Hunter

```powershell
uv run train-stage2
```

Agent spawns ~15 min before touchdown and must navigate toward intensifying
`RotationTrackML` signatures. Teaches the model to anticipate tornado initiation.
Auto-loads the Stage 1 checkpoint. Default: Tier 1 + Tier 2 events.

- Episodes: **5,000**
- Checkpoint: `E:\projects\tornado-track\checkpoints\stage2\checkpoint_final.pt`

Options:
```powershell
uv run train-stage2 --checkpoint-in "E:\projects\tornado-track\checkpoints\stage1\checkpoint_final.pt"
uv run train-stage2 --episodes 2000
uv run train-stage2 --tier 1          # keep Monster-only if reward isn't yet converged
```

**Runtime:** ~6 hrs on GPU.

---

### Step 6 — Train Stage 3: Surveyor

```powershell
uv run train-stage3
```

Full lifecycle episodes from clear-sky through post-storm. Model must correctly signal
tornado touchdown and lift. Heavy penalty for false positives. Auto-loads Stage 2 checkpoint.
Default: all tiers (Tier 1 + 2 + 3).

- Episodes: **5,000**
- Checkpoint: `E:\projects\tornado-track\checkpoints\stage3\checkpoint_final.pt`

Options:
```powershell
uv run train-stage3 --checkpoint-in "E:\projects\tornado-track\checkpoints\stage2\checkpoint_final.pt"
```

**Runtime:** ~6–10 hrs on GPU.

---

### Step 7 — Evaluate

```powershell
uv run evaluate
```

Runs the Stage 3 model over the held-out test years and reports:

| Metric | Description |
|--------|-------------|
| Track Hausdorff Distance | Max deviation between predicted and DAT path (grid cells) |
| Polygon IoU | Intersection-over-union of predicted swath vs DAT damage polygon |
| Lifecycle F1 | Precision/recall on touchdown and lift detection |
| EF Classification Accuracy | Per-class accuracy for EF0–EF5 (if enabled) |

Outputs:
- `E:\projects\tornado-track\reports\evaluation.json`
- `E:\projects\tornado-track\reports\plots\evaluation_metrics.png`

Options:
```powershell
uv run evaluate --split val
uv run evaluate --checkpoint "E:\projects\tornado-track\checkpoints\stage3\checkpoint_final.pt"
```

---

### Step 8 — Inference

```powershell
uv run predict `
    --start  "2023-05-06T22:00:00Z" `
    --end    "2023-05-07T00:30:00Z" `
    --bbox   "-99.5,35.0,-98.0,36.5" `
    --output my_track.geojson
```

Downloads MRMS data on demand, runs 200 stochastic rollouts, writes a GeoJSON with:

| Feature | Geometry | Description |
|---------|----------|-------------|
| `primary_track` | `LineString` | Mean predicted path |
| `confidence_swath_1.0sigma` | `Polygon` | 68% confidence swath |
| `confidence_swath_2.0sigma` | `Polygon` | 95% confidence swath |
| `touchdown` | `Point` | Detected tornado touchdown |
| `lift` | `Point` | Detected tornado lift |

Each feature includes a `confidence` float (0–1); values >0.7 indicate tornado on ground.

Options:
```powershell
uv run predict ... --checkpoint "E:\projects\tornado-track\checkpoints\stage3\checkpoint_final.pt"
```

---

## Project layout

```
tornado-track/
├── config/
│   ├── __init__.py             ← pydantic config loader (AppConfig)
│   └── config.yaml             ← all settings (paths, hyperparams, MRMS variables,
│                                              curriculum tiers, normalization thresholds)
├── data/
│   ├── dat_ingest.py           ← NOAA DAT → GeoParquet (tracks, EF polygons, damage points)
│   ├── build_zarr_store.py     ← S3 streaming → eccodes decode → Zarr (main pipeline)
│   ├── scan_events.py          ← SNR scoring → rotation tiers → index.parquet enrichment
│   ├── mrms_download.py        ← [legacy] sequential boto3 GRIB downloader
│   ├── mrms_download_fast.py   ← [legacy] async GRIB downloader with cached index
│   └── grib_to_xarray.py       ← [legacy] GRIB2 → xarray (lon 0-360 → WGS84)
├── env/
│   └── tornado_env.py          ← Gymnasium environment (3 spawn modes, tier filtering,
│                                              Clipped Robust Scaler normalization)
├── model/
│   ├── policy.py               ← CNN + LSTM actor-critic (Actor, Lifecycle, EF heads)
│   └── reward.py               ← Reward: track proximity + RotationTrack anchor + lifecycle
├── training/
│   ├── ppo_base.py             ← CleanRL-style PPO loop (GAE, minibatch, TensorBoard)
│   ├── stage1_follower.py      ← Stage 1: Follower (--tier 1 default)
│   ├── stage2_hunter.py        ← Stage 2: Hunter (--tier 2 default)
│   └── stage3_surveyor.py      ← Stage 3: Surveyor (--tier 3 default)
├── evaluation/
│   └── evaluate.py             ← Metrics (Hausdorff, IoU, Lifecycle F1, EF accuracy) + plots
├── inference/
│   └── predict.py              ← CLI → GeoJSON (track + 1σ/2σ swath + touchdown/lift)
├── viz/
│   └── event_viewer.py         ← Streamlit local event viewer (MRMS layers + DAT overlays)
├── pyproject.toml              ← uv project + script entrypoints
├── requirements.txt            ← pip-compatible fallback
├── PLAN.md                     ← implementation plan
└── Storm Tracking Model Implementation Plan.pdf
```

---

## Model architecture

```
Observation: (C=8, H=200, W=200) — one MRMS timestep
        │
   CNN Encoder  [Conv→BN→ReLU→MaxPool] × 3 layers  →  spatial features
        │
   LSTM  (256 hidden units)  →  temporal context
        │
        ├─► Actor Head     → Gaussian(mean, std) over [dx, dy, dr]
        ├─► Lifecycle Head → sigmoid  P(tornado on ground)   [0.0–1.0]
        ├─► Critic Head    → scalar value estimate
        └─► EF Head        → 6-class softmax  (EF0–EF5)  [optional]
```

Confidence polygons are produced by running **200 stochastic rollouts** through the
Gaussian actor and computing the 1σ and 2σ spatial envelopes of the sampled paths.

---

## Training curriculum

| Stage | Name | Spawn point | Key reward signal | Default tier |
|-------|------|-------------|-------------------|--------------|
| 1 | **Follower** | On DAT track at t=0 | `RotationTrack30min` + reflectivity overlap | Tier 1 Monster only |
| 2 | **Hunter** | 15 min before touchdown | Approach intensifying `RotationTrackML` | Tier 1+2 |
| 3 | **Surveyor** | t=0 (clear-sky) | Lifecycle accuracy; heavy penalty for false actives | All tiers |

Each stage warm-starts from the previous stage's checkpoint.

### Kankakee Curriculum — Signal Quality Tiers

Training on all events simultaneously causes **SNR collapse**: if the model sees 60 weak
events where `RotationTrack60min` never exceeds 0.005 s⁻¹ before hitting a Kankakee-style
event at 0.044 s⁻¹, the gradient updates will be conflicted and convergence will be slow.

The solution is **Prioritized Experience Replay by Signal Quality**. Events are scored
by `scan-events` and classified into tiers:

| Tier | Label | `max_rotation_score` | Strategy |
|------|-------|----------------------|----------|
| 1 | **Monster** | > 0.015 s⁻¹ | Stage 1 "type specimens" — the CNN sees only clear, high-contrast examples |
| 2 | **Moderate** | 0.005–0.015 s⁻¹ | Introduced in Stage 2 once the model knows what a tornado looks like |
| 3 | **Weak** | ≤ 0.005 s⁻¹ | Messy/low-signal events; introduced in Stage 3 for generalization |

**Rotation normalization** uses a Clipped Robust Scaler (not Z-score) to ensure full
CNN contrast on the Monster events:

```
x_norm = clip( (x - 0.010) / (0.040 - 0.010),  0,  1 )
```

- Below 0.010 s⁻¹ → `0.0` (background suppressed / dark)
- At 0.040 s⁻¹ → `1.0` (Kankakee-scale peak = full brightness)
- Values between → linearly stretched across full [0, 1] contrast range

This replaces the old `clip(x / 0.01, 0, 1)` scaler that mapped anything above 0.010 s⁻¹
to a flat `1.0`, making a strong event indistinguishable from a moderate one.

---

## Data pipeline

```
NOAA DAT API
  └─► dat_ingest.py
        ├─► dat_tracks.parquet          (polylines + start/end Points + EF)
        ├─► dat_ef_polygons.parquet     (EF damage zones)
        └─► dat_damage_points.parquet   (survey points: efscale, windspeed)
                    │
AWS s3://noaa-mrms-pds ──────────────┐
                                     │
        build_zarr_store.py  ◄───────┘  (streams GRIB2 directly via s3fs + eccodes)
              ├─► events\{event_id}\data.zarr   (C=8, T, H=200, W=200)
              ├─► index.parquet                 (train/val/test split)
              └─► stats.json                    (channel mean/std)
                    │
        scan_events.py  ◄──────────────────  reads RotationTrack60min per event
              └─► index.parquet (enriched)    adds max_rotation_score + tier columns
                    │
        viz/event_viewer.py  ◄─────────────  Streamlit: MRMS layers + DAT overlays
```


Reinforcement-learning model that tracks and identifies tornado storm paths from MRMS radar data.

See [`PLAN.md`](PLAN.md) and the original [`Storm Tracking Model Implementation Plan.pdf`](Storm%20Tracking%20Model%20Implementation%20Plan.pdf) for full design rationale.

---

## Overview

| Step | Script | Output |
|------|--------|--------|
| 1. Ingest known tornado paths | `data/dat_ingest.py` | 3 GeoParquet files under `E:\projects\tornado-track\dat\` |
| 2. Build Zarr training store | `data/build_zarr_store.py` | Zarr + index + stats under `E:\projects\tornado-track\` |
| 3. Train Stage 1 (Follower) | `training/stage1_follower.py` | Checkpoint → `…\checkpoints\stage1\` |
| 4. Train Stage 2 (Hunter) | `training/stage2_hunter.py` | Checkpoint → `…\checkpoints\stage2\` |
| 5. Train Stage 3 (Surveyor) | `training/stage3_surveyor.py` | Checkpoint → `…\checkpoints\stage3\` |
| 6. Evaluate | `evaluation/evaluate.py` | JSON report + plots → `…\reports\` |
| 7. Run local inference | `inference/predict.py` | GeoJSON with track + confidence polygons |

---

## System requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU strongly recommended for training; CPU-only works for inference
- ~200 GB free on `E:\` for data storage
- No AWS credentials required (MRMS data is publicly accessible)

---

## Setup

### Step 1 — Install uv

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```powershell
uv --version
```

### Step 2 — Install eccodes (required for GRIB2 parsing)

`eccodes` is a C library that must be installed before Python dependencies.
The easiest way on Windows is via conda:

```powershell
conda install -c conda-forge eccodes
```

If you don't have conda, download the eccodes Windows binary from:
https://confluence.ecmwf.int/display/ECC/Releases

### Step 3 — Create the virtual environment and install dependencies

From the repository root:

```powershell
uv sync
```

This reads `pyproject.toml` and installs all dependencies into `.venv\`.

### Step 4 — Install CleanRL from source

CleanRL is not on PyPI. Install it directly into the uv-managed environment:

```powershell
uv pip install "cleanrl @ git+https://github.com/vwxyzjn/cleanrl.git"
```

### Step 5 — Create the data output directory

```powershell
New-Item -ItemType Directory -Force -Path "E:\projects\tornado-track"
```

---

## Configuration

All paths and hyperparameters live in [`config/config.yaml`](config/config.yaml).

The data root is hardcoded to `E:\projects\tornado-track`. If you need to change it,
edit every `E:\\projects\\tornado-track` entry in `config/config.yaml`.

Key settings you may want to review before running:

```yaml
mrms:
  time_window_minutes: 90    # ± minutes around each tornado event
  spatial_buffer_km: 50      # buffer around the DAT track bounding box

training:
  episodes_per_stage: 5000   # episodes per curriculum stage

model:
  ef_classification: true    # set false to disable EF rating head
```

---

## Step-by-step usage

All commands below use `uv run` so they automatically use the project virtual environment.
Run every command from the repository root directory.

---

### Step 1 — Ingest NOAA DAT tornado tracks (last 5 years)

```powershell
uv run python -m data.dat_ingest
```

**What it does:** Fetches tornado events from the NOAA Damage Assessment Toolkit API
(last 5 years, no login required) and saves three GeoParquet files:

| File | Contents |
|------|----------|
| `E:\projects\tornado-track\dat\dat_tracks.parquet` | Track polylines, EF rating, derived start & end points |
| `E:\projects\tornado-track\dat\dat_ef_polygons.parquet` | EF-rated damage zone polygons |
| `E:\projects\tornado-track\dat\dat_damage_points.parquet` | Individual survey points with EF scale and windspeed |

**Options:**
```powershell
# Fetch only the last 3 years
uv run python -m data.dat_ingest --years-back 3

# Write to a different directory
uv run python -m data.dat_ingest --out-dir "D:\my-data\dat"
```

**Expected runtime:** 5–20 minutes depending on API response speed.

---

### Step 2 — Build the Zarr training store

```powershell
uv run zarr-build
```

**What it does:** For each tornado event in `dat_tracks.parquet`, streams MRMS GRIB2
radar files directly from AWS S3 (`s3://noaa-mrms-pds`) via `s3fs`, decodes each file
in memory using the `eccodes` C library (35× faster than `cfgrib`), normalizes
longitudes from 0–360° to standard WGS84 −180/180°, clips and regrids to a 200×200
grid, and writes a per-event Zarr store. No separate download step or local GRIB cache
is needed.

> **Previous iterations:** Two earlier MRMS downloaders (`mrms-download` and
> `mrms-download-fast`) still exist in the codebase. They are not part of the current
> workflow. See [Legacy MRMS Downloaders](#legacy-mrms-downloaders) for why they remain.

**8 MRMS radar variables per event:**
- `ReflectivityQC` — base reflectivity
- `AzShear_0-2kmAGL` — low-level azimuthal shear
- `AzShear_3-6kmAGL` — mid-level azimuthal shear
- `MESH` — max estimated hail size (updraft proxy)
- `RotationTrack30min` / `RotationTrack60min` — low-level rotation tracks
- `RotationTrackML30min` / `RotationTrackML60min` — mid-level rotation tracks

**Output files:**
```
E:\projects\tornado-track\
├── events\{event_id}\data.zarr     ← 8-channel tensor per tornado (C=8, T, H=200, W=200)
├── events\{event_id}\metadata.json ← event metadata (EF, times, bbox)
├── index.parquet                   ← all events with train/val/test split (70/15/15 by year)
└── stats.json                      ← per-channel normalization stats (mean/std)
```

**Options:**
```powershell
uv run zarr-build --batch-size 50    # 50 events per run (default: 100, most recent first)
uv run zarr-build --batch-size 0     # process ALL events
uv run zarr-build --workers 32       # more concurrent S3 fetches (default: 16)
uv run zarr-build --no-resume        # rebuild all events from scratch
```

Resume mode is on by default — re-runs skip events that already have a Zarr store.
Running stats are accumulated across batches so you can process events incrementally.

**Expected runtime:** ~2.3 min/event (network-bound). 100 events ≈ 4 hours. All ~4,000 events ≈ 6.4 days.
**Storage:** ~200 GB for the full Zarr store.

---

### Step 3 — Scan events for signal quality

```powershell
uv run scan-events
```

Reads each Zarr store, computes the peak `RotationTrack60min` value, and classifies
events into tiers for the **Kankakee Curriculum**. See [Step 3 above](#step-3--scan-events-for-signal-quality)
for full details. After this step, `index.parquet` contains `max_rotation_score`,
`rotation_tier`, and `curriculum_stage` columns used by the training scripts.

---

### Step 3a — Visualize events (optional)

```powershell
uv run viz-events
# opens http://localhost:8501
```

Launches a local Streamlit app to inspect individual events on an interactive map before
training. Features:

- Full-screen Folium map (cartodbpositron default) with sidebar controls
- All 8 MRMS channels as toggleable image overlays (per-channel colormap, opacity, value range)
- DAT track, EF damage polygons, and damage survey point overlays
- Timestep slider to scrub through the radar time series frame by frame
- **Summary tab** with tier distribution counts and the full quality report table

```powershell
uv run viz-events --port 8502    # use a different port
```

---

### Step 4 — Train Stage 1: Follower

```powershell
uv run python -m training.stage1_follower
```

**What it does:** Spawns the RL agent directly on the known DAT track at t=0.
The agent learns that high `RotationTrack30min` + high reflectivity = reward.
This teaches the model the **physics of following an active tornado track**.

By default, Stage 1 trains only on **Tier 1 "Monster" events** (see [Kankakee Curriculum](#kankakee-curriculum--signal-quality-tiers)).

- Episodes: **5,000** (configurable with `--episodes N`)
- Checkpoint saved to: `E:\projects\tornado-track\checkpoints\stage1\checkpoint_final.pt`
- TensorBoard logs: `E:\projects\tornado-track\reports\tensorboard\stage1_follower_*\`

**Monitor training progress in real time:**
```powershell
uv run tensorboard --logdir "E:\projects\tornado-track\reports\tensorboard"
```
Then open `http://localhost:6006` in a browser.

**Options:**
```powershell
uv run python -m training.stage1_follower --episodes 100
uv run python -m training.stage1_follower --tier 2   # expand to Monster + Moderate
```

**Expected runtime:** 4–24 hours depending on GPU. With an NVIDIA RTX-class GPU, ~6 hours.

---

### Step 5 — Train Stage 2: Hunter

```powershell
uv run python -m training.stage2_hunter
```

**What it does:** Spawns the agent ~15 minutes before tornado touchdown in a pre-tornadic
environment. The agent must navigate toward intensifying `RotationTrackML` (mid-level)
signatures. This teaches the model to **anticipate tornado initiation** by watching the
mid-levels descend.

Automatically loads the Stage 1 checkpoint as the starting point. Default: Tier 1+2 events.

- Episodes: **5,000**
- Checkpoint saved to: `E:\projects\tornado-track\checkpoints\stage2\checkpoint_final.pt`

**Options:**
```powershell
uv run python -m training.stage2_hunter --checkpoint-in "E:\projects\tornado-track\checkpoints\stage1\checkpoint_final.pt"
uv run python -m training.stage2_hunter --episodes 2000
```

**Expected runtime:** Similar to Stage 1 (~6 hours on GPU).

---

### Step 6 — Train Stage 3: Surveyor

```powershell
uv run python -m training.stage3_surveyor
```

**What it does:** Full time-series episodes from clear-sky conditions through the post-storm
period. The model must correctly signal tornado start (touchdown) and end (lift).
A heavy **cost-of-effort penalty** is applied for every step the model claims a tornado
is active when the DAT shows no damage.

Automatically loads the Stage 2 checkpoint as the starting point. Default: all tiers.

- Episodes: **5,000**
- Checkpoint saved to: `E:\projects\tornado-track\checkpoints\stage3\checkpoint_final.pt`

**Options:**
```powershell
uv run python -m training.stage3_surveyor --checkpoint-in "E:\projects\tornado-track\checkpoints\stage2\checkpoint_final.pt"
```

**Expected runtime:** ~6–10 hours on GPU (episodes are longer — full storm lifecycle).

---

### Step 7 — Evaluate the trained model

```powershell
uv run python -m evaluation.evaluate
```

**What it does:** Loads the Stage 3 checkpoint, runs it over the test split (held-out years),
and reports:

| Metric | Description |
|--------|-------------|
| Track Hausdorff Distance | Max deviation between predicted and DAT path (grid cells) |
| Polygon IoU | Intersection-over-union of predicted swath vs DAT damage polygon |
| Lifecycle F1 | Precision / recall on touchdown and lift detection |
| EF Classification Accuracy | Per-class accuracy for EF0–EF5 (if enabled) |

**Output files:**
- `E:\projects\tornado-track\reports\evaluation.json` — full metrics as JSON
- `E:\projects\tornado-track\reports\plots\evaluation_metrics.png` — bar chart

**Options:**
```powershell
# Evaluate against the validation split instead
uv run python -m evaluation.evaluate --split val

# Use a specific checkpoint
uv run python -m evaluation.evaluate --checkpoint "E:\projects\tornado-track\checkpoints\stage3\checkpoint_final.pt"
```

---

### Step 8 — Run local inference

```powershell
uv run python -m inference.predict `
    --start "2023-05-06T22:00:00Z" `
    --end   "2023-05-07T00:30:00Z" `
    --bbox  "-99.5,35.0,-98.0,36.5" `
    --output my_track.geojson
```

**Arguments:**

| Flag | Description | Example |
|------|-------------|---------|
| `--start` | Start of time window (UTC, ISO 8601) | `"2023-05-06T22:00:00Z"` |
| `--end` | End of time window (UTC, ISO 8601) | `"2023-05-07T00:30:00Z"` |
| `--bbox` | Bounding box: `minlon,minlat,maxlon,maxlat` | `"-99.5,35.0,-98.0,36.5"` |
| `--output` | Output GeoJSON file path | `my_track.geojson` |
| `--checkpoint` | *(optional)* Override checkpoint path | *(defaults to stage3 final)* |

**What it does:**
1. Downloads the required MRMS GRIB files for the requested window (cached locally)
2. Converts and normalizes the data
3. Runs the Stage 3 model with 200 stochastic trajectory samples
4. Writes a GeoJSON containing:

| Feature | Geometry | Description |
|---------|----------|-------------|
| `primary_track` | `LineString` | Most likely tornado path (mean trajectory) |
| `confidence_swath_1.0sigma` | `Polygon` | 68% confidence swath around the track |
| `confidence_swath_2.0sigma` | `Polygon` | 95% confidence swath around the track |
| `touchdown` | `Point` | Location where model detects tornado touchdown |
| `lift` | `Point` | Location where model detects tornado lifting |

Each feature includes a `confidence` property — a per-timestep float from `0.0` to `1.0`
produced by the Lifecycle head (values > 0.7 = tornado on ground).

**Example — open the result in QGIS or any GeoJSON viewer:**
```powershell
# View the output in the browser with a quick Python server
uv run python -m http.server 8080
# Then drag my_track.geojson into https://geojson.io
```

---

## Project layout

```
tornado-track/
├── config/
│   ├── __init__.py             ← pydantic config loader
│   └── config.yaml             ← all settings (paths, hyperparams, MRMS variables)
├── data/
│   ├── dat_ingest.py           ← NOAA DAT → GeoParquet (tracks, EF polygons, damage points)
│   ├── build_zarr_store.py     ← S3 streaming → eccodes decode → Zarr (main pipeline)
│   ├── mrms_download.py        ← [legacy] sequential boto3 GRIB downloader
│   ├── mrms_download_fast.py   ← [legacy] async GRIB downloader with cached index
│   └── grib_to_xarray.py       ← [legacy] GRIB2 → xarray (lon 0-360 → WGS84)
├── env/
│   └── tornado_env.py          ← Gymnasium environment (3 spawn modes)
├── model/
│   ├── policy.py               ← CNN + LSTM actor-critic (Actor, Lifecycle, EF heads)
│   └── reward.py               ← Reward: polygon IoU + RotationTrack anchor + lifecycle
├── training/
│   ├── ppo_base.py             ← Shared CleanRL-style PPO loop (GAE, minibatch, TensorBoard)
│   ├── stage1_follower.py      ← Stage 1: Follower
│   ├── stage2_hunter.py        ← Stage 2: Hunter
│   └── stage3_surveyor.py      ← Stage 3: Surveyor
├── evaluation/
│   └── evaluate.py             ← Metrics (Hausdorff, IoU, Lifecycle F1, EF accuracy) + plots
├── inference/
│   └── predict.py              ← CLI → GeoJSON (track + 1σ/2σ swath + touchdown/lift points)
├── pyproject.toml              ← uv project definition + script entrypoints
├── requirements.txt            ← pip-compatible dependency list (fallback)
├── PLAN.md                     ← implementation plan
└── Storm Tracking Model Implementation Plan.pdf
```

---

## Model architecture

```
Observation: (C=8, H=200, W=200) — one MRMS timestep
        │
   CNN Encoder  [Conv→BN→ReLU→MaxPool] × 3 layers  →  spatial features
        │
   LSTM  (256 hidden units)  →  temporal context
        │
        ├─► Actor Head     → Gaussian(mean, std) over [dx, dy, dr]
        ├─► Lifecycle Head → sigmoid  P(tornado on ground)   [0.0–1.0]
        ├─► Critic Head    → scalar value estimate
        └─► EF Head        → 6-class softmax  (EF0–EF5)  [optional]
```

Confidence polygons are produced by running **200 stochastic rollouts** through the
Gaussian actor and computing the 1σ and 2σ spatial envelopes of the sampled paths.

---

## Training curriculum

| Stage | Name | Spawn point | Key reward signal |
|-------|------|-------------|-------------------|
| 1 | **Follower** | On DAT track at t=0 | `RotationTrack30min` + reflectivity overlap |
| 2 | **Hunter** | 15 min before touchdown | Approach intensifying `RotationTrackML` |
| 3 | **Surveyor** | t=0 (clear-sky) | Lifecycle accuracy; heavy penalty for false actives |

Each stage is warm-started from the previous stage's checkpoint.

---

## Data pipeline

```
NOAA DAT API
  └─► dat_ingest.py
        ├─► dat_tracks.parquet          (polylines + start/end Points + EF)
        ├─► dat_ef_polygons.parquet     (EF damage zones)
        └─► dat_damage_points.parquet   (survey points: efscale, windspeed)
                    │
AWS s3://noaa-mrms-pds ──────────────┐
                                     │
        build_zarr_store.py  ◄───────┘  (streams GRIB2 directly via s3fs + eccodes)
              ├─► events\{event_id}\data.zarr   (C=8, T, H=200, W=200)
              ├─► index.parquet                 (train/val/test by year)
              └─► stats.json                    (channel mean/std)
```

---

## Legacy MRMS Downloaders

The `data/` directory contains two earlier MRMS download implementations that are **not
used in the current workflow**. They remain in the codebase as a record of the iteration
process and as a reference for how the pipeline evolved.

### `mrms_download.py` — Sequential boto3 downloader (v1)

The original downloader. Uses `boto3` to list and download GRIB2 files from
`s3://noaa-mrms-pds` one at a time. For each event it downloads all 8 MRMS variables
covering ±90 minutes, writing `.grib2` files to a local cache under
`mrms_cache\{event_id}\{variable}\`. A separate `grib_to_xarray.py` converter was
then needed to open these files, normalize longitudes, and clip to the event bbox.

**Why it was slow:** Sequential S3 downloads (one file at a time) and the two-step
download-then-convert pipeline meant processing a single event could take 30+ minutes.

```powershell
# Not recommended — kept for reference only
uv run mrms-download
```

### `mrms_download_fast.py` — Async downloader with cached S3 index (v2)

An improved version that pre-builds a day-level S3 key index
(`mrms_cache\mrms_key_index.json`, ~873 MB) on the first run, then uses async HTTP
with up to 16 concurrent downloads. Roughly 4–8× faster than v1, but still wrote
GRIB2 files to a local cache that required a separate Zarr build step to convert.

**Why it was replaced:** Even with concurrent downloads, the pipeline still required
~50 GB of intermediate GRIB cache on disk, and the `cfgrib` library used for GRIB
decoding took ~8 seconds per file (it eagerly loads the full 3500×7000 CONUS grid
into memory).

```powershell
# Not recommended — kept for reference only
uv run mrms-download-fast
```

### `build_zarr_store.py` — Direct S3 streaming with eccodes (v3, current)

The current pipeline eliminates both previous steps. It streams GRIB2 files directly
from S3 via `s3fs`, decodes them in memory using the `eccodes` C library (0.23s/file
vs 8s with cfgrib — a 35× speedup), and writes Zarr stores directly. No intermediate
GRIB cache, no separate conversion step, no 873 MB index file. Fetch and decode run
fully concurrently in a thread pool since eccodes is thread-safe for this use case.

| Version | Approach | Per-event time | Disk overhead |
|---------|----------|----------------|---------------|
| v1 `mrms_download` | Sequential boto3 → local GRIB → cfgrib → Zarr | ~30+ min | ~50 GB GRIB cache |
| v2 `mrms_download_fast` | Async downloads → local GRIB → cfgrib → Zarr | ~10 min | ~50 GB GRIB cache + 873 MB index |
| v3 `zarr-build` | s3fs stream → eccodes in-memory → Zarr | **~2.3 min** | **None** |
