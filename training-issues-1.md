# Stage 1 Training Issues — Post-Mortem

Documented issues discovered and fixed during the first Stage 1 (`train-stage1`) run on 2026-03-30.

---

## Issue 1: Corrupt `index.parquet` — No Training Events Loaded

**Symptom**
```
ValueError: a must be greater than 0 unless no samples are taken
```
Training crashed immediately at `env.reset()` because the event index was empty.

**Root Cause**
`E:\projects\tornado-track\index.parquet` existed but was corrupt (repetition level histogram
size mismatch), so pandas returned an empty DataFrame after filtering for `split == "train"`.

**Fix**
Rebuilt `index.parquet` by scanning all 97 event directories under
`E:\projects\tornado-track\events\`, reading each `metadata.json`, and assigning a random
70/15/15 train/val/test split (all events are from 2026 so year-based splitting was not viable).
Result: 95 events indexed — 66 train / 14 val / 15 test.

---

## Issue 2: Reward Indexing Crash — 2D Slice Passed as 3D Array

**Symptom**
```
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```
Crash on the very first `env.step()` call.

**Root Cause**
In `tornado_env.py`, the observation `obs` (shape `(C, H, W)`) was sliced to `(C, W)` before
being passed to `compute_reward()`:
```python
obs=obs[:, int(self._agent_y), :],   # Pass slice for speed
```
`rotation_anchor()` in `reward.py` then tried to index it as `obs[channel, y0:y1, x0:x1]`,
which requires 3 dimensions.

**Fix**
Removed the premature optimization — passed the full `(C, H, W)` tensor to `compute_reward()`.

---

## Issue 3: PyTorch Installed as CPU-Only Build

**Symptom**
Training log showed `device=cpu` despite an NVIDIA RTX 3050 Ti being present.
Training was projected to take hundreds of hours on CPU.

**Root Cause**
`pyproject.toml` declared `torch>=2.2.0` with no index source, causing uv to resolve the
CPU-only wheel (`torch==2.11.0+cpu`) from PyPI. The GPU driver (581.32, CUDA 13.0) was fine.

**Fix**
- Added `[tool.uv.sources]` pointing torch/torchvision/torchaudio at the PyTorch CUDA 12.6 index.
- Force-reinstalled via `uv pip install ... --index-url https://download.pytorch.org/whl/cu126 --reinstall`
  (uv's lockfile hash cache required bypassing `uv sync` directly).
- Result: `torch==2.11.0+cu126`, `CUDA available: True`, GPU confirmed active.

---

## Issue 4: Episodes Running 3× Too Long — No Truncation

**Symptom**
78 episodes took 4.5 hours. Config specified `max_steps_per_episode: 90` but each episode
was running ~278 steps (the full length of the Zarr store).

**Root Cause**
`tornado_env.step()` only terminated when `self._t >= self._max_t` (end of data).
`max_steps_per_episode` was defined in config and used by the progress bar but never enforced
in the environment itself. The `truncated` return value was hardcoded to `False`.

**Fix**
Added an `_episode_step` counter to `TornadoTrackEnv`:
- Reset to `0` in `reset()`
- Incremented each `step()`
- `truncated = self._episode_step >= cfg.training.max_steps_per_episode`

Episodes now cap at 90 steps (~3× faster per episode).

---

## Issue 5: IoU Reward Dead for 91% of Training Events

**Symptom**
`train/episode_reward` was flat (~120) with no upward trend over 78 episodes.
Entropy was *increasing* (policy becoming more random, not converging).

**Root Cause**
The primary spatial reward `polygon_iou` required an EF damage polygon from
`dat_ef_polygons.parquet`. Only **6 of 66 training events** had a matching EF polygon —
the other 60 episodes received zero spatial reward every step.

**Fix**
Replaced `polygon_iou` with `track_proximity_reward`: a Gaussian distance-based reward
measuring how close the agent's position is to the DAT tornado track `LineString`.
All 66 training events have a DAT track (100% coverage).

```
reward = exp(-0.5 × (dist_km / σ)²)    where σ = 10 km
```

| Distance from track | Reward |
|---------------------|--------|
| 0 km (on track)     | 1.00   |
| 10 km               | 0.61   |
| 20 km               | 0.14   |
| 30 km               | 0.01   |

Config key renamed `w_polygon_iou` → `w_track_proximity`.

---

## Issue 6: MRMS Channel Order Scrambled — Wrong Normalization Applied

**Symptom**
Not immediately visible — silent data corruption affecting all training.

**Root Cause**
`xr.Dataset.to_array(dim="channel")` sorts variable names **alphabetically**, not in insertion
order. The configured variable order is:

| Index | Config order (expected) | xarray alphabetical (actual) |
|-------|------------------------|------------------------------|
| 0     | ReflectivityQC         | AzShear_0-2kmAGL ✗           |
| 1     | AzShear_0-2kmAGL       | AzShear_3-6kmAGL ✗           |
| 2     | AzShear_3-6kmAGL       | MESH ✗                        |
| 3     | MESH                   | ReflectivityQC ✗              |
| 4     | RotationTrack30min     | RotationTrack30min ✓          |
| 5–7   | RotationTrack*         | RotationTrack* ✓              |

`_normalize()` then applied e.g. ReflectivityQC statistics (mean ≈ −564, std ≈ 476) to the
AzShear_0-2kmAGL channel (mean ≈ 0.0003), producing wildly incorrect normalized values.
The reward's `rotation_anchor()` used hardcoded channel indices 4 and 5 which happened to
be correct, but the CNN saw corrupted inputs for channels 0–3.

**Fix**
Explicitly select and reorder channels to match `cfg.mrms.variables` before converting to numpy:
```python
present = [v for v in cfg.mrms.variables if v in zarr_vars]
data = ds[present].to_array(dim="channel").sel(channel=present).values
```
Any missing variables are zero-padded and a warning is logged. `_normalize()` also now
logs a warning rather than silently skipping missing stats.

---

## Remaining Concern: Maysville Tornado Outlier

`Maysville Tornado` has **2,960 timesteps** vs. a median of 278 — roughly 10× the normal
episode length. When randomly sampled during training it will cause one very slow episode.
Consider capping `n_timesteps` in the index or moving this event to the test split.

---

## Summary Table

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Corrupt index.parquet | Training wouldn't start | ✅ Fixed |
| 2 | Reward 2D/3D indexing crash | Training crashed on step 1 | ✅ Fixed |
| 3 | CPU-only PyTorch build | ~300× slower training | ✅ Fixed |
| 4 | No episode truncation | 3× too long per episode | ✅ Fixed |
| 5 | IoU reward dead (91% events) | No spatial learning signal | ✅ Fixed |
| 6 | MRMS channels scrambled | Corrupted CNN inputs | ✅ Fixed |
| 7 | Maysville Tornado (2960 steps) | Occasional very slow episode | ⚠ Open |
