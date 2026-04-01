# Stage 1 Training Improvements — Reward Signal Overhaul

Planned improvements to address reward signal quality before the next training run, based on analysis of the flat reward curves and rotation signal architecture from the first run.

---

## Improvement 1: Rotation Anchor Now Uses RotationTracks, Not AzShear

**Problem**
The `rotation_anchor()` reward component was named after RotationTrack channels but its docstring referenced "azimuthal shear scale" — and the original implementation read channels 4 & 5 (`RotationTrack30min` + `RotationTrack60min`), averaging both with equal weight. This mixed the 30-minute and 60-minute accumulations without purpose and ignored the mid-level channel entirely.

**Changes — `model/reward.py`**

| | Before | After |
|--|--------|-------|
| Channels | `[4]` RotationTrack30min + `[5]` RotationTrack60min | `[5]` RotationTrack60min (LL) + `[7]` RotationTrackML60min (ML) |
| Aggregation | `np.nanmean` over radius-5 window | `np.nanmax` over radius-3 window |
| Scoring cap | `clip(mean / 0.01, 0, 1.0)` | `clip(peak, 0, 1.5)` — rewards super-threshold cores |
| Stage 2 weighting | equal | ML=0.7, LL=0.3 (see Improvement 3) |

The 60-minute accumulation products have better signal-to-noise than the 30-minute ones for tracking purposes. Switching to `np.nanmax` ensures the CNN's peak signal (the tornado core) drives the reward rather than being diluted by surrounding low-rotation rain bands.

---

## Improvement 2: Rotation Channels Now Use Clipped Linear Normalization

**Problem**
`_normalize()` applied Z-score normalization to all 8 channels uniformly. For rotation channels (`RotationTrack*`), the background field is almost entirely zero, so the standard deviation is dominated by the quiet background rather than the signal. A strong rotation value of 0.01 s⁻¹ — the physically meaningful tornado detection threshold — normalized to a tiny Z-score (~0.1 or less), making it nearly invisible to the CNN.

**Changes — `env/tornado_env.py` → `_normalize()`**

```python
# Before: Z-score for all channels
data[i] = (data[i] - mean) / std

# After: rotation channels use clipped linear scaling
if "RotationTrack" in var:
    data[i] = np.clip(data[i] / 0.01, 0.0, 1.0)   # 0.01 s⁻¹ → 1.0
else:
    data[i] = (data[i] - mean) / std                # Z-score for all others
```

| Raw value (s⁻¹) | Z-score (before) | Clipped linear (after) |
|-----------------|-------------------|------------------------|
| 0.0 (background) | ~0.0 | 0.0 |
| 0.005 (weak rotation) | ~0.05 | 0.5 |
| **0.01 (tornado threshold)** | **~0.1** | **1.0** |
| 0.02 (strong core) | ~0.2 | 1.0 (clamped) |

This applies to channels 4–7: `RotationTrack30min`, `RotationTrack60min`, `RotationTrackML30min`, `RotationTrackML60min`. The `_ROT_THRESHOLD = 1.0` constant in `reward.py` now corresponds exactly to this boundary.

---

## Improvement 3: Stage 2 (Hunter) Weights Mid-Level Rotation Higher

**Problem**
Stage 2 is designed to train the agent to navigate *before* tornado touchdown — finding the pre-tornadic mesocyclone signature in the 3–6 km mid-level rotation that precedes surface contact. With equal LL/ML weighting, there was no incentive to "look up" at the mid-level signal that predicts touchdown.

**Changes — `model/reward.py` → `rotation_anchor()`**

```python
# Stage 2: weight mid-level higher — agent must find the 3-6 km precursor
w_ml = 0.7 if stage == 2 else 0.5
w_ll = 0.3 if stage == 2 else 0.5

return float(w_ll * ll_score + w_ml * ml_score)
```

| Stage | LL weight | ML weight | Rationale |
|-------|-----------|-----------|-----------|
| 1 (Follower) | 0.5 | 0.5 | Tornado is already on the ground — both levels matter |
| **2 (Hunter)** | **0.3** | **0.7** | **Find the mesocyclone before touchdown** |
| 3 (Surveyor) | 0.5 | 0.5 | Full lifecycle — balanced signal |

---

## Improvement 4: `dat_active` Flags Use Real DAT Timestamps

**Problem**
`_load_dat_info()` contained a placeholder that marked the **middle third** of every episode's timesteps as `dat_active = True`, regardless of the actual tornado timeline. This caused incorrect lifecycle penalties: the agent was penalized for correctly predicting "no tornado" during the actual quiet period, and the `lifecycle_reward` signal was essentially noise.

```python
# Before: heuristic placeholder
t_start = n_steps // 3
t_end = 2 * n_steps // 3
for i in range(t_start, t_end):
    active[i] = True
```

**Changes — `env/tornado_env.py`**

`reset()` now extracts the zarr `time` coordinate:
```python
_zt = ds.coords.get("time", ds.coords.get("t", None))
zarr_times = pd.DatetimeIndex(_zt.values) if _zt is not None else None
```

`_load_dat_info()` uses the actual `start_time`/`end_time` from `dat_tracks.parquet`:
```python
dat_start = _to_naive(pd.Timestamp(match.iloc[0]["start_time"]))
dat_end   = _to_naive(pd.Timestamp(match.iloc[0]["end_time"]))

for i, t in enumerate(zarr_times):
    active[i] = bool(dat_start <= t <= dat_end)
```

Handles both tz-aware and tz-naive timestamps safely. Falls back to all-False (no false penalties) with a warning if timestamps are unavailable.

---

## Summary Table

| # | Improvement | Files Changed | Impact |
|---|-------------|---------------|--------|
| 1 | Rotation anchor uses RotationTrack60min (LL+ML), `nanmax`, radius-3 | `model/reward.py` | Reward locks onto tornado core rather than averaging it away |
| 2 | Clipped linear normalization for rotation channels (0.01 s⁻¹ → 1.0) | `env/tornado_env.py` | CNN sees full-brightness tornado signature instead of ~0.1 Z-score |
| 3 | Stage 2 weights mid-level rotation 70/30 over low-level | `model/reward.py` | Hunter stage now incentivizes pre-tornadic mesocyclone tracking |
| 4 | `dat_active` flags from real DAT `start_time`/`end_time` | `env/tornado_env.py` | Lifecycle reward is meaningful instead of random noise |
