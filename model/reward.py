"""
Reward function for the TornadoTrackEnv.

Combines four components:
  1. track_proximity  — Gaussian reward for being close to the DAT tornado track line
  2. rotation_anchor  — peak RotationTrack value under the agent's position
  3. false_active     — penalty for claiming tornado is active when DAT shows none (Stage 3)
  4. lifecycle        — bonus for correct touchdown / lift detection timing
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from config import cfg

_W_TRACK = cfg.training.reward.w_track_proximity
_W_ROT = cfg.training.reward.w_rotation_anchor
_W_FALSE = cfg.training.reward.w_false_active
_W_LIFE = cfg.training.reward.w_lifecycle

# Channel indices (matches config variable order)
_ROT_LL_60_IDX = 5   # RotationTrack60min     (low-level,  0–2 km)
_ROT_ML_60_IDX = 7   # RotationTrackML60min   (mid-level,  3–6 km)

# Rotation scoring constants
_ROT_THRESHOLD = 1.0  # normalized units — equals 0.01 s⁻¹ after clipped-linear scaling
_ROT_RADIUS = 3       # grid-cell radius of the peak-search window (~3 km)

# Track proximity falloff: reward = 1.0 on the line, ~0.61 at sigma_km, ~0.14 at 2×sigma_km
_TRACK_SIGMA_KM = 10.0
_DEG_TO_KM = 111.0  # 1 degree latitude ≈ 111 km


def track_proximity_reward(
    agent_y: int,
    agent_x: int,
    dat_track: LineString | None,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> float:
    """
    Gaussian reward for how close the agent is to the DAT tornado track line.

    Returns 1.0 when the agent is exactly on the track, decaying to near-zero
    beyond _TRACK_SIGMA_KM. Works for all events that have a DAT track.
    """
    if dat_track is None or len(grid_lat) == 0 or len(grid_lon) == 0:
        return 0.0

    iy = int(np.clip(agent_y, 0, len(grid_lat) - 1))
    ix = int(np.clip(agent_x, 0, len(grid_lon) - 1))
    agent_point = Point(float(grid_lon[ix]), float(grid_lat[iy]))

    # Shapely distance in degrees (lon/lat space)
    dist_deg = agent_point.distance(dat_track)
    dist_km = dist_deg * _DEG_TO_KM

    return float(np.exp(-0.5 * (dist_km / _TRACK_SIGMA_KM) ** 2))


def rotation_anchor(
    obs: np.ndarray,
    agent_y: int,
    agent_x: int,
    stage: int,
    radius_cells: int = _ROT_RADIUS,
) -> float:
    """
    Peak-based rotation reward using RotationTrack60min (LL) and RotationTrackML60min (ML).

    Uses np.nanmax over a tight radius window so the tornado's core signal is not
    washed out by surrounding low-rotation rain bands. Values are in normalized space
    where _ROT_THRESHOLD (1.0) corresponds to 0.01 s⁻¹.

    Stage 2 (Hunter) weights mid-level rotation higher to guide the agent toward
    pre-tornadic mesocyclone signatures before surface touchdown.
    """
    h, w = obs.shape[-2], obs.shape[-1]
    y0 = max(0, agent_y - radius_cells)
    y1 = min(h, agent_y + radius_cells + 1)
    x0 = max(0, agent_x - radius_cells)
    x1 = min(w, agent_x + radius_cells + 1)

    if y1 <= y0 or x1 <= x0:
        return 0.0

    ll_peak = float(np.nanmax(obs[_ROT_LL_60_IDX, y0:y1, x0:x1]))
    ml_peak = float(np.nanmax(obs[_ROT_ML_60_IDX, y0:y1, x0:x1]))

    ll_score = float(np.clip(ll_peak / _ROT_THRESHOLD, 0.0, 1.5))
    ml_score = float(np.clip(ml_peak / _ROT_THRESHOLD, 0.0, 1.5))

    # Stage 2: weight mid-level higher — the agent must "look up" for the
    # 3–6 km rotation that precedes surface tornado touchdown.
    w_ml = 0.7 if stage == 2 else 0.5
    w_ll = 0.3 if stage == 2 else 0.5

    return float(w_ll * ll_score + w_ml * ml_score)


def lifecycle_reward(
    pred_active: bool,
    dat_active: bool,
    stage: int,
) -> float:
    """
    Reward/penalty for lifecycle head accuracy.
    Returns +1 for correct, -1 × w_false_active for false positive in Stage 3.
    """
    if pred_active == dat_active:
        return 1.0
    if stage == 3 and pred_active and not dat_active:
        return -_W_FALSE  # Heavy cost-of-effort penalty
    return -0.5  # Mild penalty for other mismatches


def compute_reward(
    obs: np.ndarray,
    agent_y: int,
    agent_x: int,
    agent_radius: float,
    pred_active: bool,
    dat_track: LineString | None,
    dat_active: bool,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    stage: int,
) -> float:
    """
    Compute total step reward.

    Args:
        obs:           Current observation tensor (C, H, W) — latest timestep.
        agent_y/x:     Agent position in grid cells.
        agent_radius:  Predicted polygon radius in grid cells.
        pred_active:   Whether Lifecycle head predicts tornado on ground.
        dat_track:     Ground-truth DAT tornado track LineString or None.
        dat_active:    Whether DAT shows tornado on ground at this timestep.
        grid_lat/lon:  1D arrays of lat/lon for each grid cell.
        stage:         Curriculum stage (1, 2, or 3).

    Returns:
        Scalar reward float.
    """
    proximity = track_proximity_reward(agent_y, agent_x, dat_track, grid_lat, grid_lon)
    rot = rotation_anchor(obs, agent_y, agent_x, stage)
    life = lifecycle_reward(pred_active, dat_active, stage)

    reward = (
        _W_TRACK * proximity
        + _W_ROT * rot
        + _W_LIFE * life
    )
    return float(reward)
