"""
Reward function for the TornadoTrackEnv.

Combines four components:
  1. track_proximity  — Gaussian reward for being close to the DAT tornado track line
  2. rotation_anchor  — mean RotationTrack value under the agent's position
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
_ROTATION_TRACK_30MIN_IDX = 4   # RotationTrack30min
_ROTATION_TRACK_60MIN_IDX = 5   # RotationTrack60min

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
    radius_cells: int = 5,
) -> float:
    """
    Mean of RotationTrack30min + RotationTrack60min in a small window around
    the agent's grid position. Normalizes to [0, 1] assuming max meaningful
    rotation ~0.01 s⁻¹ (MRMS azimuthal shear scale).
    """
    max_rotation = 0.01
    h, w = obs.shape[-2], obs.shape[-1]
    y0, y1 = max(0, agent_y - radius_cells), min(h, agent_y + radius_cells + 1)
    x0, x1 = max(0, agent_x - radius_cells), min(w, agent_x + radius_cells + 1)

    rot30 = obs[_ROTATION_TRACK_30MIN_IDX, y0:y1, x0:x1]
    rot60 = obs[_ROTATION_TRACK_60MIN_IDX, y0:y1, x0:x1]
    combined = (rot30 + rot60) / 2.0
    mean_val = float(np.nanmean(combined)) if combined.size > 0 else 0.0
    return float(np.clip(mean_val / max_rotation, 0.0, 1.0))


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
    rot = rotation_anchor(obs, agent_y, agent_x)
    life = lifecycle_reward(pred_active, dat_active, stage)

    reward = (
        _W_TRACK * proximity
        + _W_ROT * rot
        + _W_LIFE * life
    )
    return float(reward)
