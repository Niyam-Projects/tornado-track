"""
Reward function for the TornadoTrackEnv.

Combines four components:
  1. polygon_iou     — spatial overlap of predicted swath vs DAT polygon
  2. rotation_anchor — mean RotationTrack value under the agent's position
  3. false_active    — penalty for claiming tornado is active when DAT shows none (Stage 3)
  4. lifecycle       — bonus for correct touchdown / lift detection timing
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon

from config import cfg

_W_IOU = cfg.training.reward.w_polygon_iou
_W_ROT = cfg.training.reward.w_rotation_anchor
_W_FALSE = cfg.training.reward.w_false_active
_W_LIFE = cfg.training.reward.w_lifecycle

# Channel indices (matches config variable order)
_ROTATION_TRACK_30MIN_IDX = 4   # RotationTrack30min
_ROTATION_TRACK_60MIN_IDX = 5   # RotationTrack60min


def polygon_iou(pred_polygon: Polygon, dat_polygon: Polygon | None) -> float:
    """Intersection-over-Union between predicted swath and DAT damage polygon."""
    if dat_polygon is None or pred_polygon is None:
        return 0.0
    if not (pred_polygon.is_valid and dat_polygon.is_valid):
        return 0.0
    intersection = pred_polygon.intersection(dat_polygon).area
    union = pred_polygon.union(dat_polygon).area
    if union == 0:
        return 0.0
    return float(intersection / union)


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
    dat_polygon: Polygon | None,
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
        dat_polygon:   Ground-truth DAT damage polygon (Shapely) or None.
        dat_active:    Whether DAT shows tornado on ground at this timestep.
        grid_lat/lon:  1D arrays of lat/lon for each grid cell.
        stage:         Curriculum stage (1, 2, or 3).

    Returns:
        Scalar reward float.
    """
    # Build predicted polygon (circle approximated as buffer in lon/lat space)
    pred_poly = None
    if 0 <= agent_y < len(grid_lat) and 0 <= agent_x < len(grid_lon):
        center = Point(grid_lon[agent_x], grid_lat[agent_y])
        radius_deg = agent_radius * abs(grid_lat[1] - grid_lat[0]) if len(grid_lat) > 1 else 0.01
        pred_poly = center.buffer(radius_deg)

    iou = polygon_iou(pred_poly, dat_polygon) if pred_poly else 0.0
    rot = rotation_anchor(obs, agent_y, agent_x)
    life = lifecycle_reward(pred_active, dat_active, stage)

    reward = (
        _W_IOU * iou
        + _W_ROT * rot
        + _W_LIFE * life
    )
    return float(reward)
