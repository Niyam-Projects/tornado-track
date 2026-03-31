"""
TornadoTrackEnv — Gymnasium environment for tornado path tracking RL.

Observation space: (C, H, W) float32 tensor — one timestep of the 8-channel
  MRMS feature cube centered on the agent's current position.

Action space: Box([dx, dy, dr]) where
  dx, dy : movement in grid cells (continuous, clipped to ±max_move)
  dr     : change in buffer radius in grid cells

The environment supports three spawn modes driven by the training stage:
  Stage 1 (Follower)  : spawn directly on the first point of the DAT track
  Stage 2 (Hunter)    : spawn 15 minutes (~8 MRMS timesteps) before touchdown
  Stage 3 (Surveyor)  : spawn at the very first timestep (clear-sky)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import gymnasium as gym
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from gymnasium import spaces
from shapely import wkt
from shapely.geometry import LineString, Point

from config import cfg
from model.reward import compute_reward

log = logging.getLogger(__name__)

# Hyperparameters
_GRID_SIZE = cfg.zarr.grid_size
_N_CHANNELS = len(cfg.mrms.variables)
_TOUCHDOWN_THRESHOLD = cfg.training.lifecycle.touchdown_threshold
_LIFT_THRESHOLD = cfg.training.lifecycle.lift_threshold
_MAX_MOVE = 10      # Max grid-cell movement per step
_INIT_RADIUS = 10   # Initial agent radius in grid cells
_MIN_RADIUS = 3
_MAX_RADIUS = 40
_PRE_TORNADO_STEPS = 8   # ~16 min before touchdown (2-min MRMS intervals)


class TornadoTrackEnv(gym.Env):
    """Gymnasium environment for 3-stage tornado track RL training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        index_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        stage: int = 1,
        split: str = "train",
    ):
        super().__init__()

        self.stage = stage
        self.split = split

        # Load event index
        idx_path = Path(index_path or cfg.data.index_path)
        self._index = pd.read_parquet(idx_path)
        self._index = self._index[self._index["split"] == split].reset_index(drop=True)

        # Load normalization stats
        stats_file = Path(stats_path or cfg.data.stats_path)
        with open(stats_file) as f:
            self._stats: dict = json.load(f)

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(_N_CHANNELS, _GRID_SIZE, _GRID_SIZE),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-_MAX_MOVE, -_MAX_MOVE, -5.0], dtype=np.float32),
            high=np.array([_MAX_MOVE, _MAX_MOVE, 5.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state (populated in reset)
        self._event: dict = {}
        self._data: np.ndarray | None = None   # (C, T, H, W)
        self._grid_lat: np.ndarray | None = None
        self._grid_lon: np.ndarray | None = None
        self._dat_track: LineString | None = None
        self._dat_polygon = None
        self._active_steps: list[bool] = []
        self._episode_step = 0
        self._t: int = 0
        self._max_t: int = 0
        self._agent_y: float = 0.0
        self._agent_x: float = 0.0
        self._agent_r: float = float(_INIT_RADIUS)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = self.np_random

        row = self._index.sample(1, random_state=rng.integers(0, 2**31)).iloc[0]
        self._event = row.to_dict()

        # Load Zarr data → numpy (C, T, H, W), channels ordered to match cfg.mrms.variables
        zarr_path = row["zarr_path"]
        ds = xr.open_zarr(zarr_path, consolidated=True)

        # xr.to_array() sorts alphabetically — reorder to config variable order
        zarr_vars = list(ds.data_vars)
        missing = [v for v in cfg.mrms.variables if v not in zarr_vars]
        if missing:
            log.warning("Event %s missing variables: %s", row.get("event_id"), missing)
        present = [v for v in cfg.mrms.variables if v in zarr_vars]
        data = ds[present].to_array(dim="channel").sel(channel=present).values  # (C, T, H, W)

        if len(present) < _N_CHANNELS:
            # Pad missing channels with zeros so observation shape stays constant
            pad = np.zeros((_N_CHANNELS - len(present), *data.shape[1:]), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)

        data = self._normalize(data)
        self._data = data.astype(np.float32)
        self._max_t = data.shape[1] - 1

        # Extract lat/lon grids
        self._grid_lat = ds.coords.get("y", ds.coords.get("latitude", None))
        self._grid_lat = np.array(self._grid_lat) if self._grid_lat is not None else np.linspace(0, 1, _GRID_SIZE)
        self._grid_lon = ds.coords.get("x", ds.coords.get("longitude", None))
        self._grid_lon = np.array(self._grid_lon) if self._grid_lon is not None else np.linspace(0, 1, _GRID_SIZE)

        # Extract zarr time coordinate for active-flag computation
        _zt = ds.coords.get("time", ds.coords.get("t", None))
        zarr_times = pd.DatetimeIndex(_zt.values) if _zt is not None else None

        # Load DAT track and derive active timestep flags
        self._dat_track, self._dat_polygon, self._active_steps = self._load_dat_info(row, zarr_times)

        # Spawn agent according to stage
        self._agent_y, self._agent_x = self._spawn_position()
        self._agent_r = float(_INIT_RADIUS)
        self._t = self._spawn_timestep()
        self._episode_step = 0

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        dx, dy, dr = float(action[0]), float(action[1]), float(action[2])

        self._episode_step += 1
        self._t = min(self._t + 1, self._max_t)

        obs = self._get_obs()

        lifecycle_prob = float(getattr(self, "_last_lifecycle_prob", 0.5))
        pred_active = lifecycle_prob > _TOUCHDOWN_THRESHOLD
        dat_active = self._active_steps[self._t] if self._t < len(self._active_steps) else False

        reward = compute_reward(
            obs=obs,
            agent_y=int(self._agent_y),
            agent_x=int(self._agent_x),
            agent_radius=self._agent_r,
            pred_active=pred_active,
            dat_track=self._dat_track,
            dat_active=dat_active,
            grid_lat=self._grid_lat,
            grid_lon=self._grid_lon,
            stage=self.stage,
        )

        terminated = self._t >= self._max_t
        if self.stage == 3 and pred_active and not dat_active and self._t > 0:
            if self._active_steps and not any(self._active_steps[: self._t]):
                terminated = True

        truncated = self._episode_step >= cfg.training.max_steps_per_episode

        info = {
            "t": self._t,
            "agent_y": self._agent_y,
            "agent_x": self._agent_x,
            "agent_r": self._agent_r,
            "dat_active": dat_active,
            "pred_active": pred_active,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Return (C, H, W) observation for current timestep."""
        t = min(self._t, self._max_t)
        obs = self._data[:, t, :, :]  # (C, H, W)
        obs = np.nan_to_num(obs, nan=0.0)
        return obs.astype(np.float32)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize each channel.

        Rotation channels use a clipped linear scaler so that the 0.01 s^-1
        threshold maps cleanly to 1.0 for the CNN -- preventing the strong core
        signal from being washed out by a high background standard deviation.

        All other channels use Z-score normalization via pre-computed stats.
        """
        for i, var in enumerate(cfg.mrms.variables):
            if "RotationTrack" in var:
                # Clipped linear: 0.01 s^-1 -> 1.0 (max brightness)
                data[i] = np.clip(data[i] / 0.01, 0.0, 1.0)
            elif var in self._stats:
                mean = self._stats[var]["mean"]
                std = max(self._stats[var]["std"], 1e-8)
                data[i] = (data[i] - mean) / std
            else:
                log.warning("No normalization stats for variable '%s' — channel %d left raw", var, i)
        return data

    def _spawn_position(self) -> tuple[float, float]:
        """Determine initial agent position based on curriculum stage."""
        center_y = _GRID_SIZE / 2.0
        center_x = _GRID_SIZE / 2.0

        if self._dat_track is None:
            return center_y, center_x

        if self.stage in (1, 2):
            # Spawn at or near the track start point (in grid space)
            start_coord = self._dat_track.coords[0]
            y, x = self._latlon_to_grid(start_coord[1], start_coord[0])
            noise = self.np_random.uniform(-2, 2, size=2) if self.stage == 2 else np.zeros(2)
            return float(np.clip(y + noise[0], 0, _GRID_SIZE - 1)), float(np.clip(x + noise[1], 0, _GRID_SIZE - 1))

        # Stage 3: random position in the grid
        return (
            float(self.np_random.uniform(0, _GRID_SIZE - 1)),
            float(self.np_random.uniform(0, _GRID_SIZE - 1)),
        )

    def _spawn_timestep(self) -> int:
        """Determine starting timestep based on curriculum stage."""
        if self.stage == 1:
            return 0  # Start right at tornado initiation
        if self.stage == 2:
            return max(0, self._active_steps.index(True) - _PRE_TORNADO_STEPS) if True in self._active_steps else 0
        return 0  # Stage 3: always start from t=0

    def _latlon_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert lat/lon to nearest grid indices."""
        if self._grid_lat is None or self._grid_lon is None:
            return _GRID_SIZE // 2, _GRID_SIZE // 2
        y = int(np.argmin(np.abs(self._grid_lat - lat)))
        x = int(np.argmin(np.abs(self._grid_lon - lon)))
        return y, x

    def _load_dat_info(
        self, row: pd.Series, zarr_times: pd.DatetimeIndex | None
    ) -> tuple[LineString | None, Any, list[bool]]:
        """Load DAT track and build per-timestep active flags from real DAT timestamps."""
        dat_dir = Path(cfg.data.dat_dir)
        track_path = dat_dir / "dat_tracks.parquet"
        ef_path = dat_dir / "dat_ef_polygons.parquet"

        track_geom = None
        ef_polygon = None
        dat_start: pd.Timestamp | None = None
        dat_end: pd.Timestamp | None = None

        event_id = row.get("event_id")
        try:
            tracks_gdf = gpd.read_parquet(track_path)
            match = tracks_gdf[tracks_gdf["event_id"] == event_id]
            if not match.empty:
                track_geom = match.iloc[0].geometry
                def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
                    return ts.tz_convert(None) if ts.tzinfo is not None else ts
                dat_start = _to_naive(pd.Timestamp(match.iloc[0]["start_time"]))
                dat_end = _to_naive(pd.Timestamp(match.iloc[0]["end_time"]))

            ef_gdf = gpd.read_parquet(ef_path)
            ef_match = ef_gdf[ef_gdf["event_id"] == event_id]
            if not ef_match.empty:
                ef_polygon = ef_match.geometry.unary_union
        except Exception as exc:
            log.warning("Could not load DAT info for event %s: %s", event_id, exc)

        # Build active flags from real DAT start/end times mapped to zarr timesteps
        n_steps = self._max_t + 1
        active = [False] * n_steps
        if zarr_times is not None and dat_start is not None and dat_end is not None:
            # Ensure timezone-naive comparison
            tz_times = zarr_times.tz_localize(None) if zarr_times.tz is not None else zarr_times
            for i, t in enumerate(tz_times):
                if i >= n_steps:
                    break
                active[i] = bool(dat_start <= t <= dat_end)
        elif dat_start is not None:
            log.warning("zarr time coordinate unavailable for event %s — active flags unset", event_id)

        return track_geom, ef_polygon, active

    def set_lifecycle_prob(self, prob: float) -> None:
        """Called by the policy to inject lifecycle probability before step()."""
        self._last_lifecycle_prob = prob
