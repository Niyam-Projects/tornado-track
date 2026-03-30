"""
Local inference script — run the trained model and output a GeoJSON.

Given a time range and bounding box, downloads/caches the needed MRMS files,
runs the Stage 3 model, and produces a GeoJSON containing:
  1. Primary Track:    LineString of the most likely (mean) path
  2. Swath Polygons:   Buffered 1σ and 2σ confidence interval areas
  3. Confidence Score: Per-timestep float (0.0–1.0) from Lifecycle head

Usage:
    python -m inference.predict \\
        --start "2023-05-06T22:00:00Z" \\
        --end   "2024-05-07T00:30:00Z" \\
        --bbox  "-99.5,35.0,-98.0,36.5" \\
        --output track_output.geojson
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from shapely.geometry import LineString, Point, mapping

from config import cfg
from data.grib_to_xarray import convert_event
from data.mrms_download import _buffer_bbox, download_event, _s3_client
from model.policy import TornadoPolicy

log = logging.getLogger(__name__)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOUCHDOWN_THRESHOLD = cfg.training.lifecycle.touchdown_threshold
_GRID_SIZE = cfg.zarr.grid_size
_N_SAMPLES = 200  # Trajectories to sample for confidence polygons


def _parse_datetime(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be: minlon,minlat,maxlon,maxlat")
    return tuple(parts)  # type: ignore


def _build_swath_polygon(positions: np.ndarray, radii: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray):
    """Build a Shapely polygon from a path of center positions and radii."""
    from shapely.ops import unary_union
    circles = []
    for i, (y, x) in enumerate(positions):
        iy = int(np.clip(round(y), 0, len(grid_lat) - 1))
        ix = int(np.clip(round(x), 0, len(grid_lon) - 1))
        lat = float(grid_lat[iy])
        lon = float(grid_lon[ix])
        r_deg = float(radii[i]) * abs(grid_lat[1] - grid_lat[0]) if len(grid_lat) > 1 else 0.02
        circles.append(Point(lon, lat).buffer(r_deg))
    return unary_union(circles) if circles else None


def run_inference(
    start_dt: datetime,
    end_dt: datetime,
    bbox: tuple[float, float, float, float],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Run model inference over the given time range and bounding box.

    Returns a GeoJSON-compatible dict.
    """
    # Download MRMS GRIB files for the requested window
    s3 = _s3_client()
    cache_dir = Path(cfg.data.root) / "mrms_cache"
    event_id = f"infer_{start_dt.strftime('%Y%m%d%H%M')}"
    buffered_bbox = _buffer_bbox(*bbox, km=cfg.mrms.spatial_buffer_km)

    log.info("Downloading MRMS for %s → %s over %s", start_dt, end_dt, bbox)
    grib_files = download_event(s3, event_id, start_dt, end_dt, buffered_bbox, cache_dir)

    # Convert to xarray
    import json as _json
    ds = convert_event(event_id, grib_files, buffered_bbox)
    if ds is None:
        raise RuntimeError("Could not load MRMS data for the requested window.")

    # Extract grid coords
    grid_lat = np.array(ds.coords.get("y", ds.coords.get("latitude", np.linspace(bbox[1], bbox[3], _GRID_SIZE))))
    grid_lon = np.array(ds.coords.get("x", ds.coords.get("longitude", np.linspace(bbox[0], bbox[2], _GRID_SIZE))))

    # Build observation sequence (C, T, H, W) → (T, C, H, W)
    data = ds.to_array(dim="channel").values.astype(np.float32)  # (C, T, H, W)
    data = np.nan_to_num(data, nan=0.0)
    obs_seq = torch.from_numpy(data).permute(1, 0, 2, 3)  # (T, C, H, W)

    # Load policy
    policy = TornadoPolicy().to(_DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=_DEVICE)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # Deterministic (mean) trajectory rollout
    mean_positions: list[tuple[float, float]] = []
    confidence_scores: list[float] = []
    radii: list[float] = []
    y, x = _GRID_SIZE / 2.0, _GRID_SIZE / 2.0
    r = 10.0
    lstm_h = lstm_c = None

    with torch.no_grad():
        for t in range(obs_seq.shape[0]):
            ob = obs_seq[t].unsqueeze(0).to(_DEVICE)
            lstm_state = (lstm_h, lstm_c) if lstm_h is not None else None
            action, _, _, _, aux = policy.get_action_and_value(ob, lstm_state=lstm_state)
            lstm_h, lstm_c = aux["lstm_h"], aux["lstm_c"]

            dx, dy, dr = action[0].tolist()
            x = float(np.clip(x + dx, 0, _GRID_SIZE - 1))
            y = float(np.clip(y + dy, 0, _GRID_SIZE - 1))
            r = float(np.clip(r + dr, 3, 40))

            lp = float(aux["lifecycle_prob"].squeeze())
            mean_positions.append((y, x))
            confidence_scores.append(lp)
            radii.append(r)

    # Stochastic trajectories for confidence polygons
    conf_data = policy.get_confidence_polygons(obs_seq.to(_DEVICE), n_samples=_N_SAMPLES)
    positions_arr = np.array(mean_positions)
    std_arr = np.array(conf_data["std_trajectory"])
    sigma_levels = conf_data["sigma_levels"]

    # Build GeoJSON features
    features = []

    # Primary track (LineString in lon/lat)
    def grid_to_latlon(pos_list):
        result = []
        for gy, gx in pos_list:
            iy = int(np.clip(round(gy), 0, len(grid_lat) - 1))
            ix = int(np.clip(round(gx), 0, len(grid_lon) - 1))
            result.append((float(grid_lon[ix]), float(grid_lat[iy])))
        return result

    track_coords = grid_to_latlon(mean_positions)
    if len(track_coords) >= 2:
        features.append({
            "type": "Feature",
            "geometry": mapping(LineString(track_coords)),
            "properties": {
                "type": "primary_track",
                "confidence": confidence_scores,
                "mean_confidence": float(np.mean(confidence_scores)),
            },
        })

    # Confidence swath polygons
    for i, sigma in enumerate(sigma_levels):
        sigma_radii = np.array(radii) * sigma
        swath = _build_swath_polygon(mean_positions, sigma_radii, grid_lat, grid_lon)
        if swath and not swath.is_empty:
            features.append({
                "type": "Feature",
                "geometry": mapping(swath),
                "properties": {
                    "type": f"confidence_swath_{sigma}sigma",
                    "sigma": sigma,
                },
            })

    # Touchdown / lift events
    active = [s > _TOUCHDOWN_THRESHOLD for s in confidence_scores]
    T = len(active)
    for t in range(T):
        if t > 0 and active[t] and not active[t - 1]:
            iy, ix = int(np.clip(round(mean_positions[t][0]), 0, len(grid_lat)-1)), \
                     int(np.clip(round(mean_positions[t][1]), 0, len(grid_lon)-1))
            features.append({
                "type": "Feature",
                "geometry": mapping(Point(float(grid_lon[ix]), float(grid_lat[iy]))),
                "properties": {"type": "touchdown", "timestep": t, "confidence": confidence_scores[t]},
            })
        if t > 0 and not active[t] and active[t - 1]:
            iy, ix = int(np.clip(round(mean_positions[t][0]), 0, len(grid_lat)-1)), \
                     int(np.clip(round(mean_positions[t][1]), 0, len(grid_lon)-1))
            features.append({
                "type": "Feature",
                "geometry": mapping(Point(float(grid_lon[ix]), float(grid_lat[iy]))),
                "properties": {"type": "lift", "timestep": t, "confidence": confidence_scores[t]},
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "bbox": bbox,
            "model_stage": int(ckpt.get("stage", 3)),
            "n_timesteps": T,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--start", required=True, help="Start datetime (ISO 8601, e.g. 2023-05-06T22:00:00Z)")
@click.option("--end", required=True, help="End datetime (ISO 8601)")
@click.option("--bbox", required=True, help="Bounding box: minlon,minlat,maxlon,maxlat")
@click.option("--output", default="track_output.geojson", show_default=True, help="Output GeoJSON path")
@click.option("--checkpoint", default=None, help="Model checkpoint path (default: stage3 final)")
def main(start: str, end: str, bbox: str, output: str, checkpoint: str | None) -> None:
    """Run tornado track inference and output a GeoJSON."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    bbox_tuple = _parse_bbox(bbox)

    ckpt_path = (
        Path(checkpoint)
        if checkpoint
        else Path(cfg.data.checkpoints_dir) / "stage3" / "checkpoint_final.pt"
    )
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        return

    result = run_inference(start_dt, end_dt, bbox_tuple, ckpt_path)

    output_path = Path(output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    n_feat = len(result["features"])
    log.info("Wrote %d features → %s", n_feat, output_path)


if __name__ == "__main__":
    main()
