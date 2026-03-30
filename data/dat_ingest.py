"""
NOAA DAT (Damage Assessment Toolkit) ingestion.

Downloads tornado events from the last N years and saves three GeoParquet files:
  - dat_tracks.parquet        : track polylines + derived start/end points + EF rating
  - dat_ef_polygons.parquet   : EF-rated damage polygons
  - dat_damage_points.parquet : individual damage survey points (ef_scale, windspeed_mph)

ArcGIS REST service:
  https://services.dat.noaa.gov/arcgis/rest/services/nws_damageassessmenttoolkit/DamageViewer/FeatureServer
  Layer 0: Damage Points SDE  (Point)
  Layer 1: Damage Lines SDE   (Polyline) ← tornado tracks
  Layer 2: Damage Polygons SDE (Polygon)

Usage:
    uv run python -m data.dat_ingest
    uv run python -m data.dat_ingest --years-back 3
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import httpx
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, shape
from tenacity import retry, stop_after_attempt, wait_exponential

from config import cfg

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAT ArcGIS REST service — corrected URL and layer IDs
# ---------------------------------------------------------------------------
_DAT_BASE = (
    "https://services.dat.noaa.gov/arcgis/rest/services"
    "/nws_damageassessmenttoolkit/DamageViewer/FeatureServer"
)
_LAYER_POINTS = 0    # Damage Points SDE  (survey points)
_LAYER_TRACKS = 1    # Damage Lines SDE   (tornado track polylines)
_LAYER_POLYGONS = 2  # Damage Polygons SDE

# Only EF-rated tornado events (excludes TSTM/Wind, N/A, UNKNOWN)
_EF_VALUES = "('EF0','EF1','EF2','EF3','EF3+','EF4','EF5')"

_OUT_FIELDS_TRACKS = (
    "event_id,stormdate,starttime,endtime,startlat,startlon,"
    "endlat,endlon,length,width,efscale,efnum,maxwind,wfo,injuries,fatalities"
)
# Layer 2 (Polygons) has no starttime/efnum/maxwind — uses stormdate
_OUT_FIELDS_POLYGONS = "event_id,stormdate,efscale,length,width,injuries,fatalities"
# Layer 0 (Points) has no starttime/efnum — windspeed is string, deaths not fatalities
_OUT_FIELDS_POINTS = "event_id,stormdate,efscale,windspeed,injuries,deaths,lat,lon"

_PAGE_SIZE = 1000   # stay well under the server max of 2000
_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Retry-wrapped HTTP GET
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(client: httpx.Client, url: str, params: dict[str, Any]) -> dict:
    resp = client.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Build WHERE clauses
# Layer 1 (Lines) has starttime; Layers 0 & 2 use stormdate instead
# ---------------------------------------------------------------------------
def _build_where_tracks(years_back: int) -> str:
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=365 * years_back)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return f"starttime >= timestamp '{cutoff}' AND efscale IN {_EF_VALUES}"


def _build_where_stormdate(years_back: int) -> str:
    """For layers 0 and 2 which use stormdate (not starttime)."""
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=365 * years_back)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return f"stormdate >= timestamp '{cutoff}' AND efscale IN {_EF_VALUES}"


# ---------------------------------------------------------------------------
# Core paginator
# ---------------------------------------------------------------------------
def _fetch_layer(
    client: httpx.Client,
    layer_id: int,
    where: str,
    out_fields: str,
) -> list[dict]:
    """Paginate through a FeatureServer layer and return all features as GeoJSON dicts."""
    url = f"{_DAT_BASE}/{layer_id}/query"
    base_params = {
        "where": where,
        "outFields": out_fields,
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": _PAGE_SIZE,
    }

    features: list[dict] = []
    offset = 0
    while True:
        data = _get(client, url, {**base_params, "resultOffset": offset})
        page = data.get("features", [])
        if not page:
            break
        features.extend(page)
        log.info(
            "Layer %d: +%d records (total %d)",
            layer_id, len(page), len(features),
        )
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    return features


# ---------------------------------------------------------------------------
# GeoDataFrame builders
# ---------------------------------------------------------------------------
def _ms_to_utc(ms: int | None) -> pd.Timestamp | None:
    """Convert epoch-millisecond timestamp to UTC Timestamp."""
    if ms is None:
        return None
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def _build_tracks_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for f in features:
        props = f.get("properties", {})
        geom_raw = f.get("geometry")
        if not geom_raw:
            continue
        geom = shape(geom_raw)

        # Flatten MultiLineString to LineString (take first / longest segment)
        if isinstance(geom, MultiLineString):
            geom = max(geom.geoms, key=lambda g: g.length)
        if not isinstance(geom, LineString) or geom.is_empty:
            continue

        coords = list(geom.coords)
        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])

        rows.append(
            {
                "event_id": props.get("event_id"),
                "ef_rating": props.get("efscale"),
                "ef_num": props.get("efnum"),
                "stormdate": _ms_to_utc(props.get("stormdate")),
                "start_time": _ms_to_utc(props.get("starttime")),
                "end_time": _ms_to_utc(props.get("endtime")),
                "start_lat": props.get("startlat"),
                "start_lon": props.get("startlon"),
                "end_lat": props.get("endlat"),
                "end_lon": props.get("endlon"),
                "length_mi": props.get("length"),
                "width_yd": props.get("width"),
                "max_wind_mph": props.get("maxwind"),
                "wfo": props.get("wfo"),
                "injuries": props.get("injuries"),
                "fatalities": props.get("fatalities"),
                # WKT for start/end points (parquet can't store multiple geometry cols)
                "start_point_wkt": start_pt.wkt,
                "end_point_wkt": end_pt.wkt,
                "geometry": geom,
            }
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["event_id", "ef_rating", "ef_num", "stormdate", "start_time", "end_time",
                     "start_lat", "start_lon", "end_lat", "end_lon", "length_mi", "width_yd",
                     "max_wind_mph", "wfo", "injuries", "fatalities",
                     "start_point_wkt", "end_point_wkt", "geometry"],
            geometry="geometry",
        ).set_crs(_CRS)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=_CRS)


def _build_ef_polygons_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for f in features:
        props = f.get("properties", {})
        geom_raw = f.get("geometry")
        if not geom_raw:
            continue
        geom = shape(geom_raw)
        if not isinstance(geom, (Polygon, MultiPolygon)) or geom.is_empty:
            continue
        rows.append(
            {
                "event_id": props.get("event_id"),
                "ef_scale": props.get("efscale"),
                "stormdate": _ms_to_utc(props.get("stormdate")),
                "length_mi": props.get("length"),
                "width_yd": props.get("width"),
                "injuries": props.get("injuries"),
                "fatalities": props.get("fatalities"),
                "area_m2": geom.area,
                "geometry": geom,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            columns=["event_id", "ef_scale", "stormdate", "length_mi", "width_yd",
                     "injuries", "fatalities", "area_m2", "geometry"],
            geometry="geometry",
        ).set_crs(_CRS)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=_CRS)


def _build_damage_points_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for f in features:
        props = f.get("properties", {})
        geom_raw = f.get("geometry")
        if not geom_raw:
            continue
        geom = shape(geom_raw)
        if not isinstance(geom, Point) or geom.is_empty:
            continue
        rows.append(
            {
                "event_id": props.get("event_id"),
                "ef_scale": props.get("efscale"),
                # windspeed is stored as a string on layer 0
                "windspeed": props.get("windspeed"),
                "stormdate": _ms_to_utc(props.get("stormdate")),
                "injuries": props.get("injuries"),
                "deaths": props.get("deaths"),
                "lat": props.get("lat") or geom.y,
                "lon": props.get("lon") or geom.x,
                "geometry": geom,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(
            columns=["event_id", "ef_scale", "windspeed", "stormdate",
                     "injuries", "deaths", "lat", "lon", "geometry"],
            geometry="geometry",
        ).set_crs(_CRS)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=_CRS)


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------
def ingest(years_back: int = 5, out_dir: Path | None = None) -> None:
    out_dir = out_dir or Path(cfg.data.dat_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    where_tracks = _build_where_tracks(years_back)
    where_stormdate = _build_where_stormdate(years_back)
    log.info("Track WHERE: %s", where_tracks)
    log.info("Stormdate WHERE: %s", where_stormdate)

    with httpx.Client(follow_redirects=True) as client:
        log.info("Fetching tornado tracks (layer %d — Damage Lines SDE)…", _LAYER_TRACKS)
        track_features = _fetch_layer(client, _LAYER_TRACKS, where_tracks, _OUT_FIELDS_TRACKS)

        log.info("Fetching EF polygons (layer %d — Damage Polygons SDE)…", _LAYER_POLYGONS)
        polygon_features = _fetch_layer(client, _LAYER_POLYGONS, where_stormdate, _OUT_FIELDS_POLYGONS)

        log.info("Fetching damage points (layer %d — Damage Points SDE)…", _LAYER_POINTS)
        point_features = _fetch_layer(client, _LAYER_POINTS, where_stormdate, _OUT_FIELDS_POINTS)

    log.info("Building GeoDataFrames…")
    tracks_gdf = _build_tracks_gdf(track_features)
    ef_gdf = _build_ef_polygons_gdf(polygon_features)
    damage_gdf = _build_damage_points_gdf(point_features)

    tracks_path = out_dir / "dat_tracks.parquet"
    ef_path = out_dir / "dat_ef_polygons.parquet"
    damage_path = out_dir / "dat_damage_points.parquet"

    tracks_gdf.to_parquet(tracks_path, index=False)
    ef_gdf.to_parquet(ef_path, index=False)
    damage_gdf.to_parquet(damage_path, index=False)

    log.info("✓ Saved %d tracks → %s", len(tracks_gdf), tracks_path)
    log.info("✓ Saved %d EF polygons → %s", len(ef_gdf), ef_path)
    log.info("✓ Saved %d damage points → %s", len(damage_gdf), damage_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--years-back", default=5, show_default=True, help="Years of history to fetch")
@click.option("--out-dir", default=None, help="Override output directory")
def main(years_back: int, out_dir: str | None) -> None:
    """Ingest NOAA DAT tornado data into GeoParquet files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest(years_back=years_back, out_dir=Path(out_dir) if out_dir else None)


if __name__ == "__main__":
    main()

