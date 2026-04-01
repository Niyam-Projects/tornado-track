"""
Tornado Event Viewer — Streamlit local visualization app.

Visualizes downloaded MRMS Zarr events on an interactive Folium map, overlaid
with NOAA DAT track, EF damage polygons, and survey points for ground-truth
comparison.

Usage:
    uv run viz-events
    # Opens http://localhost:8501 in your browser

Features:
    - Event selector sorted by quality score (run scan-events first)
    - Timestep slider with frame-by-frame navigation
    - Per-layer toggles + opacity sliders for all 8 MRMS channels
    - Normalization range sliders (min/max clamp) per layer
    - Color scheme selector per layer
    - DAT track overlay (colored by EF rating)
    - DAT EF damage polygon overlay
    - DAT damage survey points overlay
    - Event metadata panel (EF rating, start/end time, quality score)
"""
from __future__ import annotations

import io
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import folium
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    import geopandas as gpd
    from shapely.geometry import mapping
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Config (lazy import to avoid side-effects on Streamlit reload)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_cfg():
    from config import cfg  # noqa: PLC0415
    return cfg


# ---------------------------------------------------------------------------
# EF color mapping
# ---------------------------------------------------------------------------
_EF_COLORS = {
    "EF0": "#00cc44",
    "EF1": "#ffee00",
    "EF2": "#ff9900",
    "EF3": "#ff3300",
    "EF4": "#cc00ff",
    "EF5": "#ff00ff",
    "Unknown": "#888888",
}

_CMAPS = [
    "viridis", "plasma", "inferno", "magma", "hot",
    "RdBu_r", "coolwarm", "Reds", "Blues",
    "YlOrRd", "YlGn", "turbo",
]

# alpha_threshold: abs(value) must exceed this to be drawn (overrides vmin mask).
# Used for signed/diverging channels (AzShear) and channels with noisy backgrounds.
_CHANNEL_DEFAULTS: dict[str, dict] = {
    "RotationTrack60min":   {"cmap": "plasma",  "vmin": 0.003, "vmax": 0.040, "visible": True,  "opacity": 0.90},
    "RotationTrack30min":   {"cmap": "plasma",  "vmin": 0.003, "vmax": 0.040, "visible": True,  "opacity": 0.75},
    "RotationTrackML60min": {"cmap": "hot",     "vmin": 0.003, "vmax": 0.035, "visible": True,  "opacity": 0.75},
    "RotationTrackML30min": {"cmap": "hot",     "vmin": 0.003, "vmax": 0.035, "visible": True,  "opacity": 0.70},
    "ReflectivityAtLowestAltitude": {"cmap": "turbo",   "vmin": 5.0,   "vmax": 65.0,  "visible": True,  "opacity": 0.55},
    "AzShear_0-2kmAGL":     {"cmap": "RdBu_r",  "vmin": -0.01, "vmax": 0.01,  "visible": True,  "opacity": 0.70,
                             "alpha_threshold": 0.0005},
    "AzShear_3-6kmAGL":     {"cmap": "RdBu_r",  "vmin": -0.01, "vmax": 0.01,  "visible": True,  "opacity": 0.70,
                             "alpha_threshold": 0.0005},
    "MESH":                  {"cmap": "YlOrRd",  "vmin": 6.0,   "vmax": 75.0,  "visible": True,  "opacity": 0.70},
}


# ---------------------------------------------------------------------------
# Data loaders (cached to avoid re-reading on every slider interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_index(index_path: str) -> pd.DataFrame:
    path = Path(index_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource
def _load_dat_tracks(dat_dir: str) -> Optional[object]:
    if not _HAS_GEO:
        return None
    path = Path(dat_dir) / "dat_tracks.parquet"
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gpd.read_parquet(path)


@st.cache_resource
def _load_dat_ef_polygons(dat_dir: str) -> Optional[object]:
    if not _HAS_GEO:
        return None
    path = Path(dat_dir) / "dat_ef_polygons.parquet"
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gpd.read_parquet(path)


@st.cache_resource
def _load_dat_damage_points(dat_dir: str) -> Optional[object]:
    if not _HAS_GEO:
        return None
    path = Path(dat_dir) / "dat_damage_points.parquet"
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gpd.read_parquet(path)


@st.cache_resource
def _open_zarr(zarr_path: str) -> Optional[xr.Dataset]:
    path = Path(zarr_path)
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return xr.open_zarr(path, consolidated=True)
        except Exception:
            try:
                return xr.open_zarr(path, consolidated=False)
            except Exception:
                return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _channel_to_png(
    data_2d: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    opacity: float,
    alpha_threshold: float | None = None,
) -> bytes:
    """Convert a 2-D float array to a PNG byte string using matplotlib colormap.

    Transparency rules:
    - If ``alpha_threshold`` is given (used for signed/diverging channels like AzShear):
        pixels where ``abs(arr) < alpha_threshold`` are fully transparent.
    - Otherwise: pixels where ``arr <= vmin`` are fully transparent, so only data
        above the display minimum is drawn (background noise is hidden).
    """
    if not _HAS_MPL:
        return b""

    arr = np.nan_to_num(data_2d, nan=0.0)
    rng = max(vmax - vmin, 1e-12)
    normalized = np.clip((arr - vmin) / rng, 0.0, 1.0)

    cmap = mpl.colormaps.get_cmap(cmap_name)
    rgba = cmap(normalized)                                    # (H, W, 4) float64

    if alpha_threshold is not None:
        mask = np.abs(arr) > alpha_threshold
    else:
        mask = arr > vmin   # hide anything at or below the display minimum

    rgba[..., 3] = np.where(mask, opacity, 0.0)
    img_arr = (rgba * 255).astype(np.uint8)

    img = Image.fromarray(img_arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _get_rotation_times(ds: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    """Return (time_values, labels) based on RotationTrack60min valid frames.

    Uses RotationTrack60min (or RotationTrack30min as fallback) to establish the
    master time axis, filtering out timesteps where that variable has no data.
    Falls back to the full dataset time coordinate if no rotation variable is present.
    """
    rot_var = next(
        (v for v in ["RotationTrack60min", "RotationTrack30min"] if v in ds.data_vars),
        None,
    )
    t_coord = ds.coords.get("time", ds.coords.get("t", None))
    if t_coord is None:
        return np.array([]), []

    all_times = np.array(t_coord.values)

    if rot_var is not None:
        # Compute max per timestep (chunked → memory-efficient, avoids loading full array).
        # Any frame whose max > 0 has real rotation data.
        rot_da = ds[rot_var]
        spatial_dims = [d for d in rot_da.dims if d not in ("time", "t")]
        rot_max = rot_da.max(dim=spatial_dims).compute().values  # (T,) float
        has_data = np.isfinite(rot_max) & (rot_max > 0)
        valid_idx = np.where(has_data)[0]
        if 0 < len(valid_idx) < len(all_times):
            all_times = all_times[valid_idx]

    labels = [str(t)[:19] for t in all_times]
    return all_times, labels


@st.cache_data(show_spinner=False)
def _get_rotation_times_for_zarr(zarr_path: str) -> tuple[np.ndarray, list[str]]:
    """Cached wrapper so rotation time index is computed once per zarr path."""
    ds = _open_zarr(zarr_path)
    if ds is None:
        return np.array([]), []
    return _get_rotation_times(ds)


def _get_bounds(ds: xr.Dataset) -> tuple[float, float, float, float]:
    lat = ds.coords.get("y", ds.coords.get("latitude", None))
    lon = ds.coords.get("x", ds.coords.get("longitude", None))
    if lat is None or lon is None:
        return 36.0, -98.0, 38.0, -95.0  # fallback: central US
    lat_arr = np.array(lat)
    lon_arr = np.array(lon)
    return float(lat_arr.min()), float(lon_arr.min()), float(lat_arr.max()), float(lon_arr.max())


def _ef_color(ef_value) -> str:
    """Map an EF-scale value to a hex color."""
    if ef_value is None or (isinstance(ef_value, float) and np.isnan(ef_value)):
        return _EF_COLORS["Unknown"]
    key = f"EF{int(ef_value)}" if isinstance(ef_value, (int, float)) else str(ef_value)
    return _EF_COLORS.get(key, _EF_COLORS["Unknown"])


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def _app() -> None:  # noqa: C901 – intentionally long UI function
    cfg = _load_cfg()

    st.set_page_config(
        page_title="🌪 Tornado Event Viewer",
        page_icon="🌪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Maximize map area — hide default header/footer chrome and reduce padding.
    # Use height-collapse instead of display:none so the sidebar toggle button
    # remains functional (display:none removes it from the DOM, leaving users
    # unable to re-open a collapsed sidebar).
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] {
            min-height: 0 !important;
            height: 0 !important;
            overflow: visible !important;
            padding: 0 !important;
        }
        [data-testid="stDecoration"] { display: none !important; }
        /* Keep sidebar open/close toggles always accessible */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
        [data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
        .block-container { padding-top: 0.4rem !important; padding-bottom: 0 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; padding-left: 8px; }
        .stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not _HAS_FOLIUM:
        st.error(
            "Missing dependencies: `folium` and `streamlit-folium` are required. "
            "Run `uv sync` to install them."
        )
        return

    # -----------------------------------------------------------------------
    # Load index
    # -----------------------------------------------------------------------
    index = _load_index(cfg.data.index_path)

    if index.empty:
        st.error(
            f"No events found. Expected `index.parquet` at `{cfg.data.index_path}`. "
            "Run `uv run zarr-build` to build the Zarr store."
        )
        return

    # Sort by quality score if available, otherwise alphabetically
    if "max_rotation_score" in index.columns and index["max_rotation_score"].notna().any():
        index_sorted = index.sort_values("max_rotation_score", ascending=False)
        sort_note = "sorted by rotation score ↓"
    else:
        index_sorted = index.sort_values("event_id")
        sort_note = "sorted A-Z  (run `scan-events` to score)"

    event_ids = index_sorted["event_id"].tolist()

    # -----------------------------------------------------------------------
    # Load DAT data
    # -----------------------------------------------------------------------
    dat_tracks      = _load_dat_tracks(cfg.data.dat_dir)
    dat_ef_polygons = _load_dat_ef_polygons(cfg.data.dat_dir)
    dat_damage_pts  = _load_dat_damage_points(cfg.data.dat_dir)

    # -----------------------------------------------------------------------
    # Sidebar — all controls live here
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🌪 Tornado Event Viewer")
        st.caption("MRMS Zarr + NOAA DAT ground truth")

        # ---- Event selector ----
        st.markdown("### 📂 Event")
        st.caption(sort_note)

        def _label(eid: str) -> str:
            row = index_sorted[index_sorted["event_id"] == eid]
            if row.empty:
                return eid
            r = row.iloc[0]
            tier  = r.get("rotation_tier", "")
            score = r.get("max_rotation_score")
            icon  = {"monster": "🔴", "moderate": "🟡", "weak": "⚪"}.get(str(tier), "")
            score_str = f"  [{score:.4f}]" if pd.notna(score) else ""
            return f"{icon} {eid}{score_str}"

        labels = {eid: _label(eid) for eid in event_ids}
        selected_event = st.selectbox(
            "Event",
            options=event_ids,
            format_func=lambda e: labels[e],
            label_visibility="collapsed",
        )

        # ---- Load event ----
        event_row = index_sorted[index_sorted["event_id"] == selected_event].iloc[0]
        zarr_path = str(event_row.get("zarr_path", ""))
        ds = _open_zarr(zarr_path) if zarr_path else None

        # ---- Metadata panel ----
        st.markdown("### ℹ️ Event Info")
        ef    = event_row.get("ef_rating", event_row.get("ef_scale", "N/A"))
        score = event_row.get("max_rotation_score")
        tier  = event_row.get("rotation_tier", "N/A")
        split = event_row.get("split", "N/A")
        start = str(event_row.get("start_time", ""))[:19]
        end   = str(event_row.get("end_time", ""))[:19]

        icon = {"monster": "🔴", "moderate": "🟡", "weak": "⚪"}.get(str(tier), "⬜")
        col1, col2 = st.columns(2)
        col1.metric("EF Rating", ef)
        col2.metric("Split", split)
        if pd.notna(score):
            col1.metric("Peak Rotation", f"{score:.5f}")
            col2.metric("Tier", f"{icon} {tier}")
        else:
            st.caption("Run `uv run scan-events` to score events")
        if start:
            st.caption(f"⏱ {start} → {end}")

        # ---- Timestep slider (driven by RotationTrack valid frames) ----
        st.markdown("### ⏩ Timestep")
        # Placeholder filled after the map renders so skipped-layer names are known
        _skipped_placeholder = st.empty()
        if ds is not None:
            master_times, t_labels = _get_rotation_times_for_zarr(zarr_path)
            n_timesteps = len(master_times)
            if n_timesteps == 0:
                # Fallback: use full time coord
                t_coord = ds.coords.get("time", ds.coords.get("t", None))
                master_times = np.array(t_coord.values) if t_coord is not None else np.array([])
                n_timesteps = len(master_times)
                t_labels = [str(t)[:19] for t in master_times]
            t_idx = st.slider(
                "Timestep",
                min_value=0,
                max_value=max(n_timesteps - 1, 0),
                value=0,
                format="%d",
                label_visibility="collapsed",
            )
            t_val = master_times[t_idx] if n_timesteps > 0 else None
            st.caption(f"🕒 {t_labels[t_idx] if t_idx < len(t_labels) else ''}  (frame {t_idx + 1}/{n_timesteps})")

            # ---- Timestamp diagnostics (collapsed by default) ----
            with st.expander("🔬 Timestamp Debug", expanded=False):
                if t_val is not None:
                    st.caption(f"**Rotation t_val:** `{str(t_val)[:23]}`")
                    t_val_pd = pd.Timestamp(t_val)
                    rows = []
                    for v in ds.data_vars:
                        da_d = ds[v]
                        t_d = next((d for d in da_d.dims if d in ("time", "t")), None)
                        if t_d is None:
                            rows.append({"var": v, "nearest_ts": "no time dim", "offset_s": ""})
                            continue
                        tc = da_d[t_d].values
                        if len(tc) == 0:
                            rows.append({"var": v, "nearest_ts": "empty", "offset_s": ""})
                            continue
                        deltas = np.abs(
                            pd.to_datetime(tc) - t_val_pd
                        ).total_seconds()
                        best_i = int(np.argmin(deltas))
                        rows.append({
                            "var": v,
                            "nearest_ts": str(tc[best_i])[:23],
                            "offset_s": f"{deltas[best_i]:.0f}s",
                        })
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            t_idx = 0
            t_val = None
            n_timesteps = 0
            master_times = np.array([])
            st.caption("⚠️ No Zarr store loaded")

        # ---- DAT overlays ----
        st.markdown("### 🗺 DAT Overlays")
        show_dat_track  = st.checkbox("DAT Track", value=True)
        show_dat_ef     = st.checkbox("EF Damage Polygons", value=True)
        show_dat_damage = st.checkbox("Damage Survey Points", value=False)

        # ---- MRMS layer controls ----
        st.markdown("### 📡 MRMS Layers")
        layer_settings: dict[str, dict] = {}
        for var, defaults in _CHANNEL_DEFAULTS.items():
            with st.expander(var, expanded=defaults["visible"]):
                vis     = st.checkbox("Visible", value=defaults["visible"], key=f"vis_{var}")
                cmap    = st.selectbox(
                    "Colormap", _CMAPS,
                    index=_CMAPS.index(defaults["cmap"]) if defaults["cmap"] in _CMAPS else 0,
                    key=f"cmap_{var}",
                )
                vmin    = st.number_input(
                    "Value min", value=float(defaults["vmin"]),
                    format="%.5f", key=f"vmin_{var}",
                )
                vmax    = st.number_input(
                    "Value max", value=float(defaults["vmax"]),
                    format="%.5f", key=f"vmax_{var}",
                )
                opacity = st.slider(
                    "Opacity", 0.0, 1.0, float(defaults["opacity"]),
                    step=0.05, key=f"opacity_{var}",
                )
                layer_settings[var] = {
                    "visible": vis, "cmap": cmap,
                    "vmin": vmin, "vmax": vmax, "opacity": opacity,
                }

    # -----------------------------------------------------------------------
    # Main area — tabs
    # -----------------------------------------------------------------------
    tab_map, tab_summary = st.tabs(["🗺  Map", "📊  Summary"])

    # ===================================================================
    # MAP TAB
    # ===================================================================
    with tab_map:
        if ds is None:
            st.error(
                f"Could not open Zarr store for `{selected_event}`. "
                f"Expected at: `{zarr_path}`"
            )
        else:
            # ---- Compute bounds ----
            lat_min, lon_min, lat_max, lon_max = _get_bounds(ds)
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            bounds = [[lat_min, lon_min], [lat_max, lon_max]]

            # ---- Build Folium map (cartodbpositron default) ----
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=9,
                tiles=None,
            )
            # Base tile layers — cartodbpositron first with show=True = visible default
            folium.TileLayer(
                "CartoDB positron",
                name="cartodbpositron",
                show=True,
            ).add_to(m)
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
            folium.TileLayer(
                tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
                attr="USGS Topo",
                name="USGS Topo",
                show=False,
            ).add_to(m)

            # ---- MRMS overlays ----
            # Rotation variables share the master time axis — use exact isel.
            # Non-rotation variables (Reflectivity, MESH, AzShear) can have large
            # timestamp offsets in the zarr (different MRMS processing pipelines).
            # Strategy: try nearest within 10 min first; if that fails, fall back to
            # unconditional nearest so a layer is never silently dropped just because
            # its timestamps are offset. Use the "🔬 Timestamp Debug" expander to
            # investigate actual offsets.
            _ROTATION_VARS = frozenset({
                "RotationTrack60min", "RotationTrack30min",
                "RotationTrackML60min", "RotationTrackML30min",
            })
            _NEAREST_TOLERANCE = pd.Timedelta("10min")
            _skipped_layers: list[str] = []

            for var in cfg.mrms.variables:
                settings = layer_settings.get(var, _CHANNEL_DEFAULTS.get(var, {}))
                if not settings.get("visible", False):
                    continue
                if var not in ds.data_vars:
                    continue
                try:
                    da = ds[var]
                    if t_val is None:
                        continue
                    t_dim = next((d for d in da.dims if d in ("time", "t")), None)
                    if t_dim is None:
                        arr = da.values
                    elif var in _ROTATION_VARS:
                        # Exact integer index — guaranteed to be on the master axis
                        arr = da.isel({t_dim: t_idx}).values
                    else:
                        # Nearest timestamp within 10-minute window; if nothing
                        # falls within the window, fall back to unconditional
                        # nearest so layers aren't silently dropped due to offset.
                        try:
                            arr = da.sel(
                                {t_dim: t_val},
                                method="nearest",
                                tolerance=_NEAREST_TOLERANCE,
                            ).values
                        except KeyError:
                            # Outside tolerance — use unconditional nearest and
                            # note the offset so the debug panel stays informative.
                            arr = da.sel({t_dim: t_val}, method="nearest").values
                            _skipped_layers.append(var)
                            log.debug("Layer %s exceeded 10 min offset at t=%s; using unconditional nearest", var, t_val)
                    if arr.ndim > 2:
                        arr = arr.squeeze()
                    if arr.ndim != 2:
                        continue

                    # Flip rows if latitude is ascending (south→north) so
                    # image row-0 = top of map (max latitude)
                    lat_coord = ds.coords.get("y", ds.coords.get("latitude", None))
                    if lat_coord is not None:
                        lat_vals = np.array(lat_coord)
                        if lat_vals[0] < lat_vals[-1]:  # ascending → flip
                            arr = arr[::-1, :]

                    png_bytes = _channel_to_png(
                        arr,
                        cmap_name=settings["cmap"],
                        vmin=settings["vmin"],
                        vmax=settings["vmax"],
                        opacity=settings["opacity"],
                        alpha_threshold=_CHANNEL_DEFAULTS.get(var, {}).get("alpha_threshold"),
                    )
                    img_str = (
                        "data:image/png;base64,"
                        + __import__("base64").b64encode(png_bytes).decode()
                    )
                    folium.raster_layers.ImageOverlay(
                        image=img_str,
                        bounds=bounds,
                        name=var,
                        opacity=1.0,   # alpha baked into PNG
                        interactive=False,
                        cross_origin=False,
                        zindex=10,
                    ).add_to(m)
                except Exception as exc:
                    log.warning("Could not render layer %s at t=%s: %s", var, t_val, exc)

            # Surface any layers that had timestamp offset > 10 min (used nearest fallback)
            if _skipped_layers:
                _skipped_placeholder.caption(
                    f"⚠️ Timestamp offset >10 min (nearest used): {', '.join(_skipped_layers)}"
                )

            # ---- DAT track ----
            if show_dat_track and dat_tracks is not None and _HAS_GEO:
                event_tracks = dat_tracks[dat_tracks["event_id"] == selected_event]
                for _, track_row in event_tracks.iterrows():
                    geom  = track_row.geometry
                    ef_v  = track_row.get("ef_rating", track_row.get("ef_scale", None))
                    color = _ef_color(ef_v)
                    if geom is not None and hasattr(geom, "coords"):
                        coords = [[c[1], c[0]] for c in geom.coords]
                        tooltip = (
                            f"DAT Track | EF: {ef_v} | "
                            f"Start: {str(track_row.get('start_time', ''))[:19]}"
                        )
                        folium.PolyLine(
                            coords, color=color, weight=4, opacity=0.9,
                            tooltip=tooltip,
                        ).add_to(m)
                        if coords:
                            folium.CircleMarker(
                                coords[0], radius=8, color=color, fill=True,
                                fill_opacity=0.9, tooltip="Touchdown",
                            ).add_to(m)
                            folium.CircleMarker(
                                coords[-1], radius=8, color="white", fill=True,
                                fill_color=color, fill_opacity=0.7, tooltip="Lift",
                            ).add_to(m)

            # ---- EF damage polygons ----
            if show_dat_ef and dat_ef_polygons is not None and _HAS_GEO:
                event_ef = dat_ef_polygons[dat_ef_polygons["event_id"] == selected_event]
                for _, ef_row in event_ef.iterrows():
                    geom  = ef_row.geometry
                    ef_v  = ef_row.get("ef_scale", ef_row.get("ef_rating", None))
                    color = _ef_color(ef_v)
                    if geom is not None:
                        try:
                            folium.GeoJson(
                                mapping(geom),
                                style_function=lambda _f, c=color: {
                                    "fillColor": c, "color": c,
                                    "weight": 1.5, "fillOpacity": 0.25,
                                },
                                tooltip=f"EF: {ef_v}",
                            ).add_to(m)
                        except Exception:
                            pass

            # ---- Damage survey points ----
            if show_dat_damage and dat_damage_pts is not None and _HAS_GEO:
                event_pts = dat_damage_pts[dat_damage_pts["event_id"] == selected_event]
                for _, pt_row in event_pts.iterrows():
                    geom  = pt_row.geometry
                    ef_v  = pt_row.get("ef_scale", None)
                    color = _ef_color(ef_v)
                    if geom is not None and geom.geom_type == "Point":
                        wind = pt_row.get("windspeed_mph", "N/A")
                        folium.CircleMarker(
                            [geom.y, geom.x], radius=5, color=color,
                            fill=True, fill_opacity=0.8,
                            tooltip=f"EF{ef_v} | {wind} mph",
                        ).add_to(m)

            # ---- EF legend (readable dark text) ----
            legend_items = "".join(
                f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:{c};border-radius:3px;flex-shrink:0"></span>'
                f'<span style="color:#111;font-size:12px;font-weight:500">{ef}</span></div>'
                for ef, c in _EF_COLORS.items() if ef != "Unknown"
            )
            legend_html = (
                '<div style="position:fixed;bottom:30px;right:10px;z-index:1000;'
                'background:rgba(255,255,255,0.95);padding:10px 14px;border-radius:8px;'
                'font-family:sans-serif;border:1px solid #ccc;box-shadow:0 2px 6px rgba(0,0,0,0.15)">'
                '<div style="color:#111;font-size:13px;font-weight:700;margin-bottom:6px">EF Scale</div>'
                + legend_items + "</div>"
            )
            m.get_root().html.add_child(folium.Element(legend_html))

            folium.LayerControl(collapsed=False).add_to(m)

            # ---- Render ----
            st_folium(m, width="100%", height=780, returned_objects=[])

    # ===================================================================
    # SUMMARY TAB
    # ===================================================================
    with tab_summary:
        st.markdown("### Current Event")

        # Per-event metric cards
        m_cols = st.columns(4)
        m_cols[0].metric("EF Rating", ef)
        m_cols[1].metric("Split", split)
        m_cols[2].metric("Timesteps", n_timesteps)
        if pd.notna(score):
            m_cols[3].metric("Peak Rotation (s⁻¹)", f"{score:.5f}")
            tier_label = (
                "🔴 Monster (Tier 1)" if str(tier) == "monster" else
                "🟡 Moderate (Tier 2)" if str(tier) == "moderate" else
                "⚪ Weak (Tier 3)" if str(tier) == "weak" else "—"
            )
            st.info(f"**Curriculum tier:** {tier_label}")
        else:
            st.warning("Run `uv run scan-events` to compute quality scores.")

        if start:
            st.caption(f"**Start:** {start}   →   **End:** {end}")

        # Tier distribution summary (if scan-events has been run)
        if "rotation_tier" in index_sorted.columns:
            st.markdown("### Dataset Tier Distribution")
            tier_counts = index_sorted["rotation_tier"].value_counts()
            dist_cols = st.columns(3)
            dist_cols[0].metric("🔴 Monster", int(tier_counts.get("monster", 0)))
            dist_cols[1].metric("🟡 Moderate", int(tier_counts.get("moderate", 0)))
            dist_cols[2].metric("⚪ Weak",     int(tier_counts.get("weak", 0)))

        # Full quality report table
        st.markdown("### All Events — Quality Report")
        display_cols = [
            c for c in [
                "event_id", "split", "ef_rating", "start_time", "end_time",
                "max_rotation_score", "mean_rotation_core", "active_pixel_count",
                "n_timesteps", "data_completeness", "rotation_tier", "curriculum_stage",
            ]
            if c in index_sorted.columns
        ]
        if not display_cols:
            display_cols = list(index_sorted.columns)

        st.dataframe(
            index_sorted[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# CLI entrypoint (wraps streamlit run)
# ---------------------------------------------------------------------------

@click.command()
@click.option("--port", default=8501, help="Port for Streamlit server (default: 8501).")
@click.option("--host", default="localhost", help="Host for Streamlit server.")
def main(port: int, host: str) -> None:
    """Launch the Tornado Event Viewer Streamlit app."""
    import subprocess
    import sys
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).resolve()),
            "--server.port", str(port),
            "--server.address", host,
            "--",  # pass no extra args to the app itself
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Streamlit entry (when called via `streamlit run viz/event_viewer.py`)
# ---------------------------------------------------------------------------
def _is_streamlit_context() -> bool:
    """Return True when this module is executing inside a Streamlit script runner."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" or _is_streamlit_context():
    _app()
