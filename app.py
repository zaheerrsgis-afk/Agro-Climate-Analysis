import base64
import io
import json
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import streamlit.components.v1 as components
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling


# =====================================================
# Agro Climate Analysis
# Final Streamlit app.py
#
# IMPORTANT:
# 1. This version does NOT use streamlit-folium.
# 2. Rasters are read locally from GitHub:
#    data/rasters/
# 3. Boundary is read from:
#    data/vectors/Punjab_Districts.geojson
# 4. Scorecard is read from:
#    data/tables/Punjab_District_Risk_Scorecard.csv
# =====================================================


# =====================================================
# App Configuration
# =====================================================

st.set_page_config(
    page_title="Agro Climate Analysis",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RASTER_DIR = DATA_DIR / "rasters"
TABLE_DIR = DATA_DIR / "tables"
VECTOR_DIR = DATA_DIR / "vectors"

SCORECARD_FILE = TABLE_DIR / "Punjab_District_Risk_Scorecard.csv"
DISTRICTS_FILE = VECTOR_DIR / "Punjab_Districts.geojson"


# =====================================================
# Layer Configuration
# =====================================================

STANDARD_LEGEND = {
    1: ("Low", "#1a9850"),
    2: ("Moderate", "#fee08b"),
    3: ("High", "#f46d43"),
    4: ("Very High", "#a50026"),
}

LAYER_CONFIG = {
    "Environmental Stress Hotspots": {
        "file": "Punjab_Environmental_Stress_Hotspots.tif",
        "score_col": "Environmental_Stress_Score",
        "risk_col": "Overall_Risk",
        "description": "Overall agro-climate stress combining water, drought, crop, heat and soil moisture indicators.",
    },
    "Drought-Prone Areas": {
        "file": "Punjab_Drought_Prone_Areas.tif",
        "score_col": "Drought_Prone_Score",
        "risk_col": "Drought_Tendency",
        "description": "Areas repeatedly exposed to drought-like conditions.",
    },
    "Water Storage Stress": {
        "file": "Punjab_Water_Storage_Stress.tif",
        "score_col": "Water_Storage_Stress_Score",
        "risk_col": "Water_Storage_Stress",
        "description": "Regional water storage stress based on GRACE/GRACE-FO total water storage anomaly.",
    },
    "Crop and Vegetation Stress": {
        "file": "Punjab_Crop_Vegetation_Stress.tif",
        "score_col": "Crop_Vegetation_Stress_Score",
        "risk_col": "Crop_Vegetation_Stress",
        "description": "Areas repeatedly showing below-normal crop and vegetation condition.",
    },
    "Heat Stress Hotspots": {
        "file": "Punjab_Heat_Stress_Hotspots.tif",
        "score_col": "Heat_Stress_Score",
        "risk_col": "Heat_Stress",
        "description": "Areas repeatedly exposed to higher land surface temperature conditions.",
    },
    "Rainfall Deficit Zones": {
        "file": "Punjab_Rainfall_Deficit_Zones.tif",
        "score_col": "Rainfall_Deficit_Score",
        "risk_col": "Rainfall_Deficit_Tendency",
        "description": "Areas repeatedly receiving below-normal rainfall.",
    },
    "Soil Moisture Stress": {
        "file": "Punjab_Soil_Moisture_Stress.tif",
        "score_col": "Soil_Moisture_Stress_Score",
        "risk_col": "Soil_Moisture_Stress",
        "description": "Areas repeatedly showing below-normal soil moisture condition.",
    },
}


# =====================================================
# CSS
# =====================================================

st.markdown(
    """
<style>
.stApp {
    background: #f5f7fb;
}

.block-container {
    padding-top: 0.75rem;
    padding-bottom: 1.5rem;
    max-width: 1800px;
}

.panel {
    background: #ffffff;
    border: 1px solid #e7eaf0;
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    padding: 14px;
    margin-bottom: 14px;
}

.step-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #166534;
    font-weight: 800;
    font-size: 1.05rem;
    margin-bottom: 12px;
}

.step-badge {
    width: 22px;
    height: 22px;
    background: linear-gradient(135deg, #22c55e, #86efac);
    color: white;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 900;
}

.tab-row {
    display: flex;
    gap: 18px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 12px;
}

.tab-active {
    color: #15803d;
    border-bottom: 3px solid #22c55e;
    padding-bottom: 9px;
    font-weight: 800;
}

.tab-muted {
    color: #6b7280;
    padding-bottom: 9px;
    font-weight: 600;
}

.small-help {
    color: #9ca3af;
    font-size: 0.82rem;
    margin-top: 4px;
}

.module-card {
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 12px;
    background: #f0fdf4;
}

.module-title {
    font-weight: 800;
    color: #1f2937;
}

.module-sub {
    color: #9ca3af;
    font-size: 0.82rem;
}

.map-card {
    background: #ffffff;
    border: 1px solid #86efac;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(34,197,94,0.12);
    padding: 10px;
}

.map-header {
    margin-bottom: 8px;
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.pill {
    background: rgba(255,255,255,0.95);
    border: 1px solid #e5e7eb;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.82rem;
    color: #374151;
    font-weight: 700;
}

.result-title {
    color: #166534;
    font-weight: 900;
    font-size: 1.05rem;
    margin-bottom: 10px;
}

.alert-box {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 12px;
    color: #92400e;
    font-size: 0.92rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.metric-mini {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px;
}

.metric-mini .label {
    color: #6b7280;
    font-size: 0.78rem;
}

.metric-mini .value {
    color: #111827;
    font-size: 1.12rem;
    font-weight: 900;
}

.legend-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 12px;
}

.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 8px 0;
    font-size: 0.92rem;
    color: #374151;
}

.legend-swatch {
    width: 22px;
    height: 16px;
    border-radius: 5px;
    border: 1px solid rgba(0,0,0,0.15);
}

.stSelectbox label,
.stSlider label,
.stCheckbox label {
    color: #374151 !important;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True
)


# =====================================================
# Basic Utilities
# =====================================================

def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


@st.cache_data(show_spinner=False)
def load_scorecard(path: str) -> pd.DataFrame:
    p = Path(path)
    if not file_exists(p):
        return pd.DataFrame()

    df = pd.read_csv(p)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, regex=True)]
    return df


@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    p = Path(path)
    if not file_exists(p):
        return None

    with open(p, "r", encoding="utf-8") as f:
        gj = json.load(f)

    if gj.get("type") != "FeatureCollection":
        return gj

    clean_features = []

    for feature in gj.get("features", []):
        geom = feature.get("geometry")
        props = feature.get("properties", {})

        if not isinstance(geom, dict):
            continue

        geom_type = geom.get("type")

        if geom_type in ["Polygon", "MultiPolygon"] and geom.get("coordinates"):
            clean_features.append({
                "type": "Feature",
                "properties": props,
                "geometry": geom
            })

    gj["features"] = clean_features
    return gj


def get_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def district_field(gj):
    if not gj or not gj.get("features"):
        return None

    props = gj["features"][0].get("properties", {})

    for field in ["DISTRICT", "District_Name", "District", "DIST_NAME", "NAME", "Name"]:
        if field in props:
            return field

    return None


def get_district_feature(gj, district_name):
    if not gj or district_name == "All Punjab":
        return None

    field = district_field(gj)

    if not field:
        return None

    for feature in gj.get("features", []):
        if str(feature.get("properties", {}).get(field, "")) == str(district_name):
            return feature

    return None


def get_shapes_for_aoi(gj, district_name):
    selected_feature = get_district_feature(gj, district_name)

    if selected_feature:
        return [selected_feature["geometry"]], selected_feature

    if gj and gj.get("features"):
        shapes = [feature["geometry"] for feature in gj.get("features", []) if feature.get("geometry")]
        return shapes, None

    return None, None


def extract_coords(geom):
    coords = []

    def walk(obj):
        if isinstance(obj, list):
            if len(obj) >= 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                coords.append((float(obj[0]), float(obj[1])))
            else:
                for item in obj:
                    walk(item)

    if geom and "coordinates" in geom:
        walk(geom["coordinates"])

    return coords


def feature_bounds(feature):
    if not feature:
        return None

    coords = extract_coords(feature.get("geometry", {}))

    if not coords:
        return None

    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]

    return [[min(ys), min(xs)], [max(ys), max(xs)]]


def single_feature_geojson(feature):
    if not feature:
        return None

    return {
        "type": "FeatureCollection",
        "features": [feature]
    }


# =====================================================
# Dynamic Statistics
# =====================================================

def label_from_score(value):
    if pd.isna(value):
        return "Unknown"
    if value <= 0.25:
        return "Low"
    if value <= 0.50:
        return "Moderate"
    if value <= 0.75:
        return "High"
    return "Very High"


def get_layer_labels(df: pd.DataFrame, layer_name: str, balanced: bool = True):
    if df.empty:
        return pd.Series(dtype=str)

    cfg = LAYER_CONFIG[layer_name]
    risk_col = cfg["risk_col"]
    score_col = cfg["score_col"]

    if balanced and score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce")

        if scores.notna().sum() >= 4:
            q1, q2, q3 = scores.quantile([0.25, 0.50, 0.75]).values

            def q_label(v):
                if pd.isna(v):
                    return "Unknown"
                if v <= q1:
                    return "Low"
                if v <= q2:
                    return "Moderate"
                if v <= q3:
                    return "High"
                return "Very High"

            return scores.apply(q_label)

    if risk_col in df.columns:
        return df[risk_col].astype(str)

    if score_col in df.columns:
        return pd.to_numeric(df[score_col], errors="coerce").apply(label_from_score)

    return pd.Series(["Unknown"] * len(df), index=df.index)


def get_layer_stats(df: pd.DataFrame, layer_name: str, selected_district: str, balanced: bool = True):
    if df.empty:
        return {
            "high": "—",
            "very_high": "—",
            "top": "—",
            "selected": "—"
        }

    district_col = get_col(df, ["District_Name", "DISTRICT", "District", "DIST_NAME"])
    score_col = LAYER_CONFIG[layer_name]["score_col"]

    work = df.copy()
    work["_layer_risk"] = get_layer_labels(work, layer_name, balanced)

    high_count = int(work["_layer_risk"].str.lower().eq("high").sum())
    very_high_count = int(work["_layer_risk"].str.lower().eq("very high").sum())

    top_district = "—"

    if score_col in work.columns and district_col:
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        ranked = work.sort_values(score_col, ascending=False)

        if len(ranked) > 0:
            top_district = str(ranked.iloc[0][district_col])

    selected_risk = "All Punjab"

    if selected_district != "All Punjab" and district_col:
        selected_row = work[work[district_col].astype(str) == str(selected_district)]

        if len(selected_row) > 0:
            selected_risk = str(selected_row.iloc[0]["_layer_risk"])

    return {
        "high": high_count,
        "very_high": very_high_count,
        "top": top_district,
        "selected": selected_risk
    }


# =====================================================
# Raster Processing
# =====================================================

def class_to_rgba(class_array, alpha=185):
    rgba = np.zeros((class_array.shape[0], class_array.shape[1], 4), dtype=np.uint8)

    for value, (_, hex_color) in STANDARD_LEGEND.items():
        h = hex_color.replace("#", "")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        rgba[class_array == value] = (r, g, b, alpha)

    rgba[~np.isin(class_array, [1, 2, 3, 4])] = (0, 0, 0, 0)

    return rgba


def balanced_visual_classes(data):
    valid = data[np.isfinite(data)]
    valid = valid[valid > -9999]

    if valid.size < 10:
        out = np.zeros(data.shape, dtype=np.int16)
        out[np.isfinite(data)] = 1
        return out

    q1, q2, q3 = np.nanpercentile(valid, [25, 50, 75])

    if q1 == q2 or q2 == q3:
        mn, mx = float(np.nanmin(valid)), float(np.nanmax(valid))

        if mn == mx:
            out = np.zeros(data.shape, dtype=np.int16)
            out[np.isfinite(data)] = 1
            return out

        q1 = mn + (mx - mn) * 0.25
        q2 = mn + (mx - mn) * 0.50
        q3 = mn + (mx - mn) * 0.75

    out = np.zeros(data.shape, dtype=np.int16)
    out[(data <= q1) & np.isfinite(data)] = 1
    out[(data > q1) & (data <= q2)] = 2
    out[(data > q2) & (data <= q3)] = 3
    out[data > q3] = 4

    return out


def crop_to_valid_extent(data, transform):
    valid = np.isfinite(data)

    if not np.any(valid):
        return data, transform

    rows, cols = np.where(valid)

    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()

    cropped = data[r0:r1 + 1, c0:c1 + 1]
    new_transform = transform * rasterio.Affine.translation(c0, r0)

    return cropped, new_transform


@st.cache_data(show_spinner=False)
def raster_to_overlay_payload(raster_path: str, shapes_json, balanced: bool, max_size: int = 1200):
    path = Path(raster_path)

    if not file_exists(path):
        return None, f"Missing raster: {path}"

    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs

            if src.nodata is not None:
                data[data == src.nodata] = np.nan

            if crs and crs.to_string() != "EPSG:4326":
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    crs,
                    "EPSG:4326",
                    src.width,
                    src.height,
                    *src.bounds
                )

                dst = np.empty((dst_height, dst_width), dtype=np.float32)

                reproject(
                    source=data,
                    destination=dst,
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest
                )

                data = dst
                transform = dst_transform

            if shapes_json:
                mask = geometry_mask(
                    shapes_json,
                    out_shape=data.shape,
                    transform=transform,
                    invert=True
                )

                data = np.where(mask, data, np.nan)

            data, transform = crop_to_valid_extent(data, transform)

            h, w = data.shape
            factor = max(1, int(max(h, w) / max_size))

            if factor > 1:
                data = data[::factor, ::factor]
                transform = transform * rasterio.Affine.scale(factor, factor)

            if balanced:
                class_array = balanced_visual_classes(data)
            else:
                class_array = np.rint(data).astype(np.int16)
                class_array[~np.isfinite(data)] = 0

            rgba = class_to_rgba(class_array)

            buffer = io.BytesIO()
            plt.imsave(buffer, rgba)
            buffer.seek(0)

            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            west, south, east, north = rasterio.transform.array_bounds(
                data.shape[0],
                data.shape[1],
                transform
            )

            return {
                "image": f"data:image/png;base64,{encoded}",
                "bounds": [[south, west], [north, east]]
            }, None

    except Exception as exc:
        return None, f"Raster read/display error: {exc}"


# =====================================================
# Map and Legend
# =====================================================

def make_map(layer_name, district_name, boundary_geojson, opacity, balanced):
    cfg = LAYER_CONFIG[layer_name]
    raster_path = RASTER_DIR / cfg["file"]
    shapes, selected_feature = get_shapes_for_aoi(boundary_geojson, district_name)

    m = folium.Map(
        location=[31.2, 72.9],
        zoom_start=7,
        tiles=None,
        control_scale=True
    )

    folium.TileLayer(
        "CartoDB positron",
        name="Light map",
        control=True
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True
    ).add_to(m)

    overlay, error_message = raster_to_overlay_payload(
        str(raster_path),
        shapes,
        balanced
    )

    if overlay:
        folium.raster_layers.ImageOverlay(
            image=overlay["image"],
            bounds=overlay["bounds"],
            opacity=opacity,
            name=layer_name,
            interactive=True,
            zindex=2
        ).add_to(m)
    else:
        folium.Marker(
            [31.2, 72.9],
            tooltip="Raster not displayed",
            popup=error_message or "Raster not displayed"
        ).add_to(m)

    if boundary_geojson and boundary_geojson.get("features"):
        field = district_field(boundary_geojson)
        tooltip = folium.GeoJsonTooltip(fields=[field], aliases=["District"], sticky=True) if field else None

        if district_name == "All Punjab":
            folium.GeoJson(
                boundary_geojson,
                name="Punjab districts",
                style_function=lambda feature: {
                    "fillColor": "transparent",
                    "color": "#0f8f6a",
                    "weight": 1.0,
                    "fillOpacity": 0,
                    "opacity": 0.75
                },
                tooltip=tooltip
            ).add_to(m)

    if selected_feature:
        selected_gj = single_feature_geojson(selected_feature)
        field = district_field(selected_gj)
        tooltip = folium.GeoJsonTooltip(fields=[field], aliases=["District"], sticky=True) if field else None

        folium.GeoJson(
            selected_gj,
            name=f"Selected district: {district_name}",
            style_function=lambda feature: {
                "fillColor": "transparent",
                "color": "#16a34a",
                "weight": 3,
                "fillOpacity": 0,
                "opacity": 1
            },
            tooltip=tooltip
        ).add_to(m)

        bounds = feature_bounds(selected_feature)

        if bounds:
            m.fit_bounds(bounds)

    folium.LayerControl(collapsed=True).add_to(m)

    return m, cfg["file"], error_message


def render_legend():
    rows = ""

    for _, (label, color) in STANDARD_LEGEND.items():
        rows += f"""
        <div class="legend-row">
            <div class="legend-swatch" style="background:{color};"></div>
            <span>{label}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="legend-box">
            <b>Stress level</b>
            {rows}
        </div>
        """,
        unsafe_allow_html=True
    )


def raster_diagnostics_table():
    rows = []

    for layer_name, cfg in LAYER_CONFIG.items():
        raster_path = RASTER_DIR / cfg["file"]

        row = {
            "Layer": layer_name,
            "File": cfg["file"],
            "Exists": raster_path.exists(),
            "Size_KB": round(raster_path.stat().st_size / 1024, 2) if raster_path.exists() else None,
            "Min": None,
            "Max": None,
            "Values_sample": None
        }

        if raster_path.exists():
            try:
                with rasterio.open(raster_path) as src:
                    arr = src.read(1).astype(float)

                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan

                    valid = arr[np.isfinite(arr)]

                    if valid.size:
                        row["Min"] = float(np.nanmin(valid))
                        row["Max"] = float(np.nanmax(valid))
                        values = np.unique(valid)
                        row["Values_sample"] = ", ".join([str(v) for v in values[:12]])
                    else:
                        row["Values_sample"] = "No valid pixels"

            except Exception as exc:
                row["Values_sample"] = f"Read error: {exc}"

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================
# Load Data
# =====================================================

scorecard = load_scorecard(str(SCORECARD_FILE))
boundary_geojson = load_geojson(str(DISTRICTS_FILE))

district_col = get_col(scorecard, ["District_Name", "DISTRICT", "District", "DIST_NAME"]) if not scorecard.empty else None

district_names = ["All Punjab"]

if district_col:
    district_names += sorted(scorecard[district_col].dropna().astype(str).unique().tolist())
elif boundary_geojson:
    field = district_field(boundary_geojson)

    if field:
        district_names += sorted([
            str(feature["properties"].get(field))
            for feature in boundary_geojson.get("features", [])
            if feature.get("properties", {}).get(field) is not None
        ])


# =====================================================
# Header
# =====================================================

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
        <div>
            <h2 style="margin:0; color:#166534; font-size:2rem; font-weight:850;">Agro Climate Analysis</h2>
            <div style="color:#6b7280; font-size:0.95rem;">
                Satellite-based agro-climate stress monitoring and district-level risk analysis
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Main Layout
# =====================================================

left_col, map_col, right_col = st.columns([1.05, 3.15, 1.1], gap="medium")


# =====================================================
# Left Controls
# =====================================================

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">1</span> Area of Interest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tab-row"><span class="tab-active">District</span><span class="tab-muted">GeoJSON</span><span class="tab-muted">Shapefile</span><span class="tab-muted">Raster</span></div>',
        unsafe_allow_html=True
    )

    selected_district = st.selectbox(
        "Select district / AOI",
        district_names,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="small-help">Boundary source: GeoJSON. Choose All Punjab or zoom to one district.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">2</span> Satellite Indicators</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
          <div class="module-card"><div class="module-title">CHIRPS</div><div class="module-sub">Rainfall</div></div>
          <div class="module-card"><div class="module-title">Sentinel-2</div><div class="module-sub">NDVI</div></div>
          <div class="module-card"><div class="module-title">MODIS</div><div class="module-sub">LST</div></div>
          <div class="module-card"><div class="module-title">GLDAS</div><div class="module-sub">Soil moisture</div></div>
          <div class="module-card"><div class="module-title">GRACE-FO</div><div class="module-sub">Water storage</div></div>
          <div class="module-card"><div class="module-title">GNSS</div><div class="module-sub">GPS / Baidu</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">3</span> Analysis Module</div>', unsafe_allow_html=True)

    selected_layer = st.selectbox(
        "Select analysis layer",
        list(LAYER_CONFIG.keys()),
        label_visibility="collapsed"
    )

    balanced_visual = st.checkbox("Balanced visual classes", value=True)

    opacity = st.slider(
        "Layer opacity",
        0.10,
        1.00,
        0.68,
        0.05
    )

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# Dynamic Stats
# =====================================================

stats = get_layer_stats(scorecard, selected_layer, selected_district, balanced_visual)


# =====================================================
# Map Panel
# =====================================================

with map_col:
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="map-header">
            <span class="pill">AOI: {selected_district}</span>
            <span class="pill">Layer: {selected_layer}</span>
            <span class="pill">Baseline: 2018–2024</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    folium_map, raster_name, raster_error = make_map(
        selected_layer,
        selected_district,
        boundary_geojson,
        opacity,
        balanced_visual
    )

    components.html(
        folium_map.get_root().render(),
        height=650,
        scrolling=False
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if raster_error:
        st.error(raster_error)

    st.caption(f"Raster displayed: {raster_name}. Maps are clipped to Punjab or selected district.")


# =====================================================
# Right Panel
# =====================================================

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Active AOI</div>', unsafe_allow_html=True)

    if selected_district == "All Punjab":
        st.markdown(
            '<div class="alert-box"><b>All Punjab selected.</b><br>Select a district to inspect local risk and download a district-wise profile.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="alert-box"><b>{selected_district}</b><br>District-specific raster and statistics are active.</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Result & Legend</div>', unsafe_allow_html=True)

    render_legend()

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">High districts</div><div class="value">{stats["high"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">Very high districts</div><div class="value">{stats["very_high"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">Top district</div><div class="value">{stats["top"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-mini"><div class="label">Selected risk</div><div class="value">{stats["selected"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Export & Reports</div>', unsafe_allow_html=True)

    if not scorecard.empty and district_col and selected_district != "All Punjab":
        row = scorecard[scorecard[district_col].astype(str) == str(selected_district)]
        st.download_button(
            "Download District Profile",
            data=row.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_district.replace(' ', '_')}_Risk_Profile.csv",
            mime="text/csv",
            use_container_width=True
        )
    elif file_exists(SCORECARD_FILE):
        with open(SCORECARD_FILE, "rb") as f:
            st.download_button(
                "Download Scorecard",
                f,
                file_name=SCORECARD_FILE.name,
                mime="text/csv",
                use_container_width=True
            )

    raster_path = RASTER_DIR / LAYER_CONFIG[selected_layer]["file"]

    if file_exists(raster_path):
        with open(raster_path, "rb") as f:
            st.download_button(
                "Download Result Raster",
                f,
                file_name=raster_path.name,
                mime="image/tiff",
                use_container_width=True
            )

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# Bottom Tabs
# =====================================================

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "District Scorecard",
    "District Profile",
    "Methodology",
    "Raster Diagnostics"
])


with tab1:
    st.subheader(f"District Scorecard — {selected_layer}")

    if scorecard.empty:
        st.warning("Scorecard CSV not found.")
    else:
        cfg = LAYER_CONFIG[selected_layer]
        work = scorecard.copy()

        if selected_district != "All Punjab" and district_col:
            work = work[work[district_col].astype(str) == str(selected_district)]

        work["Selected_Layer_Risk"] = get_layer_labels(work, selected_layer, balanced_visual)

        display_cols = [
            c for c in [
                "District_Name",
                "DISTRICT",
                "Selected_Layer_Risk",
                cfg["score_col"],
                cfg["risk_col"],
                "Overall_Risk",
                "Priority_Level",
                "Main_Concern",
                "Recommended_Action",
            ]
            if c in work.columns
        ]

        view = work[display_cols] if display_cols else work

        st.dataframe(view, use_container_width=True, height=430)


with tab2:
    st.subheader("District Profile")

    if scorecard.empty or not district_col:
        st.warning("District profile unavailable.")
    else:
        profile_district = selected_district

        if profile_district == "All Punjab":
            profile_district = st.selectbox(
                "Select district for profile",
                sorted(scorecard[district_col].dropna().astype(str).unique().tolist())
            )

        row = scorecard[scorecard[district_col].astype(str) == str(profile_district)].iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Overall Risk", row.get("Overall_Risk", "—"))
        c2.metric("Main Concern", row.get("Main_Concern", "—"))
        c3.metric("Priority", row.get("Priority_Level", "—"))

        st.success(row.get("Recommended_Action", "Routine monitoring recommended."))


with tab3:
    st.subheader("Methodology")
    st.markdown(
        """
        This dashboard uses backend satellite indicators from 2018–2024 to prepare public-facing agro-climate stress layers for Punjab.

        The map is clipped to Punjab boundaries. When a district is selected, the result raster is clipped to the selected district.

        **Indicators used:** CHIRPS rainfall, Sentinel-2 NDVI, MODIS LST, GLDAS soil moisture, GRACE/GRACE-FO total water storage anomaly, and GNSS/GPS/Baidu as an optional field-observation/navigation indicator.

        **Important note:** Water Storage Stress is not exact groundwater depth. It represents broad regional water storage pressure.
        """
    )


with tab4:
    st.subheader("Raster Diagnostics")
    st.dataframe(raster_diagnostics_table(), use_container_width=True)
    st.info(
        "For class rasters, values should normally include 1, 2, 3 and 4. "
        "If a raster is missing here, upload it under data/rasters/ in GitHub."
    )
