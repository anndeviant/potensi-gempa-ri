from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from timeseries_monthly.archive.model_m5_monthly_radius50 import (
    ARTIFACT_DIR,
    DATA_PATH,
    load_filtered_events,
    predict_next_month_point,
    predict_next_month_region,
    train_and_save,
)


st.set_page_config(page_title="Regional Grid Forecast M>=5", layout="wide")


def model_bundle_path(artifact_dir: Path) -> Path:
    return artifact_dir / "saved_models" / "m5_monthly_radius50_bundle.joblib"


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


@st.cache_data
def load_events(path: str) -> pd.DataFrame:
    return load_filtered_events(Path(path))


def _cell_center_from_bin(
    lat_bin: int, lon_bin: int, grid_deg: float
) -> tuple[float, float]:
    return (lat_bin + 0.5) * grid_deg, (lon_bin + 0.5) * grid_deg


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371.0 * c


def region_profile(events_df: pd.DataFrame, region_name: str) -> dict:
    region_df = events_df[events_df["remark"] == region_name].copy()
    if region_df.empty:
        return {
            "region": region_name,
            "n_events": 0,
            "centroid_lat": 0.0,
            "centroid_lon": 0.0,
            "adaptive_radius_km": 250.0,
        }

    c_lat = float(region_df["lat"].mean())
    c_lon = float(region_df["lon"].mean())

    dists = haversine_km(
        region_df["lat"].to_numpy(),
        region_df["lon"].to_numpy(),
        c_lat,
        c_lon,
    )
    p80 = float(np.percentile(dists, 80)) if len(dists) > 0 else 200.0
    adaptive_radius = float(np.clip(p80 * 1.35, 120.0, 1200.0))

    return {
        "region": region_name,
        "n_events": int(len(region_df)),
        "centroid_lat": c_lat,
        "centroid_lon": c_lon,
        "adaptive_radius_km": adaptive_radius,
    }


@st.cache_data
def build_all_cell_predictions(
    model_path: str, reference_month: str | None
) -> tuple[pd.DataFrame, str, str]:
    bundle = joblib.load(model_path)
    cell_lookup = bundle["spatial"]["cell_lookup"].copy()
    grid_deg = float(bundle["metadata"]["grid_deg"])

    rows = []
    target_month = None
    used_reference = None

    for row in cell_lookup.itertuples(index=False):
        lat, lon = _cell_center_from_bin(int(row.lat_bin), int(row.lon_bin), grid_deg)
        pred = predict_next_month_point(
            bundle=bundle,
            lat=float(lat),
            lon=float(lon),
            reference_month=reference_month,
        )
        if target_month is None:
            target_month = pred["target_month"]
            used_reference = pred["reference_month"]

        rows.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "lat_bin": int(pred["lat_bin"]),
                "lon_bin": int(pred["lon_bin"]),
                "pred_count_m5_next_month": float(pred["pred_count_m5_next_month"]),
            }
        )

    pred_df = pd.DataFrame(rows)
    pred_df["radius"] = (
        pred_df["pred_count_m5_next_month"].clip(lower=0.0) * 30000.0
    ).clip(lower=4500.0)
    pred_df = pred_df.sort_values("pred_count_m5_next_month", ascending=False)

    return pred_df, str(used_reference), str(target_month)


def render_regional_map(
    map_df: pd.DataFrame,
    centroid_lat: float,
    centroid_lon: float,
    radius_km: float,
    map_style: str,
):
    if map_df.empty:
        st.warning(
            "Tidak ada grid prediksi yang lolos filter region/radius. Coba turunkan min prediksi atau naikkan radius."
        )
        return

    view_state = pdk.ViewState(
        latitude=float(centroid_lat),
        longitude=float(centroid_lon),
        zoom=5.2,
        pitch=0,
    )

    layers = []
    if map_style == "Heatmap":
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_weight="pred_count_m5_next_month",
                radiusPixels=85,
                aggregation="SUM",
                pickable=True,
            )
        )
    else:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[255, 140, 0, 160]",
                pickable=True,
            )
        )

    marker_df = pd.DataFrame(
        [
            {
                "lat": float(centroid_lat),
                "lon": float(centroid_lon),
                "label": "Region centroid",
            }
        ]
    )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=marker_df,
            get_position="[lon, lat]",
            get_radius=9000,
            get_fill_color=[30, 30, 220, 220],
            pickable=True,
        )
    )

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "text": "Lat: {lat}\nLon: {lon}\nPred M>=5: {pred_count_m5_next_month}"
        },
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(
        f"Centroid region ditandai titik biru. Area peta difilter dalam radius {radius_km:.0f} km dari centroid region."
    )


st.title("Regional Grid Forecast Gempa M>=5 (1 Bulan)")
st.caption(
    "Versi ini fokus ke regional-grid. Peta dipotong berdasarkan centroid dan radius region agar hasil tidak lompat ke region jauh."
)

with st.sidebar:
    st.header("Konfigurasi")
    artifact_dir_text = st.text_input("Artifact directory", value=str(ARTIFACT_DIR))
    artifact_dir = Path(artifact_dir_text)
    model_path = model_bundle_path(artifact_dir)

    st.write("Data path:", str(DATA_PATH))
    st.write("Model path:", str(model_path))

    if st.button("Train / Retrain Model", type="primary"):
        with st.spinner("Training model..."):
            summary = train_and_save(data_path=DATA_PATH, artifact_dir=artifact_dir)
        st.success("Training selesai")
        st.json(summary)
        st.cache_resource.clear()
        st.cache_data.clear()

if not model_path.exists():
    st.warning("Model belum tersedia. Train dulu di sidebar.")
    st.stop()

bundle = load_bundle(str(model_path))
events = load_events(str(DATA_PATH))

region_map = bundle["region"]["region_map"]
all_regions = sorted(region_map["entity"].unique().tolist())

c1, c2, c3 = st.columns(3)
with c1:
    region_name = st.selectbox("Pilih region", options=all_regions, index=0)
with c2:
    ref_month_input = st.text_input("Reference Month (opsional, YYYY-MM)", value="")
with c3:
    map_style = st.selectbox("Tipe peta", ["Heatmap", "Scatter"], index=0)

profile = region_profile(events, region_name)
adaptive_radius = float(profile["adaptive_radius_km"])

s1, s2, s3 = st.columns(3)
with s1:
    radius_km = st.slider(
        "Radius regional filter (km)", 80, 1500, int(round(adaptive_radius)), 20
    )
with s2:
    min_pred = st.slider("Minimum prediksi", 0.0, 2.0, 0.02, 0.01)
with s3:
    top_n = st.slider("Top grid tampil", 10, 600, 180, 10)

if st.button("Prediksi Region + Refresh Peta", type="primary"):
    st.session_state["run_region"] = {
        "region_name": region_name,
        "ref_month": (ref_month_input.strip() or None),
        "radius_km": radius_km,
        "min_pred": min_pred,
        "top_n": top_n,
        "map_style": map_style,
    }

if "run_region" in st.session_state:
    params = st.session_state["run_region"]

    result = predict_next_month_region(
        bundle=bundle,
        region_name=params["region_name"],
        reference_month=params["ref_month"],
    )

    p = region_profile(events, params["region_name"])
    c_lat = float(p["centroid_lat"])
    c_lon = float(p["centroid_lon"])

    with st.spinner("Membangun prediksi grid regional..."):
        all_pred_df, used_ref, target_month = build_all_cell_predictions(
            str(model_path), params["ref_month"]
        )

    all_pred_df["dist_km"] = haversine_km(
        all_pred_df["lat"].to_numpy(),
        all_pred_df["lon"].to_numpy(),
        c_lat,
        c_lon,
    )

    regional_df = all_pred_df[
        (all_pred_df["dist_km"] <= float(params["radius_km"]))
        & (all_pred_df["pred_count_m5_next_month"] >= float(params["min_pred"]))
    ].copy()
    regional_df = regional_df.sort_values(
        "pred_count_m5_next_month", ascending=False
    ).head(int(params["top_n"]))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Region", result["region_name"])
    with m2:
        st.metric(
            "Prediksi M>=5 (1 bulan)",
            f"{float(result['pred_count_m5_next_month']):.4f}",
        )
    with m3:
        st.metric("Target month", result["target_month"])
    with m4:
        st.metric("Grid lolos filter", str(len(regional_df)))

    st.info(
        f"Model ini deterministik: input yang sama menghasilkan output yang sama. Ubah region/radius/reference month agar peta berubah."
    )
    st.caption(
        f"Reference dipakai: {used_ref} | Target: {target_month} | Radius adaptif rekomendasi untuk {params['region_name']}: {p['adaptive_radius_km']:.0f} km"
    )

    render_regional_map(
        regional_df,
        centroid_lat=c_lat,
        centroid_lon=c_lon,
        radius_km=float(params["radius_km"]),
        map_style=params["map_style"],
    )

    if not regional_df.empty:
        view_df = regional_df[
            ["lat", "lon", "dist_km", "pred_count_m5_next_month"]
        ].copy()
        # Keep location less sensitive by rounding to coarse grid on table display.
        view_df["lat"] = view_df["lat"].round(2)
        view_df["lon"] = view_df["lon"].round(2)
        view_df["dist_km"] = view_df["dist_km"].round(1)
        view_df["pred_count_m5_next_month"] = view_df["pred_count_m5_next_month"].round(
            4
        )
        st.dataframe(view_df.head(30), use_container_width=True)
