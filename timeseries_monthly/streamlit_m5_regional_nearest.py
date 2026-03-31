from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

from model_m5_regional_nearest import (
    ARTIFACT_DIR,
    DATA_PATH,
    forecast_within_radius,
    haversine_km,
    load_filtered_events_regional,
    train_and_save_regional_model,
)


st.set_page_config(page_title="Regional Nearest Forecast M>=5", layout="wide")


def model_path_from_dir(artifact_dir: Path) -> Path:
    return artifact_dir / "saved_models" / "m5_regional_nearest_bundle.joblib"


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


@st.cache_data
def load_events_for_map(path: str) -> pd.DataFrame:
    events = load_filtered_events_regional(Path(path)).copy()
    return events[["lat", "lon", "mag", "remark", "month"]].reset_index(drop=True)


def filter_events_within_radius(
    events: pd.DataFrame, input_lat: float, input_lon: float, radius_km: float
) -> pd.DataFrame:
    out = events.copy()
    out["dist_km"] = haversine_km(
        out["lat"].to_numpy(), out["lon"].to_numpy(), input_lat, input_lon
    )
    out = out[out["dist_km"] <= float(radius_km)].copy()
    return out.sort_values("dist_km").reset_index(drop=True)


def render_map(
    input_lat: float,
    input_lon: float,
    centroid_lat: float,
    centroid_lon: float,
    radius_km: float,
    included_regions: pd.DataFrame,
    events_in_radius: pd.DataFrame,
):
    points_input = pd.DataFrame(
        [{"lat": input_lat, "lon": input_lon, "label": "Input user"}]
    )
    points_centroid = pd.DataFrame(
        [
            {
                "lat": centroid_lat,
                "lon": centroid_lon,
                "label": "Nearest region centroid",
            }
        ]
    )
    points_radius = pd.DataFrame(
        [{"lat": input_lat, "lon": input_lon, "label": f"Radius {radius_km:.0f} km"}]
    )
    points_included = included_regions.copy()
    points_included["label"] = (
        points_included["region_name"].astype(str)
        + "\nDist: "
        + points_included["dist_km"].round(1).astype(str)
        + " km"
    )
    points_included = points_included.rename(
        columns={"centroid_lat": "lat", "centroid_lon": "lon"}
    )
    heat_data = events_in_radius[["lat", "lon", "mag"]].copy()
    events_points = events_in_radius[["lat", "lon", "mag", "dist_km"]].copy()
    events_points["label"] = (
        "M"
        + events_points["mag"].round(2).astype(str)
        + "\nDist: "
        + events_points["dist_km"].round(1).astype(str)
        + " km"
    )

    view_state = pdk.ViewState(
        latitude=float((input_lat + centroid_lat) / 2.0),
        longitude=float((input_lon + centroid_lon) / 2.0),
        zoom=5,
        pitch=0,
    )

    layer_input = pdk.Layer(
        "ScatterplotLayer",
        data=points_input,
        get_position="[lon, lat]",
        get_radius=14000,
        get_fill_color=[30, 30, 220, 220],
        pickable=True,
    )
    layer_centroid = pdk.Layer(
        "ScatterplotLayer",
        data=points_centroid,
        get_position="[lon, lat]",
        get_radius=14000,
        get_fill_color=[255, 140, 0, 220],
        pickable=True,
    )
    layer_radius = pdk.Layer(
        "ScatterplotLayer",
        data=points_radius,
        get_position="[lon, lat]",
        get_radius=float(radius_km) * 1000.0,
        get_fill_color=[0, 0, 0, 0],
        stroked=True,
        filled=False,
        get_line_color=[0, 180, 120, 220],
        line_width_min_pixels=2,
        pickable=False,
    )
    layer_included = pdk.Layer(
        "ScatterplotLayer",
        data=points_included,
        get_position="[lon, lat]",
        get_radius=9000,
        get_fill_color=[80, 200, 120, 170],
        pickable=True,
    )
    layer_events = pdk.Layer(
        "ScatterplotLayer",
        data=events_points,
        get_position="[lon, lat]",
        get_radius=3500,
        get_fill_color=[255, 255, 255, 90],
        pickable=True,
    )
    layer_heat = pdk.Layer(
        "HeatmapLayer",
        data=heat_data,
        get_position="[lon, lat]",
        get_weight="mag",
        radius_pixels=60,
        intensity=1.0,
        threshold=0.1,
        opacity=0.6,
    )

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=view_state,
        layers=[
            layer_heat,
            layer_events,
            layer_radius,
            layer_included,
            layer_input,
            layer_centroid,
        ],
        tooltip={"text": "{label}\nLat: {lat}\nLon: {lon}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


st.title("Prediksi Regional M>=5 dari Input Lat/Lon")
st.caption(
    "User input titik, sistem cari region terdekat, lalu tampilkan estimasi total gempa M>=5 untuk 1 bulan ke depan dari waktu user membuka website."
)

with st.sidebar:
    st.header("Konfigurasi")
    artifact_dir_text = st.text_input("Artifact directory", value=str(ARTIFACT_DIR))
    artifact_dir = Path(artifact_dir_text)
    model_path = model_path_from_dir(artifact_dir)

    st.write("Data path:", str(DATA_PATH))
    st.write("Model path:", str(model_path))

    if st.button("Train / Retrain Model", type="primary"):
        with st.spinner("Training model regional khusus..."):
            summary = train_and_save_regional_model(
                data_path=DATA_PATH, artifact_dir=artifact_dir
            )
        st.success("Training selesai")
        st.json(summary)
        st.cache_resource.clear()

if not model_path.exists():
    st.warning("Model regional belum ada. Klik Train / Retrain Model di sidebar.")
    st.stop()

bundle = load_bundle(str(model_path))
events_map = load_events_for_map(str(DATA_PATH))
last_obs = str(bundle["metadata"]["last_observed_month"])
max_end_month = pd.Timestamp(str(bundle["metadata"].get("max_end_month", "2026-12-01")))

c1, c2, c3 = st.columns(3)
with c1:
    lat = st.number_input("Latitude input", value=-6.2, format="%.6f")
with c2:
    lon = st.number_input("Longitude input", value=106.8, format="%.6f")
with c3:
    open_date = st.date_input(
        "Tanggal user membuka website",
        value=pd.Timestamp.today().date(),
    )

radius_km = st.slider(
    "Radius akumulasi prediksi (km)", min_value=50, max_value=200, value=50, step=10
)

open_month = pd.Timestamp(open_date).to_period("M").to_timestamp()
target_month = (open_month.to_period("M") + 1).to_timestamp()

st.caption(
    f"Waktu prediksi otomatis: bulan depan dari tanggal buka web = {target_month.strftime('%Y-%m')}"
)
st.caption(
    f"Catatan data model: observasi terakhir {last_obs[:7]}, sehingga prediksi bulan depan dilakukan sebagai proyeksi dari histori sampai horizon target."
)

if st.button("Prediksi Akumulasi Radius", type="primary"):
    if target_month > max_end_month:
        st.error(
            f"Target month {target_month.strftime('%Y-%m')} melebihi batas model {max_end_month.strftime('%Y-%m')}. Pilih tanggal buka <= {max_end_month.to_period('M').to_timestamp().strftime('%Y-%m-01')}"
        )
        st.stop()

    result = forecast_within_radius(
        bundle=bundle,
        lat=float(lat),
        lon=float(lon),
        radius_km=float(radius_km),
        end_month=target_month.strftime("%Y-%m"),
        reference_month=None,
    )
    st.session_state["radius_result"] = result
    st.session_state["target_month"] = target_month.strftime("%Y-%m-%d")
    st.session_state["open_month"] = open_month.strftime("%Y-%m-%d")

if "radius_result" in st.session_state:
    out = st.session_state["radius_result"]
    nearest = out["nearest_region"]
    fcst = out["forecast"].copy()
    fcst["target_month"] = pd.to_datetime(fcst["target_month"])
    pred_row = fcst.sort_values("target_month").iloc[-1]
    included = out["included_regions"].copy()
    events_in_radius = filter_events_within_radius(
        events=events_map,
        input_lat=float(out["input_lat"]),
        input_lon=float(out["input_lon"]),
        radius_km=float(out["radius_km"]),
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Nearest region", nearest["region_name"])
    with m2:
        st.metric("Distance input-ke-centroid", f"{nearest['distance_km']:.1f} km")
    with m3:
        st.metric("Radius", f"{out['radius_km']:.0f} km")
    with m4:
        st.metric("Region ter-cover", int(len(included)))
    with m5:
        st.metric("Event historis dalam radius", int(len(events_in_radius)))
    with m6:
        st.metric("Open month", st.session_state.get("open_month", "-"))

    st.metric("Target month", st.session_state.get("target_month", "-"))

    st.success(
        f"Estimasi total gempa M>=5 dalam radius {out['radius_km']:.0f} km untuk 1 bulan ke depan: {float(pred_row['pred_count_m5']):.4f} kejadian"
    )

    render_map(
        input_lat=float(out["input_lat"]),
        input_lon=float(out["input_lon"]),
        centroid_lat=float(nearest["centroid_lat"]),
        centroid_lon=float(nearest["centroid_lon"]),
        radius_km=float(out["radius_km"]),
        included_regions=included,
        events_in_radius=events_in_radius,
    )

    detail_df = pd.DataFrame(
        [
            {
                "input_lat": round(float(out["input_lat"]), 3),
                "input_lon": round(float(out["input_lon"]), 3),
                "nearest_region": nearest["region_name"],
                "radius_km": round(float(out["radius_km"]), 1),
                "n_regions_in_radius": int(len(included)),
                "distance_to_region_centroid_km": round(
                    float(nearest["distance_km"]), 2
                ),
                "target_month": st.session_state.get("target_month", "-"),
                "estimated_total_m5_in_radius": round(
                    float(pred_row["pred_count_m5"]), 4
                ),
            }
        ]
    )
    st.dataframe(detail_df, use_container_width=True)

    with st.expander("Daftar region dalam radius"):
        region_df = included[["region_name", "dist_km"]].copy()
        region_df = region_df.rename(
            columns={"region_name": "region", "dist_km": "distance_km"}
        )
        region_df["distance_km"] = region_df["distance_km"].round(2)
        st.dataframe(region_df, use_container_width=True)

    with st.expander("Sampel event M>=5 dalam radius (untuk heatmap)"):
        event_df = events_in_radius[
            ["month", "lat", "lon", "mag", "remark", "dist_km"]
        ].copy()
        event_df["dist_km"] = event_df["dist_km"].round(2)
        st.dataframe(event_df.head(300), use_container_width=True)

    st.info(
        "Akumulasi dihitung dari penjumlahan prediksi semua region yang centroid-nya masuk radius terpilih."
    )
