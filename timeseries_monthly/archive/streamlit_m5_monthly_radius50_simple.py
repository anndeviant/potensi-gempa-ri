from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

from timeseries_monthly.archive.model_m5_monthly_radius50 import (
    ARTIFACT_DIR,
    DATA_PATH,
    predict_next_month_point,
    predict_next_month_region,
    train_and_save,
)


st.set_page_config(page_title="M5 Monthly Radius50 - Simple Test", layout="wide")


def default_artifact_dir() -> Path:
    return ARTIFACT_DIR


def model_bundle_path(artifact_dir: Path) -> Path:
    return artifact_dir / "saved_models" / "m5_monthly_radius50_bundle.joblib"


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def load_metrics(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _cell_center_from_bin(
    lat_bin: int, lon_bin: int, grid_deg: float
) -> tuple[float, float]:
    return (lat_bin + 0.5) * grid_deg, (lon_bin + 0.5) * grid_deg


@st.cache_data
def build_spatial_prediction_map(
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
                "entity": pred["lat_bin"].__str__() + "_" + pred["lon_bin"].__str__(),
                "lat": float(lat),
                "lon": float(lon),
                "lat_bin": int(pred["lat_bin"]),
                "lon_bin": int(pred["lon_bin"]),
                "pred_count_m5_next_month": float(pred["pred_count_m5_next_month"]),
            }
        )

    pred_df = pd.DataFrame(rows)
    # PyDeck JSON expressions do not allow function calls such as max(...),
    # so radius is precomputed as a plain numeric column.
    pred_df["radius"] = (
        pred_df["pred_count_m5_next_month"].clip(lower=0.0) * 35000.0
    ).clip(lower=3000.0)
    pred_df = pred_df.sort_values("pred_count_m5_next_month", ascending=False)
    return pred_df, str(used_reference), str(target_month)


def render_input_only_map(
    lat: float, lon: float, pred_count: float, map_style: str = "Heatmap"
):
    pred_non_negative = max(float(pred_count), 0.0)

    map_df = pd.DataFrame(
        [
            {
                "lat": float(lat),
                "lon": float(lon),
                "pred": pred_non_negative,
                "pred_weight": max(pred_non_negative, 0.05),
                "radius": max(25000.0, pred_non_negative * 250000.0),
            }
        ]
    )

    view_state = pdk.ViewState(
        latitude=float(lat),
        longitude=float(lon),
        zoom=6,
        pitch=0,
    )

    if map_style == "Heatmap":
        layers = [
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_weight="pred_weight",
                radiusPixels=120,
                aggregation="SUM",
                pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius=6000,
                get_fill_color=[20, 20, 220, 220],
                pickable=False,
            ),
        ]
    else:
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color=[255, 140, 0, 180],
                pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius=6000,
                get_fill_color=[20, 20, 220, 220],
                pickable=False,
            ),
        ]

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "Input point\nLat: {lat}\nLon: {lon}\nPred M>=5: {pred}"},
    )
    st.pydeck_chart(deck, use_container_width=True)


def render_prediction_map(
    pred_df: pd.DataFrame,
    highlighted_point: tuple[float, float] | None = None,
    map_style: str = "Scatter",
):
    if pred_df.empty:
        st.info("Data prediksi peta belum tersedia")
        return

    view_state = pdk.ViewState(
        latitude=float(pred_df["lat"].mean()),
        longitude=float(pred_df["lon"].mean()),
        zoom=4.2,
        pitch=0,
    )

    layers = []
    if map_style == "Scatter":
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pred_df,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[255, 140, 0, 140]",
                pickable=True,
            )
        )
    else:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=pred_df,
                get_position="[lon, lat]",
                get_weight="pred_count_m5_next_month",
                aggregation="SUM",
                pickable=True,
            )
        )

    if highlighted_point is not None:
        point_df = pd.DataFrame(
            [{"lat": float(highlighted_point[0]), "lon": float(highlighted_point[1])}]
        )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=point_df,
                get_position="[lon, lat]",
                get_radius=50000,
                get_fill_color=[20, 20, 220, 220],
                pickable=False,
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


st.title("Gempa M>=5 Bulanan - Simple Testing UI")
st.caption("Mode titik (proxy cell radius 50km) dan mode region, horizon 1 bulan.")

with st.sidebar:
    st.header("Konfigurasi")
    artifact_dir_text = st.text_input(
        "Artifact directory", value=str(default_artifact_dir())
    )
    artifact_dir = Path(artifact_dir_text)
    model_path = model_bundle_path(artifact_dir)
    metrics_path = artifact_dir / "metrics.csv"
    metadata_path = artifact_dir / "metadata.json"

    st.write("Data path:", str(DATA_PATH))
    st.write("Model path:", str(model_path))

    if st.button("Train / Retrain Model", type="primary"):
        with st.spinner("Training model, mohon tunggu..."):
            summary = train_and_save(data_path=DATA_PATH, artifact_dir=artifact_dir)
        st.success("Training selesai")
        st.json(summary)
        st.cache_resource.clear()


if not model_path.exists():
    st.warning("Model belum tersedia. Klik 'Train / Retrain Model' di sidebar.")
    st.stop()


bundle = load_bundle(str(model_path))
metadata = bundle.get("metadata", {})

col_meta, col_metrics = st.columns(2)
with col_meta:
    st.subheader("Metadata")
    st.json(metadata)

with col_metrics:
    st.subheader("Metrics")
    metrics_df = load_metrics(metrics_path)
    if metrics_df.empty:
        st.info("metrics.csv belum ada")
    else:
        st.dataframe(metrics_df, use_container_width=True)


st.divider()
mode = st.radio("Pilih mode inferensi", ["Point", "Region"], horizontal=True)

map_settings_1, map_settings_2, map_settings_3 = st.columns(3)
with map_settings_1:
    map_style = st.selectbox("Tipe peta", ["Heatmap", "Scatter"], index=0)
with map_settings_2:
    min_pred = st.slider("Filter minimum prediksi", 0.0, 2.0, 0.01, 0.01)
with map_settings_3:
    top_n = st.slider("Top hotspot ditampilkan", 10, 300, 80, 10)

if mode == "Point":
    st.subheader("Inferensi Mode Point")
    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.number_input("Latitude", value=-6.2, format="%.6f")
    with c2:
        lon = st.number_input("Longitude", value=106.8, format="%.6f")
    with c3:
        ref_month = st.text_input("Reference Month (opsional, YYYY-MM)", value="")

    ref_arg = ref_month.strip() or None

    if st.button("Prediksi Point", type="primary"):
        result = predict_next_month_point(
            bundle=bundle,
            lat=float(lat),
            lon=float(lon),
            reference_month=ref_arg,
        )
        st.session_state["point_result"] = result

    if "point_result" in st.session_state:
        result = st.session_state["point_result"]
        st.success("Prediksi berhasil")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Prediksi M>=5 bulan depan", f"{result['pred_count_m5_next_month']:.4f}"
            )
        with col_b:
            st.metric("Reference month", result["reference_month"])
        with col_c:
            st.metric("Target month", result["target_month"])

        st.caption(
            "Peta menampilkan hanya titik input Anda. Warna/area adalah visualisasi level prediksi di lokasi tersebut."
        )
        render_input_only_map(
            lat=float(result["input_lat"]),
            lon=float(result["input_lon"]),
            pred_count=float(result["pred_count_m5_next_month"]),
            map_style=map_style,
        )
        st.info(
            f"Prediksi jumlah gempa M>=5 untuk 1 bulan ke depan di titik ini: {float(result['pred_count_m5_next_month']):.4f} kejadian."
        )

if mode == "Region":
    st.subheader("Inferensi Mode Region")
    region_map = bundle["region"]["region_map"]
    all_regions = sorted(region_map["entity"].unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        region_name = st.selectbox("Pilih region", options=all_regions, index=0)
    with c2:
        ref_month = st.text_input("Reference Month (opsional, YYYY-MM)", value="")

    ref_arg = ref_month.strip() or None

    if st.button("Prediksi Region", type="primary"):
        result = predict_next_month_region(
            bundle=bundle,
            region_name=region_name,
            reference_month=ref_arg,
        )
        st.session_state["region_result"] = result

    if "region_result" in st.session_state:
        result = st.session_state["region_result"]
        st.success("Prediksi berhasil")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Prediksi M>=5 bulan depan", f"{result['pred_count_m5_next_month']:.4f}"
            )
        with col_b:
            st.metric("Region", result["region_name"])
        with col_c:
            st.metric("Target month", result["target_month"])

        st.json(result)

        with st.spinner("Membangun peta prediksi spasial..."):
            map_df, used_ref, target_month = build_spatial_prediction_map(
                str(model_path), ref_arg
            )

        map_df = map_df[map_df["pred_count_m5_next_month"] >= min_pred].head(top_n)
        st.caption(
            f"Peta hotspot spasial untuk target month {target_month} (reference {used_ref}), difilter prediksi >= {min_pred}."
        )
        render_prediction_map(map_df, highlighted_point=None, map_style=map_style)
        st.dataframe(
            map_df[["lat", "lon", "pred_count_m5_next_month"]].head(20),
            use_container_width=True,
        )


with st.expander("Raw metadata.json"):
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="json")
    else:
        st.info("metadata.json belum ada")
