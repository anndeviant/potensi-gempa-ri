from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent

try:
    from timeseries_infer import forecast_within_radius

    TIMESERIES_IMPORT_ERROR = None
except Exception as import_err:  # pragma: no cover - defensive for serverless env
    forecast_within_radius = None
    TIMESERIES_IMPORT_ERROR = str(import_err)

app = Flask(__name__)
API_KEY = os.getenv("API_KEY", "")
allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://peta-potensi-gempa-ri.vercel.app",
)

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://peta-potensi-gempa-ri.vercel.app",
]

allowed_origins = [
    origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()
]
for origin in DEFAULT_ALLOWED_ORIGINS:
    if origin not in allowed_origins:
        allowed_origins.append(origin)

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key"],
)

MODEL_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"

HAZARD_MODEL_DIR = MODEL_DIR / "hazard"
TIMESERIES_MODEL_DIR = MODEL_DIR / "timeseries"
HAZARD_DATA_DIR = DATA_DIR / "hazard"

MODEL_PATH_1_JOBLIB = HAZARD_MODEL_DIR / "hazard_model.joblib"
LABEL_ENCODER_MODEL_1 = HAZARD_MODEL_DIR / "label_encoder.pkl"
HAZARD_METADATA_PATH = HAZARD_MODEL_DIR / "training_metadata.json"
M5_BUNDLE_PATH = TIMESERIES_MODEL_DIR / "m5_regional_nearest_bundle.joblib"

MODEL_PATH_1_JOBLIB_ROOT = BACKEND_DIR / "hazard_model.joblib"
LABEL_ENCODER_MODEL_1_ROOT = BACKEND_DIR / "label_encoder.pkl"
HAZARD_METADATA_PATH_ROOT = BACKEND_DIR / "training_metadata.json"
GRID_DF_PATH_ROOT = BACKEND_DIR / "grid_df.csv"


def first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Biarkan preflight CORS lewat tanpa API key agar browser bisa lanjut ke request utama.
        if request.method == "OPTIONS":
            return ("", 204)

        if not API_KEY:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Server belum dikonfigurasi: API_KEY belum diset",
                    }
                ),
                500,
            )

        user_key = request.headers.get("x-api-key")
        if not user_key or user_key != API_KEY:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Unauthorized: API Key salah atau tidak ada",
                    }
                ),
                401,
            )
        return f(*args, **kwargs)

    return decorated_function


model_1 = None
label_encoder = None
hazard_class_labels = []
hazard_quantiles = {}
hazard_load_error = None
grid_load_error = None
hazard_joblib_path = first_existing(MODEL_PATH_1_JOBLIB, MODEL_PATH_1_JOBLIB_ROOT)
label_encoder_path = first_existing(LABEL_ENCODER_MODEL_1, LABEL_ENCODER_MODEL_1_ROOT)
hazard_metadata_path = first_existing(HAZARD_METADATA_PATH, HAZARD_METADATA_PATH_ROOT)

if hazard_metadata_path is not None:
    try:
        with open(hazard_metadata_path, "r", encoding="utf-8") as f:
            hazard_metadata = json.load(f)
    except Exception:
        hazard_metadata = None

    if isinstance(hazard_metadata, dict):
        classes = hazard_metadata.get("classes", [])
        if isinstance(classes, list):
            hazard_class_labels = [str(c) for c in classes]

        quantiles = hazard_metadata.get("hazard_score_quantiles", {})
        if isinstance(quantiles, dict):
            try:
                hazard_quantiles = {
                    "q1": float(quantiles["q1"]),
                    "q2": float(quantiles["q2"]),
                    "q3": float(quantiles["q3"]),
                }
            except Exception:
                hazard_quantiles = {}

if hazard_joblib_path is not None:
    try:
        model_1 = joblib.load(hazard_joblib_path)
        if label_encoder_path is not None:
            label_encoder = joblib.load(label_encoder_path)
        else:
            print("label_encoder.pkl tidak ditemukan, menggunakan fallback label.")
        print("Model hazard (joblib) berhasil dimuat.")
    except Exception as e:
        hazard_load_error = str(e)
        print(f"Gagal memuat model hazard (joblib): {e}")
else:
    hazard_load_error = "hazard_model.joblib tidak ditemukan"
    print("File model hazard tidak ditemukan")

m5_bundle_path = first_existing(M5_BUNDLE_PATH)
if m5_bundle_path is not None:
    try:
        m5_bundle = joblib.load(m5_bundle_path)
        print("Model regional M>=5 berhasil dimuat.")
    except Exception as e:
        m5_bundle = None
        print(f"Gagal memuat model M>=5: {e}")
else:
    m5_bundle = None
    print(f"File model M>=5 tidak ditemukan: {M5_BUNDLE_PATH}")


grid_df_path = first_existing(HAZARD_DATA_DIR / "grid_df.csv", GRID_DF_PATH_ROOT)
if grid_df_path is None:
    grid_df = None
    grid_load_error = "grid_df.csv tidak ditemukan"
    print("File grid_df.csv tidak ditemukan di backend/data/hazard")
else:
    try:
        grid_df = pd.read_csv(grid_df_path)
    except Exception as e:
        grid_df = None
        grid_load_error = str(e)
        print(f"Gagal memuat grid_df.csv: {e}")


def get_nearest_features(lat, lon, grid_df):
    distances = (grid_df["lat"] - lat) ** 2 + (grid_df["lon"] - lon) ** 2
    idx = distances.idxmin()
    row = grid_df.loc[[idx]]
    return row[["max_mag", "avg_mag", "avg_depth", "gempa_in_radius_50", "density"]]


def predict_hazard_class(features: pd.DataFrame):
    pred = model_1.predict(features)
    return pred[0]


def predict_hazard_fallback(features: pd.DataFrame):
    # Fallback ini meniru logika pelabelan saat training model hazard.
    row = features.iloc[0]
    score = (
        float(row["max_mag"]) * 0.4
        + float(row["density"]) * 0.3
        + float(row["gempa_in_radius_50"]) * 0.3
    )

    q1 = hazard_quantiles.get("q1")
    q2 = hazard_quantiles.get("q2")
    q3 = hazard_quantiles.get("q3")
    if q1 is None or q2 is None or q3 is None:
        raise RuntimeError("Metadata quantile hazard belum tersedia")

    if score <= q1:
        return "low"
    if score <= q2:
        return "medium"
    if score <= q3:
        return "high"
    return "very_high"


def decode_hazard_label(pred_value) -> str:
    if label_encoder is not None:
        return str(label_encoder.inverse_transform([pred_value])[0])

    if isinstance(pred_value, str):
        return pred_value

    if isinstance(pred_value, (int, float)):
        idx = int(pred_value)
        if 0 <= idx < len(hazard_class_labels):
            return hazard_class_labels[idx]

    return str(pred_value)


@app.route("/predict", methods=["GET"])
@require_api_key
def predict():
    try:
        hazard_ready = model_1 is not None
        fallback_ready = bool(hazard_quantiles)

        if grid_df is None or (not hazard_ready and not fallback_ready):
            reasons = []
            if grid_df is None:
                reasons.append(f"grid_df: {grid_load_error or 'tidak siap'}")
            if not hazard_ready and not fallback_ready:
                reasons.append(
                    f"hazard_model/fallback: {hazard_load_error or 'model tidak bisa dimuat dan metadata quantile tidak tersedia'}"
                )

            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Model hazard belum siap. Periksa hazard_model.joblib, training_metadata.json, dan grid_df.csv",
                        "detail": reasons,
                    }
                ),
                500,
            )

        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        if lat is None or lng is None:
            return jsonify({"error": "Parameter lat dan lng diperlukan"}), 400

        nearest_features = get_nearest_features(lat, lng, grid_df)
        if hazard_ready:
            prediction = predict_hazard_class(nearest_features)
            label = decode_hazard_label(prediction)
            source = "model"
        else:
            label = predict_hazard_fallback(nearest_features)
            source = "fallback_metadata"

        return jsonify(
            {
                "status": "success",
                "data": {
                    "hazard_level": label,
                    "lat": lat,
                    "lng": lng,
                    "source": source,
                },
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict-m5-radius", methods=["GET"])
@require_api_key
def predict_m5_radius():
    try:
        if forecast_within_radius is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Fitur M>=5 tidak tersedia di environment ini",
                        "detail": TIMESERIES_IMPORT_ERROR,
                    }
                ),
                500,
            )

        if m5_bundle is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Model M>=5 belum tersedia. Jalankan training model terlebih dahulu.",
                    }
                ),
                500,
            )

        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        radius_km = request.args.get("radius_km", default=50, type=float)

        if lat is None or lng is None:
            return (
                jsonify(
                    {"status": "error", "message": "Parameter lat dan lng diperlukan"}
                ),
                400,
            )
        if radius_km < 50 or radius_km > 200:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "radius_km harus di antara 50 dan 200",
                    }
                ),
                400,
            )

        open_month = datetime.utcnow().strftime("%Y-%m")
        now_month = pd.Timestamp.now().to_period("M").to_timestamp()
        target_month = (now_month.to_period("M") + 1).to_timestamp().strftime("%Y-%m")

        result = forecast_within_radius(
            bundle=m5_bundle,
            lat=float(lat),
            lon=float(lng),
            radius_km=float(radius_km),
            end_month=target_month,
            reference_month=None,
        )

        forecast_df = result["forecast"].copy()
        forecast_df["target_month"] = pd.to_datetime(forecast_df["target_month"])
        pred_row = forecast_df.sort_values("target_month").iloc[-1]

        return jsonify(
            {
                "status": "success",
                "data": {
                    "lat": float(result["input_lat"]),
                    "lng": float(result["input_lon"]),
                    "radius_km": float(result["radius_km"]),
                    "nearest_region": result["nearest_region"]["region_name"],
                    "distance_km": float(result["nearest_region"]["distance_km"]),
                    "n_regions_in_radius": int(len(result["included_regions"])),
                    "estimated_total_m5_in_radius": float(pred_row["pred_count_m5"]),
                    "open_month": open_month,
                    "target_month": str(pd.Timestamp(pred_row["target_month"]).date()),
                    "model_last_observed_month": str(
                        m5_bundle["metadata"].get("last_observed_month", "")
                    ),
                },
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
