from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
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
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [
    origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()
]

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
M5_BUNDLE_PATH = TIMESERIES_MODEL_DIR / "m5_regional_nearest_bundle.joblib"


def first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
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
hazard_joblib_path = first_existing(MODEL_PATH_1_JOBLIB)
label_encoder_path = first_existing(LABEL_ENCODER_MODEL_1)

if hazard_joblib_path is not None and label_encoder_path is not None:
    try:
        model_1 = joblib.load(hazard_joblib_path)
        label_encoder = joblib.load(label_encoder_path)
        print("Model hazard (joblib) berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model hazard (joblib): {e}")
else:
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


grid_df_path = first_existing(HAZARD_DATA_DIR / "grid_df.csv")
if grid_df_path is None:
    grid_df = None
    print("File grid_df.csv tidak ditemukan di backend/data/hazard")
else:
    grid_df = pd.read_csv(grid_df_path)


def get_nearest_features(lat, lon, grid_df):
    distances = (grid_df["lat"] - lat) ** 2 + (grid_df["lon"] - lon) ** 2
    idx = distances.idxmin()
    row = grid_df.loc[[idx]]
    return row[["max_mag", "avg_mag", "avg_depth", "gempa_in_radius_50", "density"]]


def predict_hazard_class(features: pd.DataFrame) -> int:
    pred = model_1.predict(features)
    return int(pred[0])


@app.route("/predict", methods=["GET"])
@require_api_key
def predict():
    try:
        if model_1 is None or label_encoder is None or grid_df is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Model hazard belum siap. Periksa hazard_model.joblib, label_encoder.pkl, dan grid_df.csv",
                    }
                ),
                500,
            )

        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        if lat is None or lng is None:
            return jsonify({"error": "Parameter lat dan lng diperlukan"}), 400

        nearest_features = get_nearest_features(lat, lng, grid_df)
        prediction = [predict_hazard_class(nearest_features)]
        label = label_encoder.inverse_transform(prediction)[0]

        return jsonify(
            {
                "status": "success",
                "data": {
                    "hazard_level": label,
                    "lat": lat,
                    "lng": lng,
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
