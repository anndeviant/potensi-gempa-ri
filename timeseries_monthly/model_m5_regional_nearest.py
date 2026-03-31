from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from timeseries_monthly.archive.model_m5_monthly_radius50 import (
        MAG_THRESHOLD,
        DatasetConfig,
        _build_feature_columns,
        _evaluate_panel,
        _fit_regressor,
        _split_train_test_by_month,
        build_region_panel,
    )
except ModuleNotFoundError:
    from archive.model_m5_monthly_radius50 import (
        MAG_THRESHOLD,
        DatasetConfig,
        _build_feature_columns,
        _evaluate_panel,
        _fit_regressor,
        _split_train_test_by_month,
        build_region_panel,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "timeseries_monthly" / "data_gempa_kaggle" / "katalog_gempa_v2.tsv"
)
ARTIFACT_DIR = (
    PROJECT_ROOT / "timeseries_monthly" / "artifacts" / "model_m5_regional_nearest"
)
MODEL_PATH = ARTIFACT_DIR / "saved_models" / "m5_regional_nearest_bundle.joblib"
MAX_END_MONTH = pd.Timestamp("2026-12-01")


def load_filtered_events_regional(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        raw = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        raw = pd.read_csv(path, low_memory=False)

    cols = set(raw.columns)
    if {"datetime", "latitude", "longitude", "magnitude", "location"}.issubset(cols):
        dt = pd.to_datetime(raw["datetime"], errors="coerce", utc=True).dt.tz_localize(
            None
        )
        events = pd.DataFrame(
            {
                "month": dt.dt.to_period("M").dt.to_timestamp(),
                "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
                "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
                "mag": pd.to_numeric(raw["magnitude"], errors="coerce"),
                "remark": raw["location"],
            }
        )
    elif {"tgl", "lat", "lon", "mag", "remark"}.issubset(cols):
        dt = pd.to_datetime(raw["tgl"], errors="coerce")
        events = pd.DataFrame(
            {
                "month": dt.dt.to_period("M").dt.to_timestamp(),
                "lat": pd.to_numeric(raw["lat"], errors="coerce"),
                "lon": pd.to_numeric(raw["lon"], errors="coerce"),
                "mag": pd.to_numeric(raw["mag"], errors="coerce"),
                "remark": raw["remark"],
            }
        )
    else:
        raise ValueError(
            "Format data tidak dikenali. Wajib ada kolom v2 (datetime, latitude, longitude, magnitude, location) atau format lama (tgl, lat, lon, mag, remark)."
        )

    events = events.dropna(subset=["month", "lat", "lon", "mag"]).copy()
    events = events[events["mag"] >= MAG_THRESHOLD].copy()
    events["remark"] = events["remark"].fillna("Unknown").astype(str)
    return events


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


def build_region_centroids(events: pd.DataFrame) -> pd.DataFrame:
    centroids = (
        events.groupby("remark", as_index=False)
        .agg(
            centroid_lat=("lat", "mean"),
            centroid_lon=("lon", "mean"),
            n_events=("remark", "count"),
        )
        .rename(columns={"remark": "region_name"})
        .sort_values("region_name")
        .reset_index(drop=True)
    )
    return centroids


def nearest_region(lat: float, lon: float, centroids: pd.DataFrame) -> dict:
    c = centroids.copy()
    c["dist_km"] = haversine_km(
        c["centroid_lat"].to_numpy(), c["centroid_lon"].to_numpy(), lat, lon
    )
    nearest = c.sort_values("dist_km", ascending=True).iloc[0]
    return {
        "region_name": str(nearest["region_name"]),
        "distance_km": float(nearest["dist_km"]),
        "centroid_lat": float(nearest["centroid_lat"]),
        "centroid_lon": float(nearest["centroid_lon"]),
    }


def train_and_save_regional_model(
    data_path: Path = DATA_PATH, artifact_dir: Path = ARTIFACT_DIR
) -> dict:
    cfg = DatasetConfig()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "saved_models").mkdir(parents=True, exist_ok=True)

    events = load_filtered_events_regional(data_path)
    centroids = build_region_centroids(events)

    region_panel, region_map = build_region_panel(events, cfg)
    feature_cols = _build_feature_columns(region_panel, extra_cols=["region_code"])

    train_df, test_df = _split_train_test_by_month(region_panel, cfg.test_horizon)
    metrics, pred_test = _evaluate_panel(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        group_cols=["entity", "region_code"],
    )

    model = _fit_regressor(region_panel, feature_cols)
    history = (
        region_panel.pivot(index="month", columns="entity", values="count_m5")
        .sort_index()
        .fillna(0.0)
    )

    bundle = {
        "metadata": {
            "model_type": "regional_only",
            "max_end_month": str(MAX_END_MONTH.date()),
            "last_observed_month": str(events["month"].max().date()),
            "test_horizon": int(cfg.test_horizon),
            "lags": list(cfg.lags),
            "roll_windows": list(cfg.roll_windows),
            "n_regions": int(region_map.shape[0]),
            "n_events": int(len(events)),
        },
        "region": {
            "model": model,
            "feature_cols": feature_cols,
            "region_map": region_map,
            "history": history,
            "centroids": centroids,
        },
    }

    joblib.dump(
        bundle, artifact_dir / "saved_models" / "m5_regional_nearest_bundle.joblib"
    )

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, indent=2)

    pd.DataFrame([metrics]).to_csv(artifact_dir / "metrics.csv", index=False)
    pred_test.to_csv(artifact_dir / "predictions_test_region.csv", index=False)
    centroids.to_csv(artifact_dir / "region_centroids.csv", index=False)

    return {
        "artifact_dir": str(artifact_dir),
        "model_path": str(
            artifact_dir / "saved_models" / "m5_regional_nearest_bundle.joblib"
        ),
        "metrics": metrics,
    }


def _build_single_feature(
    history_series: pd.Series,
    target_month: pd.Timestamp,
    lags: list[int],
    roll_windows: list[int],
    region_code: int,
) -> dict:
    target_month = target_month.to_period("M").to_timestamp()
    feat = {}

    for lag in lags:
        m = (target_month.to_period("M") - lag).to_timestamp()
        feat[f"lag_{lag}"] = float(history_series.get(m, 0.0))

    for w in roll_windows:
        vals = []
        for k in range(1, w + 1):
            m = (target_month.to_period("M") - k).to_timestamp()
            vals.append(float(history_series.get(m, 0.0)))
        feat[f"roll_mean_{w}"] = float(np.mean(vals))
        feat[f"roll_std_{w}"] = float(np.std(vals))

    feat["month_sin"] = float(np.sin(2 * np.pi * target_month.month / 12.0))
    feat["month_cos"] = float(np.cos(2 * np.pi * target_month.month / 12.0))
    feat["region_code"] = int(region_code)
    return feat


def forecast_region(
    bundle: dict, region_name: str, end_month: str, reference_month: str | None = None
) -> pd.DataFrame:
    region_map = bundle["region"]["region_map"]
    match = region_map[region_map["entity"] == region_name]
    if match.empty:
        raise ValueError(f"Region tidak ditemukan: {region_name}")

    region_code = int(match["region_code"].iloc[0])

    if reference_month is None:
        ref = (
            pd.Timestamp(bundle["metadata"]["last_observed_month"])
            .to_period("M")
            .to_timestamp()
        )
    else:
        ref = pd.Timestamp(reference_month).to_period("M").to_timestamp()

    end_ts = pd.Timestamp(end_month).to_period("M").to_timestamp()
    if end_ts <= ref:
        raise ValueError("end_month harus setelah reference_month")
    if end_ts > MAX_END_MONTH:
        raise ValueError(f"end_month maksimal {MAX_END_MONTH.strftime('%Y-%m')}")

    lags = list(bundle["metadata"]["lags"])
    roll_windows = list(bundle["metadata"]["roll_windows"])
    feature_cols = list(bundle["region"]["feature_cols"])

    history_df: pd.DataFrame = bundle["region"]["history"]
    if region_name in history_df.columns:
        sim_series = history_df[region_name].copy()
    else:
        sim_series = pd.Series(0.0, index=history_df.index)

    model = bundle["region"]["model"]
    outputs = []

    target_months = pd.date_range(
        (ref.to_period("M") + 1).to_timestamp(), end_ts, freq="MS"
    )
    for target in target_months:
        feat = _build_single_feature(
            history_series=sim_series,
            target_month=target,
            lags=lags,
            roll_windows=roll_windows,
            region_code=region_code,
        )
        X = pd.DataFrame([feat])[feature_cols]
        pred = float(np.clip(model.predict(X)[0], 0.0, None))

        sim_series.loc[target] = pred
        outputs.append(
            {
                "region_name": region_name,
                "reference_month": str(ref.date()),
                "target_month": str(target.date()),
                "pred_count_m5": pred,
            }
        )

    return pd.DataFrame(outputs)


def forecast_from_latlon(
    bundle: dict,
    lat: float,
    lon: float,
    end_month: str,
    reference_month: str | None = None,
) -> dict:
    centroids: pd.DataFrame = bundle["region"]["centroids"]
    nearest = nearest_region(lat, lon, centroids)
    fcst = forecast_region(
        bundle=bundle,
        region_name=nearest["region_name"],
        end_month=end_month,
        reference_month=reference_month,
    )

    return {
        "input_lat": float(lat),
        "input_lon": float(lon),
        "nearest_region": nearest,
        "forecast": fcst,
    }


def forecast_within_radius(
    bundle: dict,
    lat: float,
    lon: float,
    radius_km: float,
    end_month: str,
    reference_month: str | None = None,
) -> dict:
    if radius_km < 50.0 or radius_km > 200.0:
        raise ValueError("radius_km harus di antara 50 dan 200")

    centroids: pd.DataFrame = bundle["region"]["centroids"].copy()
    centroids["dist_km"] = haversine_km(
        centroids["centroid_lat"].to_numpy(),
        centroids["centroid_lon"].to_numpy(),
        lat,
        lon,
    )
    nearest = centroids.sort_values("dist_km", ascending=True).iloc[0]

    selected = (
        centroids[centroids["dist_km"] <= float(radius_km)]
        .sort_values("dist_km")
        .reset_index(drop=True)
    )
    if selected.empty:
        selected = centroids.sort_values("dist_km", ascending=True).head(1).copy()

    region_forecasts = []
    for region_name in selected["region_name"].tolist():
        fcst = forecast_region(
            bundle=bundle,
            region_name=str(region_name),
            end_month=end_month,
            reference_month=reference_month,
        )
        region_forecasts.append(fcst)

    forecast_detail = pd.concat(region_forecasts, ignore_index=True)
    forecast_total = (
        forecast_detail.groupby("target_month", as_index=False)
        .agg(pred_count_m5=("pred_count_m5", "sum"))
        .sort_values("target_month")
    )

    return {
        "input_lat": float(lat),
        "input_lon": float(lon),
        "radius_km": float(radius_km),
        "nearest_region": {
            "region_name": str(nearest["region_name"]),
            "distance_km": float(nearest["dist_km"]),
            "centroid_lat": float(nearest["centroid_lat"]),
            "centroid_lon": float(nearest["centroid_lon"]),
        },
        "included_regions": selected[
            ["region_name", "centroid_lat", "centroid_lon", "dist_km"]
        ].copy(),
        "forecast_region_detail": forecast_detail,
        "forecast": forecast_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Model khusus regional: nearest region + forecast bulanan"
    )
    parser.add_argument("--action", choices=["train", "infer_latlon"], default="train")
    parser.add_argument("--artifact-dir", type=str, default=str(ARTIFACT_DIR))
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--end-month", type=str, default="2026-12")
    parser.add_argument("--reference-month", type=str, default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    model_path = artifact_dir / "saved_models" / "m5_regional_nearest_bundle.joblib"

    if args.action == "train":
        res = train_and_save_regional_model(
            data_path=DATA_PATH, artifact_dir=artifact_dir
        )
        print(json.dumps(res, indent=2))
        return

    if not model_path.exists():
        raise FileNotFoundError(f"Model belum ada: {model_path}")
    if args.lat is None or args.lon is None:
        raise ValueError("infer_latlon butuh --lat dan --lon")

    bundle = joblib.load(model_path)
    out = forecast_from_latlon(
        bundle=bundle,
        lat=float(args.lat),
        lon=float(args.lon),
        end_month=args.end_month,
        reference_month=args.reference_month,
    )

    compact = {
        "input_lat": out["input_lat"],
        "input_lon": out["input_lon"],
        "nearest_region": out["nearest_region"],
        "forecast_head": out["forecast"].head(6).to_dict(orient="records"),
        "forecast_rows": int(len(out["forecast"])),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
