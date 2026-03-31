from __future__ import annotations

import numpy as np
import pandas as pd

MAX_END_MONTH = pd.Timestamp("2026-12-01")


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
        x_pred = pd.DataFrame([feat])[feature_cols]
        pred = float(np.clip(model.predict(x_pred)[0], 0.0, None))

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
