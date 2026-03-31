from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


MAG_THRESHOLD = 5.0
RADIUS_KM = 50.0
GRID_DEG = RADIUS_KM / 111.0
TEST_HORIZON = 12
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "timeseries_monthly" / "data_gempa_kaggle" / "katalog_gempa.csv"
)
ARTIFACT_DIR = (
    PROJECT_ROOT / "timeseries_monthly" / "artifacts" / "model_m5_monthly_radius50"
)
MODEL_PATH = ARTIFACT_DIR / "saved_models" / "m5_monthly_radius50_bundle.joblib"


@dataclass
class DatasetConfig:
    test_horizon: int = TEST_HORIZON
    lags: list[int] = None
    roll_windows: list[int] = None

    def __post_init__(self):
        if self.lags is None:
            self.lags = LAGS
        if self.roll_windows is None:
            self.roll_windows = ROLL_WINDOWS


def _to_month(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", format="%m/%d/%Y")
    fallback_mask = dt.isna()
    if fallback_mask.any():
        dt.loc[fallback_mask] = pd.to_datetime(
            series.loc[fallback_mask], errors="coerce", format="%m/%d/%y"
        )
    return dt.dt.to_period("M").dt.to_timestamp()


def load_filtered_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month"] = _to_month(df["tgl"])
    for col in ["lat", "lon", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["month", "lat", "lon", "mag"]).copy()
    df = df[df["mag"] >= MAG_THRESHOLD].copy()
    df["remark"] = df["remark"].fillna("Unknown").astype(str)
    return df


def _build_time_features(
    df: pd.DataFrame, lags: list[int], roll_windows: list[int]
) -> pd.DataFrame:
    out = df.sort_values(["entity", "month"]).copy()

    for lag in lags:
        out[f"lag_{lag}"] = out.groupby("entity")["count_m5"].shift(lag)

    for w in roll_windows:
        shifted = out.groupby("entity")["count_m5"].shift(1)
        out[f"roll_mean_{w}"] = (
            shifted.groupby(out["entity"])
            .rolling(w)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"roll_std_{w}"] = (
            shifted.groupby(out["entity"])
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    out["roll_std_3"] = out["roll_std_3"].fillna(0.0)
    out["roll_std_6"] = out["roll_std_6"].fillna(0.0)
    out["roll_std_12"] = out["roll_std_12"].fillna(0.0)

    out["month_num"] = out["month"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)
    out = out.drop(columns=["month_num"])
    return out


def build_spatial_panel(
    events: pd.DataFrame, cfg: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = events.copy()
    df["lat_bin"] = np.floor(df["lat"] / GRID_DEG).astype(int)
    df["lon_bin"] = np.floor(df["lon"] / GRID_DEG).astype(int)
    df["entity"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    cells = (
        df[["entity", "lat_bin", "lon_bin"]]
        .drop_duplicates()
        .sort_values(["lat_bin", "lon_bin"])
        .reset_index(drop=True)
    )

    panel_index = pd.MultiIndex.from_product(
        [cells["entity"], months], names=["entity", "month"]
    )
    panel = pd.DataFrame(index=panel_index).reset_index()
    panel = panel.merge(cells, on="entity", how="left")

    counts = (
        df.groupby(["entity", "month"], as_index=False)
        .size()
        .rename(columns={"size": "count_m5"})
    )
    panel = panel.merge(counts, on=["entity", "month"], how="left")
    panel["count_m5"] = panel["count_m5"].fillna(0.0)

    panel = _build_time_features(panel, cfg.lags, cfg.roll_windows)
    panel = panel.dropna(subset=[f"lag_{max(cfg.lags)}"]).reset_index(drop=True)
    return panel, cells


def build_region_panel(
    events: pd.DataFrame, cfg: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = events.copy()
    df["entity"] = df["remark"]

    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    regions = pd.DataFrame({"entity": sorted(df["entity"].unique())})

    panel_index = pd.MultiIndex.from_product(
        [regions["entity"], months], names=["entity", "month"]
    )
    panel = pd.DataFrame(index=panel_index).reset_index()

    counts = (
        df.groupby(["entity", "month"], as_index=False)
        .size()
        .rename(columns={"size": "count_m5"})
    )
    panel = panel.merge(counts, on=["entity", "month"], how="left")
    panel["count_m5"] = panel["count_m5"].fillna(0.0)
    panel["region_code"] = panel["entity"].astype("category").cat.codes

    panel = _build_time_features(panel, cfg.lags, cfg.roll_windows)
    panel = panel.dropna(subset=[f"lag_{max(cfg.lags)}"]).reset_index(drop=True)

    region_map = (
        panel[["entity", "region_code"]].drop_duplicates().sort_values("entity")
    )
    return panel, region_map


def _split_train_test_by_month(
    df: pd.DataFrame, test_horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_months = sorted(df["month"].unique())
    if len(unique_months) <= test_horizon:
        raise ValueError("Jumlah bulan tidak cukup untuk train/test split.")

    cutoff = unique_months[-test_horizon]
    train_df = df[df["month"] < cutoff].copy()
    test_df = df[df["month"] >= cutoff].copy()
    return train_df, test_df


def _fit_regressor(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=400,
        max_depth=8,
        min_samples_leaf=30,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["count_m5"])
    return model


def _evaluate_panel(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    group_cols: list[str],
) -> tuple[dict, pd.DataFrame]:
    model = _fit_regressor(train_df, feature_cols)
    pred = np.clip(model.predict(test_df[feature_cols]), 0.0, None)

    pred_df = test_df[group_cols + ["month", "count_m5"]].copy()
    pred_df["pred_count_m5"] = pred

    mae = mean_absolute_error(pred_df["count_m5"], pred_df["pred_count_m5"])
    rmse = root_mean_squared_error(pred_df["count_m5"], pred_df["pred_count_m5"])

    monthly = (
        pred_df.groupby("month", as_index=False)
        .agg(actual_total=("count_m5", "sum"), pred_total=("pred_count_m5", "sum"))
        .sort_values("month")
    )
    monthly_mae = mean_absolute_error(monthly["actual_total"], monthly["pred_total"])
    monthly_rmse = root_mean_squared_error(
        monthly["actual_total"], monthly["pred_total"]
    )

    metrics = {
        "mae_cell_or_region": float(mae),
        "rmse_cell_or_region": float(rmse),
        "mae_monthly_total": float(monthly_mae),
        "rmse_monthly_total": float(monthly_rmse),
        "n_test_rows": int(len(pred_df)),
    }

    return metrics, pred_df


def _build_feature_columns(
    df: pd.DataFrame, extra_cols: list[str] | None = None
) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith("lag_")
        or c.startswith("roll_mean_")
        or c.startswith("roll_std_")
    ]
    cols += ["month_sin", "month_cos"]
    if extra_cols:
        cols += extra_cols
    return cols


def train_and_save(
    data_path: Path = DATA_PATH, artifact_dir: Path = ARTIFACT_DIR
) -> dict:
    cfg = DatasetConfig()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "saved_models").mkdir(parents=True, exist_ok=True)

    events = load_filtered_events(data_path)

    spatial_panel, cells = build_spatial_panel(events, cfg)
    spatial_features = _build_feature_columns(
        spatial_panel, extra_cols=["lat_bin", "lon_bin"]
    )
    sp_train, sp_test = _split_train_test_by_month(spatial_panel, cfg.test_horizon)
    spatial_metrics, spatial_pred_test = _evaluate_panel(
        sp_train,
        sp_test,
        feature_cols=spatial_features,
        group_cols=["entity", "lat_bin", "lon_bin"],
    )
    spatial_model_full = _fit_regressor(spatial_panel, spatial_features)

    region_panel, region_map = build_region_panel(events, cfg)
    region_features = _build_feature_columns(region_panel, extra_cols=["region_code"])
    rg_train, rg_test = _split_train_test_by_month(region_panel, cfg.test_horizon)
    region_metrics, region_pred_test = _evaluate_panel(
        rg_train,
        rg_test,
        feature_cols=region_features,
        group_cols=["entity", "region_code"],
    )
    region_model_full = _fit_regressor(region_panel, region_features)

    history_spatial = (
        spatial_panel.pivot(index="month", columns="entity", values="count_m5")
        .sort_index()
        .fillna(0.0)
    )
    history_region = (
        region_panel.pivot(index="month", columns="entity", values="count_m5")
        .sort_index()
        .fillna(0.0)
    )

    bundle = {
        "metadata": {
            "mag_threshold": MAG_THRESHOLD,
            "radius_km": RADIUS_KM,
            "grid_deg": GRID_DEG,
            "test_horizon": cfg.test_horizon,
            "lags": cfg.lags,
            "roll_windows": cfg.roll_windows,
            "last_observed_month": str(events["month"].max().date()),
            "n_events_filtered": int(len(events)),
            "n_cells": int(cells.shape[0]),
            "n_regions": int(region_map.shape[0]),
        },
        "spatial": {
            "model": spatial_model_full,
            "feature_cols": spatial_features,
            "cell_lookup": cells,
            "history": history_spatial,
        },
        "region": {
            "model": region_model_full,
            "feature_cols": region_features,
            "region_map": region_map,
            "history": history_region,
        },
    }

    joblib.dump(
        bundle, artifact_dir / "saved_models" / "m5_monthly_radius50_bundle.joblib"
    )

    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, indent=2)

    metrics_df = pd.DataFrame(
        [
            {"scope": "spatial", **spatial_metrics},
            {"scope": "region", **region_metrics},
        ]
    )
    metrics_df.to_csv(artifact_dir / "metrics.csv", index=False)
    spatial_pred_test.to_csv(artifact_dir / "predictions_test_spatial.csv", index=False)
    region_pred_test.to_csv(artifact_dir / "predictions_test_region.csv", index=False)

    summary = {
        "artifact_dir": str(artifact_dir),
        "model_path": str(
            artifact_dir / "saved_models" / "m5_monthly_radius50_bundle.joblib"
        ),
        "spatial_metrics": spatial_metrics,
        "region_metrics": region_metrics,
    }
    return summary


def _build_single_feature(
    history_series: pd.Series,
    target_month: pd.Timestamp,
    lags: list[int],
    roll_windows: list[int],
    extra: dict,
) -> dict:
    feat: dict[str, float] = {}
    target_month = target_month.to_period("M").to_timestamp()

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
    feat.update(extra)
    return feat


def predict_next_month_point(
    bundle: dict,
    lat: float,
    lon: float,
    reference_month: str | None = None,
) -> dict:
    grid_deg = float(bundle["metadata"]["grid_deg"])
    lags = list(bundle["metadata"]["lags"])
    roll_windows = list(bundle["metadata"]["roll_windows"])

    lat_bin = int(np.floor(lat / grid_deg))
    lon_bin = int(np.floor(lon / grid_deg))
    entity = f"{lat_bin}_{lon_bin}"

    history: pd.DataFrame = bundle["spatial"]["history"]
    if reference_month is None:
        ref = (
            pd.Timestamp(bundle["metadata"]["last_observed_month"])
            .to_period("M")
            .to_timestamp()
        )
    else:
        ref = pd.Timestamp(reference_month).to_period("M").to_timestamp()
    target = (ref.to_period("M") + 1).to_timestamp()

    if entity in history.columns:
        series = history[entity]
        known = True
    else:
        series = pd.Series(0.0, index=history.index)
        known = False

    feat = _build_single_feature(
        history_series=series,
        target_month=target,
        lags=lags,
        roll_windows=roll_windows,
        extra={"lat_bin": lat_bin, "lon_bin": lon_bin},
    )
    X = pd.DataFrame([feat])[bundle["spatial"]["feature_cols"]]
    pred = float(np.clip(bundle["spatial"]["model"].predict(X)[0], 0.0, None))

    return {
        "mode": "point_radius50_proxy_cell",
        "input_lat": float(lat),
        "input_lon": float(lon),
        "lat_bin": lat_bin,
        "lon_bin": lon_bin,
        "known_cell": known,
        "reference_month": str(ref.date()),
        "target_month": str(target.date()),
        "pred_count_m5_next_month": pred,
    }


def predict_next_month_region(
    bundle: dict,
    region_name: str,
    reference_month: str | None = None,
) -> dict:
    lags = list(bundle["metadata"]["lags"])
    roll_windows = list(bundle["metadata"]["roll_windows"])

    region_map: pd.DataFrame = bundle["region"]["region_map"]
    exact = region_map[region_map["entity"] == region_name]
    if exact.empty:
        raise ValueError(f"Region tidak ditemukan: {region_name}")

    region_code = int(exact["region_code"].iloc[0])
    history: pd.DataFrame = bundle["region"]["history"]

    if reference_month is None:
        ref = (
            pd.Timestamp(bundle["metadata"]["last_observed_month"])
            .to_period("M")
            .to_timestamp()
        )
    else:
        ref = pd.Timestamp(reference_month).to_period("M").to_timestamp()
    target = (ref.to_period("M") + 1).to_timestamp()

    if region_name in history.columns:
        series = history[region_name]
    else:
        series = pd.Series(0.0, index=history.index)

    feat = _build_single_feature(
        history_series=series,
        target_month=target,
        lags=lags,
        roll_windows=roll_windows,
        extra={"region_code": region_code},
    )
    X = pd.DataFrame([feat])[bundle["region"]["feature_cols"]]
    pred = float(np.clip(bundle["region"]["model"].predict(X)[0], 0.0, None))

    return {
        "mode": "region",
        "region_name": region_name,
        "reference_month": str(ref.date()),
        "target_month": str(target.date()),
        "pred_count_m5_next_month": pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model M>=5 bulanan dengan mode titik radius 50km dan mode region."
    )
    parser.add_argument(
        "--action", choices=["train", "infer_point", "infer_region"], default="train"
    )
    parser.add_argument("--lat", type=float, help="Latitude untuk infer_point")
    parser.add_argument("--lon", type=float, help="Longitude untuk infer_point")
    parser.add_argument("--region", type=str, help="Nama region untuk infer_region")
    parser.add_argument(
        "--reference-month", type=str, default=None, help="Format YYYY-MM"
    )
    parser.add_argument("--artifact-dir", type=str, default=str(ARTIFACT_DIR))
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    model_path = artifact_dir / "saved_models" / "m5_monthly_radius50_bundle.joblib"

    if args.action == "train":
        summary = train_and_save(data_path=DATA_PATH, artifact_dir=artifact_dir)
        print(json.dumps(summary, indent=2))
        return

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model belum ada: {model_path}. Jalankan --action train dulu."
        )
    bundle = joblib.load(model_path)

    if args.action == "infer_point":
        if args.lat is None or args.lon is None:
            raise ValueError("infer_point butuh --lat dan --lon")
        res = predict_next_month_point(
            bundle=bundle,
            lat=args.lat,
            lon=args.lon,
            reference_month=args.reference_month,
        )
        print(json.dumps(res, indent=2))
        return

    if args.action == "infer_region":
        if not args.region:
            raise ValueError("infer_region butuh --region")
        res = predict_next_month_region(
            bundle=bundle,
            region_name=args.region,
            reference_month=args.reference_month,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
