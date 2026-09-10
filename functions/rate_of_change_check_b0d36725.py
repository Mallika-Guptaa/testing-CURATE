"""Self-contained runtime embedded into each built-in FaaSr action.

The generated action receives FaaSr helpers as globals. The deployed function
calls the injected ``faasr_get_file``, ``faasr_put_file``, ``faasr_log``, and
``faasr_secret`` names directly, with no runtime-module dependency.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import statistics
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = ["timestamp", "sensor_id", "vwc_m3_m3"]
OPTIONAL_CANONICAL_COLUMNS = [
    "latitude", "longitude", "depth_cm", "soil_temperature_c", "precipitation_mm",
    "evapotranspiration_mm", "humidity_pct", "sand_pct", "silt_pct", "clay_pct",
    "reference_vwc_m3_m3", "uncertainty", "qc_flag", "gap_fill_flag",
]
SENSITIVE_KEY = re.compile(r"(?i)(secret|password|passphrase|token|api[_-]?key|private[_-]?key)")
SENSITIVE_VALUE = re.compile(r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})")


def _log(message):
    try:
        faasr_log(message)
    except Exception:
        print(message)


def _optional_secret(name):
    try:
        return faasr_secret(name)
    except Exception:
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _redact_export(value):
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if SENSITIVE_KEY.search(str(key)) else _redact_export(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact_export(item) for item in value]
    if isinstance(value, str):
        return SENSITIVE_VALUE.sub("[REDACTED]", value)
    return _json_safe(value)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_local_name(value, fallback="download.dat"):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(str(value)).name).strip(".-")
    return name or fallback


def _software_versions(packages):
    versions = {}
    for package in dict.fromkeys(["pandas", "numpy", "pyarrow", *packages]):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-reported"
    return versions


def _download(folder, remote_name, local_name=None):
    local_name = local_name or Path(remote_name).name
    faasr_get_file(local_file=local_name, remote_folder=folder, remote_file=remote_name)
    return Path(local_name)


def _upload(folder, local_path, remote_name):
    faasr_put_file(local_file=str(local_path), remote_folder=folder, remote_file=remote_name)


def _read_table(path, params=None):
    params = params or {}
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, na_values=params.get("missing_value_markers"))
    if suffix in {".xlsx", ".xls"}:
        sheet = params.get("sheet_name", 0)
        if isinstance(sheet, str) and sheet.isdigit(): sheet = int(sheet)
        return pd.read_excel(path, sheet_name=sheet, na_values=params.get("missing_value_markers"))
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".nc", ".netcdf"}:
        import xarray as xr
        with xr.open_dataset(path, group=params.get("group") or None) as dataset:
            variables = params.get("variables")
            frame = dataset[variables].to_dataframe().reset_index() if variables else dataset.to_dataframe().reset_index()
        return frame
    if suffix in {".h5", ".hdf", ".hdf5"}:
        key = params.get("group")
        try:
            return pd.read_hdf(path, key=key)
        except (ImportError, ValueError, TypeError):
            import h5py
            with h5py.File(path, "r") as handle:
                group = handle[key] if key else handle
                requested = {str(value).lower() for value in params.get("variables", [])}
                arrays = {}

                def collect(name, value):
                    if not isinstance(value, h5py.Dataset) or len(value.shape) not in {1, 2} or not value.size:
                        return
                    basename = Path(name).name.lower()
                    if requested and basename not in requested and not any(item in name.lower() for item in requested):
                        return
                    data = np.asarray(value[()])
                    if np.issubdtype(data.dtype, np.number):
                        data = data.astype(float)
                        fill = value.attrs.get("_FillValue", value.attrs.get("missing_value"))
                        if fill is not None:
                            data[np.isclose(data, float(np.asarray(fill).reshape(-1)[0]))] = np.nan
                        scale = float(np.asarray(value.attrs.get("scale_factor", 1.0)).reshape(-1)[0])
                        offset = float(np.asarray(value.attrs.get("add_offset", 0.0)).reshape(-1)[0])
                        data = data * scale + offset
                    column_name = re.sub(r"[^A-Za-z0-9_]+", "_", Path(name).name).strip("_") or "value"
                    if column_name in arrays:
                        column_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
                    arrays[column_name] = data.reshape(-1)

                if isinstance(group, h5py.Group):
                    group.visititems(collect)
                else:
                    collect(Path(key or "value").name, group)
            if not arrays:
                raise ValueError("The selected HDF5 group contains no readable one- or two-dimensional variables")
            size_counts = Counter(len(value) for value in arrays.values())
            row_count = max(size_counts, key=lambda size: (size_counts[size], size))
            compatible = {name: values for name, values in arrays.items() if len(values) == row_count}
            return pd.DataFrame(compatible)
    if suffix in {".tif", ".tiff"}:
        import rasterio
        from rasterio.warp import transform
        with rasterio.open(path) as source:
            if source.crs is None:
                raise ValueError("GeoTIFF inputs need a declared coordinate reference system")
            band_numbers = params.get("bands") or list(range(1, source.count + 1))
            frames = []
            for band_number in band_numbers:
                values = source.read(int(band_number), masked=True)
                mask = np.ma.getmaskarray(values)
                rows, cols = np.where(~mask)
                xs, ys = rasterio.transform.xy(source.transform, rows, cols)
                longitudes, latitudes = transform(source.crs, "EPSG:4326", list(xs), list(ys))
                frames.append(pd.DataFrame({
                    "longitude": longitudes,
                    "latitude": latitudes,
                    f"band_{band_number}": np.asarray(values[rows, cols]),
                }))
            frame = frames[0]
            for additional in frames[1:]:
                frame = frame.merge(additional, on=["longitude", "latitude"], how="outer")
            return frame
    raise ValueError(f"Unsupported input format: {suffix or 'no extension'}")


def _canonicalize(frame, params, warnings, strict=False):
    result = frame.copy()
    mapping = params.get("column_mapping", {})
    reverse_mapping = {source: target for target, source in mapping.items() if source in result.columns}
    result = result.rename(columns=reverse_mapping)
    for target, aliases in {
        "timestamp": ["datetime", "date_time", "time", "date"],
        "sensor_id": ["sensor", "station", "station_id", "site", "site_id"],
        "vwc_m3_m3": ["vwc", "soil_moisture", "soilmoisture", "swc", "sm"],
        "latitude": ["lat"], "longitude": ["lon", "lng", "long"], "depth_cm": ["depth"],
    }.items():
        if target not in result.columns:
            match = next((column for column in result.columns if str(column).lower() in aliases), None)
            if match:
                result = result.rename(columns={match: target})
    if "sensor_id" not in result.columns:
        result["sensor_id"] = params.get("default_sensor_id", "sensor-1")
        warnings.append("sensor_id was missing; a configured default was used")
    missing_required = [column for column in ["timestamp", "vwc_m3_m3"] if column not in result.columns]
    if missing_required and strict:
        raise ValueError(f"Map the required source field(s) before canonicalization: {', '.join(missing_required)}")
    if missing_required and not strict:
        warnings.append(f"Canonical fields still need mapping or collocation: {', '.join(missing_required)}")
    if "timestamp" in result.columns:
        parsed = pd.to_datetime(result["timestamp"], errors="coerce")
        timezone_name = params.get("timezone", "UTC")
        if getattr(parsed.dt, "tz", None) is None:
            parsed = parsed.dt.tz_localize(timezone_name, ambiguous="NaT", nonexistent="NaT")
        result["timestamp"] = parsed.dt.tz_convert("UTC")
    if "vwc_m3_m3" in result.columns:
        values = pd.to_numeric(result["vwc_m3_m3"], errors="coerce")
        source_unit = params.get("vwc_unit", params.get("source_unit", "m3/m3"))
        if source_unit in {"percent", "%", "volumetric_percent"}:
            values = values / 100.0
        elif source_unit in {"mm/m", "mm_per_m"}:
            values = values / 1000.0
        result["vwc_m3_m3"] = values
    for column in ["latitude", "longitude", "depth_cm"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    ordered = [column for column in CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS if column in result.columns]
    ordered.extend(column for column in result.columns if column not in ordered)
    return result[ordered]


def _numeric_column(frame, params):
    requested = params.get("column", "vwc_m3_m3")
    if requested not in frame.columns:
        numeric = frame.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError("This action needs at least one numeric column")
        return numeric[0]
    return requested


def _grouped(frame, column):
    if "sensor_id" in frame.columns:
        return frame.groupby("sensor_id", group_keys=False, dropna=False)[column]
    return frame[column]


def _robust_z(values):
    median = values.median()
    mad = (values - median).abs().median()
    return 0.6745 * (values - median) / (mad if mad else np.nan)


def _kalman_fill(series, process_variance=1e-5, observation_variance=1e-2):
    observed = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    valid = observed[np.isfinite(observed)]
    if len(valid) == 0:
        return series
    estimate = float(valid[0])
    error = 1.0
    filled = observed.copy()
    for index, measurement in enumerate(observed):
        error += process_variance
        if np.isfinite(measurement):
            gain = error / (error + observation_variance)
            estimate += gain * (measurement - estimate)
            error *= 1 - gain
        else:
            filled[index] = estimate
    return pd.Series(filled, index=series.index)


def _time_features(frame, target_column="vwc_m3_m3"):
    timestamp = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    seconds = timestamp.astype("int64") / 1e9
    seconds = seconds.where(timestamp.notna(), np.nan)
    day = timestamp.dt.dayofyear.fillna(1)
    hour = timestamp.dt.hour.fillna(0)
    features = pd.DataFrame(index=frame.index)
    features["time"] = seconds.replace([np.inf, -np.inf], np.nan)
    features["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    for column in frame.select_dtypes(include="number").columns:
        if column != target_column:
            features[column] = frame[column]
    if "sensor_id" in frame:
        features = pd.concat([features, pd.get_dummies(frame["sensor_id"].astype(str), prefix="sensor", dtype=float)], axis=1)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def _ml_fill(frame, column, method, params):
    features = _time_features(frame, column)
    target = pd.to_numeric(frame[column], errors="coerce")
    observed = target.notna()
    if observed.sum() < max(8, int(params.get("minimum_training_rows", 8))):
        raise ValueError("Not enough observed rows to train the requested gap-filling model")
    if method == "random-forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"), random_state=int(params.get("random_seed", 42)), n_jobs=-1,
        )
        model.fit(features.loc[observed], target.loc[observed])
        target.loc[~observed] = model.predict(features.loc[~observed])
    elif method == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=int(params.get("n_estimators", 250)), max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            random_state=int(params.get("random_seed", 42)), n_jobs=1,
        )
        model.fit(features.loc[observed], target.loc[observed])
        target.loc[~observed] = model.predict(features.loc[~observed])
    else:
        import torch
        from torch import nn
        torch.manual_seed(int(params.get("random_seed", 42)))
        x = torch.tensor(features.to_numpy(dtype=np.float32)).unsqueeze(0)
        y = torch.tensor(target.fillna(target.mean()).to_numpy(dtype=np.float32)).reshape(1, -1, 1)
        mask = torch.tensor(observed.to_numpy()).reshape(1, -1, 1)
        class TemporalLSTM(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)
            def forward(self, values):
                encoded, _ = self.lstm(values)
                return self.head(encoded)
        model = TemporalLSTM(x.shape[2], int(params.get("hidden_size", 24)))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("learning_rate", 0.01)))
        for _ in range(int(params.get("epochs", 150))):
            optimizer.zero_grad()
            prediction = model(x)
            loss = ((prediction[mask] - y[mask]) ** 2).mean()
            loss.backward()
            optimizer.step()
        predictions = model(x).detach().numpy().reshape(-1)
        target.loc[~observed] = predictions[~observed.to_numpy()]
    return target


def _flag(frame, mask, label, treatment="flag", column="vwc_m3_m3"):
    result = frame.copy()
    flag_column = "qc_flag"
    existing = result.get(flag_column, pd.Series("", index=result.index)).fillna("").astype(str)
    result[flag_column] = np.where(mask, existing.str.strip("|") + np.where(existing.eq(""), "", "|") + label, existing)
    if treatment == "remove":
        result = result.loc[~mask].copy()
    elif treatment in {"correct", "reconstruct"} and column in result.columns:
        result.loc[mask, column] = np.nan
        result[column] = result.groupby("sensor_id")[column].transform(lambda values: values.interpolate(limit_direction="both")) if "sensor_id" in result else result[column].interpolate(limit_direction="both")
    return result


def _classical_outlier_mask(slug, values, params):
    """Calculate single-series outlier decisions without mixing sensor baselines."""
    values = pd.to_numeric(values, errors="coerce")
    if slug == "standard-z-score":
        window = int(params.get("window", 0))
        if window > 1:
            center = values.rolling(window, center=True, min_periods=max(3, window // 2)).mean()
            spread = values.rolling(window, center=True, min_periods=max(3, window // 2)).std(ddof=0)
            score = (values - center) / spread.replace(0, np.nan)
        else:
            score = (values - values.mean()) / values.std(ddof=0)
        return score.abs().gt(float(params.get("threshold", 3.0)))
    if slug == "modified-z-score":
        return _robust_z(values).abs().gt(float(params.get("threshold", 3.5)))
    if slug == "hampel-filter":
        window = int(params.get("window", 7))
        center = values.rolling(window, center=True, min_periods=3).median()
        mad = (values - center).abs().rolling(window, center=True, min_periods=3).median()
        return (values - center).abs().gt(float(params.get("threshold", 3.0)) * 1.4826 * mad)
    if slug == "iqr-outliers":
        q1, q3 = values.quantile([0.25, 0.75])
        width = q3 - q1
        multiplier = float(params.get("multiplier", 1.5))
        return values.lt(q1 - multiplier * width) | values.gt(q3 + multiplier * width)
    if slug == "stl-residual-outliers":
        from statsmodels.tsa.seasonal import STL
        period = int(params.get("period", 24))
        if values.notna().sum() < max(period * 2, 7):
            raise ValueError(f"STL outlier detection needs at least {max(period * 2, 7)} observed records per sensor")
        fitted = STL(values.interpolate(limit_direction="both"), period=period, robust=True).fit()
        return _robust_z(pd.Series(fitted.resid, index=values.index)).abs().gt(float(params.get("threshold", 3.5)))
    raise ValueError(f"Unknown classical outlier action: {slug}")


def _apply_action(slug, frame, references, params, warnings, metrics):
    original = frame.copy()
    column = _numeric_column(frame, params) if len(frame.columns) else "vwc_m3_m3"
    treatment = params.get("treatment", "flag")

    if slug in {"load-dataset", "map-columns", "canonicalize-dataset"}:
        frame = _canonicalize(frame, params, warnings, strict=slug == "canonicalize-dataset")
    elif slug == "normalize-timestamps":
        if "timestamp" not in frame:
            raise ValueError("Choose a timestamp column before normalizing time")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["timestamp"].dt.tz is None:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize(params.get("timezone", "UTC"), ambiguous="NaT", nonexistent="NaT")
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC")
    elif slug == "convert-units":
        factor = float(params.get("factor", 1.0))
        offset = float(params.get("offset", 0.0))
        source_unit, target_unit = params.get("source_unit"), params.get("target_unit")
        if source_unit in {"percent", "%"} and target_unit == "m3/m3": factor = 0.01
        if source_unit == "m3/m3" and target_unit in {"percent", "%"}: factor = 100.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor + offset
    elif slug == "aggregate-resample":
        if "timestamp" not in frame:
            raise ValueError("Resampling requires a timestamp")
        frequency = params.get("frequency", "1h")
        method = params.get("method", "mean")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        groups = ["sensor_id"] if "sensor_id" in frame else []
        numeric = frame.select_dtypes(include="number").columns.tolist()
        frame = frame.set_index("timestamp").groupby(groups)[numeric].resample(frequency).agg(method).reset_index() if groups else frame.set_index("timestamp")[numeric].resample(frequency).agg(method).reset_index()
    elif slug == "handle-incomplete-records":
        required = params.get("required_columns", CANONICAL_COLUMNS)
        minimum_fraction = float(params.get("minimum_complete_fraction", 1.0))
        present = [item for item in required if item in frame]
        absent = [item for item in required if item not in frame]
        complete_count = frame[present].notna().sum(axis=1) if present else pd.Series(0, index=frame.index)
        complete_fraction = complete_count / max(len(required), 1)
        if absent: warnings.append(f"Required columns are absent: {', '.join(absent)}")
        frame = _flag(frame, complete_fraction < minimum_fraction, "incomplete_record", params.get("treatment", "flag"), column)
    elif slug == "synchronize-datasets":
        if not references:
            raise ValueError("Synchronization requires a second dataset")
        tolerance = params.get("tolerance", "30min")
        direction = params.get("direction", "nearest")
        left = frame.copy(); right = references[0].copy()
        left["timestamp"] = pd.to_datetime(left["timestamp"], errors="coerce", utc=True)
        right["timestamp"] = pd.to_datetime(right["timestamp"], errors="coerce", utc=True)
        left = left.dropna(subset=["timestamp"]); right = right.dropna(subset=["timestamp"])
        by = "sensor_id" if "sensor_id" in left and "sensor_id" in right else None
        order = ["timestamp", by] if by else ["timestamp"]
        left = left.sort_values(order); right = right.sort_values(order)
        frame = pd.merge_asof(left, right, on="timestamp", by=by, tolerance=pd.Timedelta(tolerance), direction=direction, suffixes=("", "_reference"))
    elif slug == "remove-duplicates":
        requested = params.get("key_columns", ["timestamp", "sensor_id", "depth_cm"])
        subset = [item for item in requested if item in frame]
        if not subset: raise ValueError("Choose at least one duplicate-key column that exists in the dataset")
        absent = [item for item in requested if item not in frame]
        if absent: warnings.append(f"Duplicate keys not present and skipped: {', '.join(absent)}")
        frame = frame.drop_duplicates(subset=subset, keep=params.get("keep", "first"))
    elif slug == "validate-timestamps":
        timestamps = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
        bad = timestamps.isna() | timestamps.duplicated(keep=False)
        if params.get("require_monotonic", True) and len(timestamps):
            bad |= timestamps.diff().lt(pd.Timedelta(0)).fillna(False)
        frame = _flag(frame, bad, "invalid_timestamp", treatment, column)
    elif slug == "check-missing-metadata":
        columns = params.get("metadata_columns", ["sensor_id", "latitude", "longitude", "depth_cm"])
        present = [item for item in columns if item in frame]
        absent = [item for item in columns if item not in frame]
        if absent: warnings.append(f"Metadata columns absent from the dataset: {', '.join(absent)}")
        missing = frame[present].isna().any(axis=1) if present else pd.Series(True, index=frame.index)
        frame = _flag(frame, missing, "missing_metadata", treatment, column)
    elif slug == "soil-texture-range-check":
        texture_thresholds = {
            "sand": (0.03, 0.45), "loamy sand": (0.05, 0.46), "sandy loam": (0.07, 0.48),
            "loam": (0.10, 0.50), "silt loam": (0.12, 0.52), "clay loam": (0.18, 0.55),
            "clay": (0.20, 0.65),
        }
        lower = pd.Series(float(params.get("minimum", 0.0)), index=frame.index)
        upper = pd.Series(float(params.get("maximum", 0.8)), index=frame.index)
        texture_column = params.get("soil_texture_column", "soil_texture")
        if texture_column in frame.columns:
            texture = frame[texture_column].fillna("").astype(str).str.lower()
        else:
            texture = pd.Series(str(params.get("soil_texture", "custom")).lower(), index=frame.index)
        wilting_override = params.get("wilting_point_override")
        porosity_override = params.get("porosity_override")
        for name, (wilting_point, porosity) in texture_thresholds.items():
            selected = texture.eq(name)
            lower.loc[selected] = float(wilting_override if wilting_override is not None else wilting_point)
            upper.loc[selected] = float(porosity_override if porosity_override is not None else porosity)
        if all(item in frame for item in ["sand_pct", "silt_pct", "clay_pct"]):
            texture_total = frame[["sand_pct", "silt_pct", "clay_pct"]].sum(axis=1, min_count=3)
            if texture_total.notna().any() and not texture_total.dropna().between(95, 105).all():
                warnings.append("Some sand/silt/clay percentages do not sum to approximately 100")
            porosity = 0.505 - 0.00142 * frame["sand_pct"] - 0.00037 * frame["clay_pct"]
            upper = np.minimum(upper, porosity + float(params.get("porosity_tolerance", 0.05)))
        bad = frame[column].lt(lower) | frame[column].gt(upper)
        frame = _flag(frame, bad, "soil_texture_range", treatment, column)
    elif slug == "sensor-persistence-check":
        window = int(params.get("window", 12)); tolerance = float(params.get("tolerance", 1e-5))
        rolling = _grouped(frame, column).rolling(window, min_periods=window).std().reset_index(level=0, drop=True) if "sensor_id" in frame else frame[column].rolling(window, min_periods=window).std()
        frame = _flag(frame, rolling.le(tolerance).fillna(False), "sensor_inactive", treatment, column)
    elif slug in {"abrupt-shift-check", "unrealistic-fluctuation-check", "rate-of-change-check"}:
        differences = _grouped(frame, column).diff().abs() if "sensor_id" in frame else frame[column].diff().abs()
        if slug == "rate-of-change-check" and "timestamp" in frame:
            parsed_time = pd.to_datetime(frame["timestamp"], utc=True)
            hours = frame.assign(_parsed_time=parsed_time).groupby("sensor_id", dropna=False)["_parsed_time"].diff().dt.total_seconds().div(3600).replace(0, np.nan) if "sensor_id" in frame else parsed_time.diff().dt.total_seconds().div(3600).replace(0, np.nan)
            differences = differences.div(hours.abs())
        threshold = float(params.get("threshold", 0.15 if slug != "rate-of-change-check" else 0.08))
        frame = _flag(frame, differences.gt(threshold).fillna(False), slug.replace("-check", ""), treatment, column)
    elif slug == "spatial-consistency-check":
        if "timestamp" not in frame:
            raise ValueError("Spatial consistency needs timestamps")
        median = frame.groupby("timestamp")[column].transform("median")
        deviation = (frame[column] - median).abs()
        threshold = float(params.get("threshold", 0.12))
        frame = _flag(frame, deviation.gt(threshold), "spatial_inconsistency", treatment, column)
    elif slug == "climatological-check":
        timestamp = pd.to_datetime(frame["timestamp"], utc=True)
        month = timestamp.dt.month
        center = frame.groupby(month)[column].transform("median")
        spread = frame.groupby(month)[column].transform(lambda values: (values - values.median()).abs().median())
        score = 0.6745 * (frame[column] - center) / spread.replace(0, np.nan)
        frame = _flag(frame, score.abs().gt(float(params.get("threshold", 3.5))).fillna(False), "climatological_outlier", treatment, column)
    elif slug == "detect-missing-values":
        if "timestamp" in frame and params.get("insert_missing_timestamps", True):
            frequency = params.get("expected_frequency", "1h")
            rebuilt = []
            groups = frame.groupby("sensor_id", dropna=False) if "sensor_id" in frame else [(None, frame)]
            for sensor, group in groups:
                group = group.copy(); group["timestamp"] = pd.to_datetime(group["timestamp"], utc=True)
                group = group.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
                if group.empty or group.index.notna().sum() == 0: continue
                group = group.loc[group.index.notna()]
                expected = pd.date_range(group.index.min(), group.index.max(), freq=frequency, tz="UTC")
                group = group.reindex(expected); group.index.name = "timestamp"
                if "sensor_id" in frame: group["sensor_id"] = sensor
                metadata = [item for item in ["latitude", "longitude", "depth_cm", "sand_pct", "silt_pct", "clay_pct"] if item in group]
                group[metadata] = group[metadata].ffill().bfill()
                rebuilt.append(group.reset_index())
            if rebuilt: frame = pd.concat(rebuilt, ignore_index=True)
        missing = frame[column].isna()
        frame["missing_flag"] = missing
        metrics["missing_count"] = int(missing.sum())
        metrics["missing_fraction"] = float(missing.mean()) if len(missing) else 0
    elif slug == "classify-gap-duration":
        missing = frame[column].isna()
        def gap_lengths(values):
            groups = values.ne(values.shift()).cumsum()
            return values.groupby(groups).transform("sum")
        length = missing.groupby(frame["sensor_id"], dropna=False).transform(gap_lengths) if "sensor_id" in frame else gap_lengths(missing)
        short = int(params.get("short_max_records", 3)); medium = int(params.get("medium_max_records", 12))
        frame["gap_class"] = np.where(~missing, "observed", np.where(length <= short, "short", np.where(length <= medium, "medium", "long")))
    elif slug in {"linear-interpolation", "spline-interpolation", "kalman-filter", "random-forest-gap-fill", "xgboost-gap-fill", "lstm-gap-fill"}:
        before_missing = frame[column].isna()
        limit = params.get("maximum_gap_records")
        if slug == "linear-interpolation":
            filled = _grouped(frame, column).transform(lambda values: values.interpolate(method="linear", limit=limit, limit_direction="both")) if "sensor_id" in frame else frame[column].interpolate(method="linear", limit=limit, limit_direction="both")
        elif slug == "spline-interpolation":
            order = int(params.get("order", 3))
            filled = _grouped(frame, column).transform(lambda values: values.interpolate(method="spline", order=order, limit=limit, limit_direction="both")) if "sensor_id" in frame else frame[column].interpolate(method="spline", order=order, limit=limit, limit_direction="both")
        elif slug == "kalman-filter":
            process_variance = float(params.get("process_variance", 1e-5))
            observation_variance = float(params.get("observation_variance", 1e-2))
            filled = _grouped(frame, column).transform(lambda values: _kalman_fill(values, process_variance, observation_variance)) if "sensor_id" in frame else _kalman_fill(frame[column], process_variance, observation_variance)
        else:
            method = {"random-forest-gap-fill": "random-forest", "xgboost-gap-fill": "xgboost", "lstm-gap-fill": "lstm"}[slug]
            filled = _ml_fill(frame, column, method, params)
        frame[column] = filled
        frame["gap_fill_flag"] = np.where(before_missing & frame[column].notna(), slug, frame.get("gap_fill_flag", ""))
    elif slug in {"standard-z-score", "modified-z-score", "hampel-filter", "iqr-outliers", "stl-residual-outliers", "isolation-forest", "local-outlier-factor", "dbscan-outliers"}:
        values = pd.to_numeric(frame[column], errors="coerce")
        if slug in {"standard-z-score", "modified-z-score", "hampel-filter", "iqr-outliers", "stl-residual-outliers"}:
            if "sensor_id" in frame:
                mask = frame.groupby("sensor_id", dropna=False)[column].transform(
                    lambda series: _classical_outlier_mask(slug, series, params)
                ).astype(bool)
            else:
                mask = _classical_outlier_mask(slug, values, params)
        else:
            feature_columns = params.get("feature_columns") or [column]
            matrix = frame[feature_columns].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").fillna(0)
            if slug == "isolation-forest":
                from sklearn.ensemble import IsolationForest
                labels = IsolationForest(contamination=float(params.get("contamination", 0.02)), random_state=int(params.get("random_seed", 42))).fit_predict(matrix)
            elif slug == "local-outlier-factor":
                from sklearn.neighbors import LocalOutlierFactor
                labels = LocalOutlierFactor(n_neighbors=int(params.get("neighbors", 20)), contamination=float(params.get("contamination", 0.02))).fit_predict(matrix)
            else:
                from sklearn.cluster import DBSCAN
                labels = DBSCAN(eps=float(params.get("eps", 0.05)), min_samples=int(params.get("minimum_samples", 5))).fit_predict(matrix)
            mask = pd.Series(labels == -1, index=frame.index)
        frame = _flag(frame, mask.fillna(False), slug, treatment, column)
        metrics["outlier_count"] = int(mask.fillna(False).sum())
    elif slug == "apply-outlier-treatment":
        flag_name = params.get("flag_column", "qc_flag")
        mask = frame[flag_name].fillna("").astype(str).ne("") if flag_name in frame else pd.Series(False, index=frame.index)
        frame = _flag(frame, mask, "treated", params.get("treatment", "reconstruct"), column)
    elif slug == "collocate-reference-data":
        if not references:
            raise ValueError("Collocation requires a reference dataset")
        right = references[0].copy(); frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
        right = right.rename(columns={params.get("reference_column", "vwc_m3_m3"): "reference_vwc_m3_m3"})
        by = "sensor_id" if "sensor_id" in frame and "sensor_id" in right else None
        order = ["timestamp", by] if by else ["timestamp"]
        frame = pd.merge_asof(
            frame.dropna(subset=["timestamp"]).sort_values(order), right.dropna(subset=["timestamp"]).sort_values(order), on="timestamp", by=by,
            tolerance=pd.Timedelta(params.get("time_tolerance", "12h")), direction="nearest", suffixes=("", "_reference")
        )
    elif slug == "spatial-collocation":
        if not references: raise ValueError("Spatial collocation requires a reference dataset")
        right = references[0].copy(); required_coordinates = {"latitude", "longitude"}
        if not required_coordinates.issubset(frame.columns) or not required_coordinates.issubset(right.columns):
            raise ValueError("Spatial collocation requires latitude and longitude in both datasets")
        reference_column = params.get("reference_column", "band_1")
        if reference_column not in right:
            numeric = [name for name in right.select_dtypes(include="number") if name not in {"latitude", "longitude"}]
            if not numeric: raise ValueError("Choose a numeric reference variable or raster band")
            reference_column = numeric[0]; warnings.append(f"Reference column was unavailable; used {reference_column}")
        output_column = params.get("output_column", "reference_vwc_m3_m3")
        maximum_distance = float(params.get("maximum_distance_km", 25))
        time_tolerance = pd.Timedelta(params.get("time_tolerance", "12h"))
        if "timestamp" in frame: frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if "timestamp" in right: right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
        matched_values = []; matched_distances = []; matched_time_deltas = []
        for _, observation in frame.iterrows():
            candidates = right.dropna(subset=["latitude", "longitude"])
            if pd.isna(observation.get("latitude")) or pd.isna(observation.get("longitude")):
                matched_values.append(np.nan); matched_distances.append(np.nan); matched_time_deltas.append(np.nan); continue
            if "timestamp" in frame and "timestamp" in right and pd.notna(observation.get("timestamp")):
                time_delta = (candidates["timestamp"] - observation["timestamp"]).abs()
                candidates = candidates.loc[time_delta.le(time_tolerance)]
            if candidates.empty:
                matched_values.append(np.nan); matched_distances.append(np.nan); matched_time_deltas.append(np.nan); continue
            lat1 = np.radians(float(observation["latitude"])); lon1 = np.radians(float(observation["longitude"]))
            lat2 = np.radians(pd.to_numeric(candidates["latitude"], errors="coerce")); lon2 = np.radians(pd.to_numeric(candidates["longitude"], errors="coerce"))
            haversine = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
            distances = 6371.0088 * 2 * np.arcsin(np.sqrt(haversine.clip(0, 1)))
            nearest_index = distances.idxmin(); distance = float(distances.loc[nearest_index])
            if distance > maximum_distance:
                matched_values.append(np.nan); matched_distances.append(distance); matched_time_deltas.append(np.nan); continue
            matched_values.append(candidates.loc[nearest_index, reference_column]); matched_distances.append(distance)
            matched_time_deltas.append(abs((candidates.loc[nearest_index, "timestamp"] - observation["timestamp"]).total_seconds()) / 3600 if "timestamp" in candidates and pd.notna(observation.get("timestamp")) else np.nan)
        frame[output_column] = matched_values; frame["collocation_distance_km"] = matched_distances; frame["collocation_time_difference_hours"] = matched_time_deltas
        metrics["collocated_count"] = int(frame[output_column].notna().sum()); metrics["unmatched_count"] = int(frame[output_column].isna().sum())
    elif slug == "detect-bias":
        reference = params.get("reference_column", "reference_vwc_m3_m3")
        paired = frame[[column, reference]].apply(pd.to_numeric, errors="coerce").dropna()
        if paired.empty:
            raise ValueError("Bias detection needs at least one paired sensor and reference observation")
        difference = paired[column] - paired[reference]
        metrics.update({"mean_bias": float(difference.mean()), "median_bias": float(difference.median()), "bias_std": float(difference.std())})
        frame["bias"] = np.nan; frame.loc[difference.index, "bias"] = difference
    elif slug in {"additive-bias-correction", "multiplicative-bias-correction", "linear-regression-correction", "quantile-mapping"}:
        reference = params.get("reference_column", "reference_vwc_m3_m3")
        paired = frame[[column, reference]].dropna()
        if paired.empty:
            raise ValueError("Bias correction needs paired sensor and reference values")
        if slug == "additive-bias-correction":
            frame[column] = frame[column] - (paired[column] - paired[reference]).mean()
        elif slug == "multiplicative-bias-correction":
            if math.isclose(float(paired[column].mean()), 0.0, abs_tol=1e-15):
                raise ValueError("Multiplicative correction cannot use a zero sensor mean")
            ratio = paired[reference].mean() / paired[column].mean()
            frame[column] = frame[column] * ratio
        elif slug == "linear-regression-correction":
            slope, intercept = np.polyfit(paired[column], paired[reference], 1)
            frame[column] = intercept + slope * frame[column]
            metrics.update({"correction_slope": float(slope), "correction_intercept": float(intercept)})
        else:
            quantiles = np.linspace(0, 1, int(params.get("quantiles", 101)))
            source_q = paired[column].quantile(quantiles).to_numpy(); target_q = paired[reference].quantile(quantiles).to_numpy()
            frame[column] = np.interp(frame[column], source_q, target_q)
    elif slug in {"moving-average", "savitzky-golay", "low-pass-filter"}:
        window = int(params.get("window", 7)); window += 1 - window % 2
        if slug == "moving-average":
            frame[column] = _grouped(frame, column).transform(lambda values: values.rolling(window, center=True, min_periods=1).mean()) if "sensor_id" in frame else frame[column].rolling(window, center=True, min_periods=1).mean()
        elif slug == "savitzky-golay":
            from scipy.signal import savgol_filter
            polyorder = min(int(params.get("polyorder", 2)), window - 1)
            frame[column] = _grouped(frame, column).transform(lambda values: savgol_filter(values.interpolate(limit_direction="both"), window, polyorder, mode="interp")) if "sensor_id" in frame else savgol_filter(frame[column].interpolate(limit_direction="both"), window, polyorder, mode="interp")
        else:
            from scipy.signal import butter, filtfilt
            cutoff = float(params.get("normalized_cutoff", 0.1)); order = int(params.get("order", 3))
            b, a = butter(order, cutoff, btype="low")
            frame[column] = _grouped(frame, column).transform(lambda values: filtfilt(b, a, values.interpolate(limit_direction="both"))) if "sensor_id" in frame else filtfilt(b, a, frame[column].interpolate(limit_direction="both"))
    elif slug in {"pearson-correlation", "spearman-correlation"}:
        columns = params.get("columns") or frame.select_dtypes(include="number").columns.tolist()
        method = "pearson" if slug.startswith("pearson") else "spearman"
        matrix = frame[columns].corr(method=method)
        metrics["correlation_method"] = method
        metrics["correlation_matrix"] = _json_safe(matrix.to_dict())
    elif slug in {"simple-regression", "multiple-regression"}:
        import statsmodels.api as sm
        response = params.get("response", column)
        predictors = params.get("predictors") or ([params.get("predictor", "reference_vwc_m3_m3")] if slug == "simple-regression" else [])
        if not predictors:
            predictors = [item for item in frame.select_dtypes(include="number") if item != response][:3]
        model_data = frame[[response] + predictors].dropna()
        fitted = sm.OLS(model_data[response], sm.add_constant(model_data[predictors])).fit()
        frame.loc[model_data.index, "regression_prediction"] = fitted.predict(sm.add_constant(model_data[predictors]))
        metrics.update({"r2": float(fitted.rsquared), "adjusted_r2": float(fitted.rsquared_adj), "coefficients": _json_safe(fitted.params.to_dict()), "p_values": _json_safe(fitted.pvalues.to_dict())})
    elif slug == "regression-diagnostics":
        observed = frame[params.get("observed_column", column)]
        predicted = frame[params.get("predicted_column", "regression_prediction")]
        residual = observed - predicted
        frame["residual"] = residual
        metrics.update({"residual_mean": float(residual.mean()), "residual_std": float(residual.std()), "durbin_watson": float(np.sum(np.diff(residual.dropna()) ** 2) / np.sum(residual.dropna() ** 2)) if residual.notna().sum() > 2 else None})
    elif slug == "confidence-intervals":
        level = float(params.get("level", 0.95)); from scipy.stats import t
        count = frame[column].notna().sum(); mean = frame[column].mean(); sem = frame[column].std() / math.sqrt(max(count, 1))
        margin = t.ppf((1 + level) / 2, max(count - 1, 1)) * sem
        metrics.update({"mean": float(mean), "confidence_level": level, "lower": float(mean - margin), "upper": float(mean + margin)})
    elif slug == "residual-analysis":
        observed = frame[params.get("observed_column", column)]; predicted = frame[params.get("predicted_column", "reference_vwc_m3_m3")]
        frame["residual"] = observed - predicted
        metrics.update({"residual_mean": float(frame["residual"].mean()), "residual_rmse": float(np.sqrt(np.nanmean(frame["residual"] ** 2)))})
    elif slug == "validation-metrics":
        observed = pd.to_numeric(frame[params.get("observed_column", column)], errors="coerce"); reference = pd.to_numeric(frame[params.get("reference_column", "reference_vwc_m3_m3")], errors="coerce")
        valid = observed.notna() & reference.notna(); error = observed[valid] - reference[valid]
        if not valid.any():
            raise ValueError("Validation metrics need at least one paired sensor and reference observation")
        centered = error - error.mean()
        ss_res = float(np.sum(error ** 2)); ss_tot = float(np.sum((reference[valid] - reference[valid].mean()) ** 2))
        metrics.update({
            "n": int(valid.sum()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mbe": float(np.mean(error)),
            "r2": 1 - ss_res / ss_tot if ss_tot else None,
            "ubrmse": float(np.sqrt(np.mean(centered ** 2))),
        })
    elif slug == "time-holdout-validation":
        fraction = float(params.get("holdout_fraction", 0.2))
        ordered = frame.sort_values("timestamp") if "timestamp" in frame else frame
        groups = ordered.groupby("sensor_id", dropna=False) if "sensor_id" in ordered else [(None, ordered)]
        errors = []
        for _, group in groups:
            if len(group) < 2: continue
            split = min(len(group) - 1, max(1, int(len(group) * (1 - fraction))))
            train, holdout = group.iloc[:split], group.iloc[split:]
            baseline = pd.to_numeric(train[column], errors="coerce").mean()
            errors.append(pd.to_numeric(holdout[column], errors="coerce") - baseline)
        if not errors:
            raise ValueError("Time-based holdout validation needs at least two observations in a sensor series")
        error = pd.concat(errors)
        if not error.notna().any():
            raise ValueError("Time-based holdout validation has no numeric holdout observations")
        metrics.update({"holdout_rows": int(error.notna().sum()), "baseline_rmse": float(np.sqrt(np.nanmean(error ** 2)))})
    elif slug in {"rainfall-response-check", "dry-down-check", "seasonal-consistency-check", "irrigation-response-check"}:
        if slug in {"rainfall-response-check", "irrigation-response-check"}:
            driver = params.get("driver_column", "precipitation_mm" if slug.startswith("rain") else "irrigation_mm")
            threshold = float(params.get("event_threshold", 1.0)); lag = int(params.get("response_lag_records", 3))
            future = frame.groupby("sensor_id", dropna=False)[column].shift(-lag) if "sensor_id" in frame else frame[column].shift(-lag)
            delta = future - frame[column]
            events = frame[driver].fillna(0).ge(threshold)
            minimum_response = float(params.get("minimum_response", 0))
            evaluable = events & delta.notna()
            metrics.update({
                "event_count": int(events.sum()), "evaluable_event_count": int(evaluable.sum()),
                "positive_response_fraction": float(delta[evaluable].gt(minimum_response).mean()) if evaluable.any() else None,
            })
            flag_column = slug.removesuffix("-check").replace("-", "_") + "_flag"
            frame[flag_column] = evaluable & delta.le(minimum_response)
        elif slug == "dry-down-check":
            precipitation = frame.get(params.get("precipitation_column", "precipitation_mm"), pd.Series(0, index=frame.index)).fillna(0)
            dry = precipitation.le(float(params.get("rain_threshold", 0.1)))
            differences = _grouped(frame, column).diff() if "sensor_id" in frame else frame[column].diff()
            increases = differences.gt(float(params.get("increase_tolerance", 0.02)))
            frame["dry_down_flag"] = dry & increases
            metrics["dry_down_violation_count"] = int(frame["dry_down_flag"].sum())
        else:
            month = pd.to_datetime(frame["timestamp"], utc=True).dt.month
            monthly = frame.groupby(month)[column].agg(["mean", "std", "count"])
            metrics["monthly_summary"] = _json_safe(monthly.to_dict(orient="index"))
    elif slug == "uncertainty-summary":
        uncertainty_column = params.get("uncertainty_column", "uncertainty")
        if uncertainty_column in frame:
            metrics.update({"mean_uncertainty": float(frame[uncertainty_column].mean()), "p95_uncertainty": float(frame[uncertainty_column].quantile(0.95))})
        else:
            metrics.update({"standard_error": float(frame[column].std() / math.sqrt(max(frame[column].notna().sum(), 1)))})
    elif slug in {"export-results", "create-result-bundle"}:
        frame = frame.copy()
    else:
        raise ValueError(f"Unknown scientific action: {slug}")
    metrics.setdefault("rows_before", int(len(original)))
    metrics.setdefault("rows_after", int(len(frame)))
    metrics.setdefault("rows_removed", int(max(0, len(original) - len(frame))))
    metrics.setdefault("rows_added", int(max(0, len(frame) - len(original))))
    if "qc_flag" in frame:
        flagged = frame["qc_flag"].fillna("").astype(str).ne("")
        # For a remove treatment, removed rows represent the failed decisions
        # that are no longer available to carry a flag in the result table.
        flagged_count = int(flagged.sum()) + (int(max(0, len(original) - len(frame))) if treatment == "remove" else 0)
        metrics.setdefault("flagged_count", flagged_count)
        metrics.setdefault("flagged_fraction", flagged_count / max(len(original), 1))
    if "gap_fill_flag" in frame:
        filled = frame["gap_fill_flag"].fillna("").astype(str).ne("")
        metrics.setdefault("gap_filled_count", int(filled.sum()))
        metrics.setdefault("gap_filled_fraction", float(filled.mean()) if len(filled) else 0.0)
    return frame, original


def _provider_frame(slug, params, warnings):
    if slug == "nasa-power-connector":
        import requests
        parameters = ",".join(params.get("parameters", ["PRECTOTCORR", "T2M", "RH2M"]))
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        response = requests.get(url, params={"parameters": parameters, "community": params.get("community", "AG"), "longitude": params["longitude"], "latitude": params["latitude"], "start": params["start"].replace("-", ""), "end": params["end"].replace("-", ""), "format": "JSON"}, timeout=120)
        response.raise_for_status(); payload = response.json()["properties"]["parameter"]
        frame = pd.DataFrame(payload); frame.index.name = "timestamp"; return frame.reset_index()
    if slug == "noaa-cdo-connector":
        import requests
        token = faasr_secret("NOAA_CDO_TOKEN")
        response = requests.get("https://www.ncei.noaa.gov/cdo-web/api/v2/data", headers={"token": token}, params={"datasetid": params["dataset_id"], "stationid": params.get("station_id"), "startdate": params["start"], "enddate": params["end"], "limit": min(int(params.get("limit", 1000)), 1000), "units": params.get("units", "metric")}, timeout=120)
        response.raise_for_status(); values = response.json().get("results", []); return pd.DataFrame(values)
    if slug == "s3-object-connector":
        import boto3
        target = Path(_safe_local_name(params.get("filename") or Path(params["key"]).name))
        client = boto3.client("s3", endpoint_url=params.get("endpoint_url"), aws_access_key_id=faasr_secret("AWS_ACCESS_KEY_ID") if params.get("use_studio_credentials", True) else None, aws_secret_access_key=faasr_secret("AWS_SECRET_ACCESS_KEY") if params.get("use_studio_credentials", True) else None, aws_session_token=_optional_secret("AWS_SESSION_TOKEN") if params.get("use_studio_credentials", True) else None, region_name=params.get("region"))
        client.download_file(params["bucket"], params["key"], str(target)); return _read_table(target, params)
    if slug in {"public-url-connector", "authenticated-url-connector", "esa-cci-connector"}:
        import requests
        if slug == "esa-cci-connector" and not params.get("usage_terms_acknowledged"):
            raise ValueError("Acknowledge the ESA CCI product citation and usage terms before downloading")
        headers = {}
        if slug != "public-url-connector":
            secret_name = params.get("token_secret", "DATA_PROVIDER_TOKEN")
            headers[params.get("token_header", "Authorization")] = params.get("token_prefix", "Bearer ") + faasr_secret(secret_name)
        response = requests.get(params["url"], headers=headers, timeout=300, stream=True); response.raise_for_status()
        filename = _safe_local_name(params.get("filename") or Path(params["url"].split("?", 1)[0]).name, "download.csv")
        target = Path(filename)
        with target.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024): handle.write(chunk)
        return _read_table(target, params)
    if slug == "smap-earthdata-connector":
        import earthaccess
        earthaccess.login(strategy=params.get("login_strategy", "environment"), persist=False)
        results = earthaccess.search_data(short_name=params.get("short_name", "SPL3SMP_E"), temporal=(params["start"], params["end"]), bounding_box=tuple(params["bounding_box"]) if params.get("bounding_box") else None, count=int(params.get("count", 10)))
        files = earthaccess.download(results, local_path="earthdata")
        frames = []
        for file in files:
            decoded = _read_table(Path(file), params)
            decoded["source_granule"] = Path(file).name
            match = re.search(r"(20\d{6})(?:T(\d{6}))?", Path(file).name)
            if match:
                decoded["timestamp"] = pd.to_datetime(
                    match.group(1) + (match.group(2) or "000000"), format="%Y%m%d%H%M%S", utc=True
                )
            frames.append(decoded)
        if not frames:
            raise ValueError("Earthdata returned no downloadable SMAP granules for this request")
        return pd.concat(frames, ignore_index=True)
    if slug == "soilgrids-connector":
        import requests
        if params.get("access_method", "wcs") == "webdav":
            url = str(params.get("webdav_url") or "").strip()
            if not url.startswith(("https://", "http://")):
                raise ValueError("Provide the complete SoilGrids WebDAV GeoTIFF URL")
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            target = Path("soilgrids-webdav.tif")
            with target.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            return _read_table(target, params)
        property_name = params.get("property", "clay"); depth = params.get("depth", "0-5cm"); statistic = params.get("statistic", "mean")
        coverage = f"{property_name}_{depth}_{statistic}"
        base = params.get("wcs_url", "https://maps.isric.org/mapserv")
        response = requests.get(base, params={"map": f"/map/{property_name}.map", "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage", "COVERAGEID": coverage, "SUBSET": [f"X({params['west']},{params['east']})", f"Y({params['south']},{params['north']})"], "FORMAT": "image/tiff"}, timeout=300)
        response.raise_for_status(); target = Path("soilgrids.tif"); target.write_bytes(response.content); return _read_table(target, params)
    if slug == "era5-land-connector":
        if not params.get("terms_accepted"):
            raise ValueError("Accept the selected Copernicus dataset terms before requesting ERA5-Land")
        import cdsapi
        token = faasr_secret("CDS_API_TOKEN")
        os.environ["CDSAPI_KEY"] = token
        target = "era5-land.nc"
        cdsapi.Client(url=params.get("api_url", "https://cds.climate.copernicus.eu/api"), key=token, quiet=True).retrieve(params.get("dataset", "reanalysis-era5-land"), params["request"], target)
        return _read_table(Path(target), params)
    raise ValueError(f"Unknown connector: {slug}")


def _summary(frame, before, metrics, warnings):
    numeric = frame.select_dtypes(include="number")
    return {
        "rows_before": int(len(before)), "rows_after": int(len(frame)),
        "columns": list(frame.columns), "warnings": warnings, "metrics": metrics,
        "missing_before": int(before.isna().sum().sum()), "missing_after": int(frame.isna().sum().sum()),
        "numeric_summary": _json_safe(numeric.describe().to_dict()) if not numeric.empty else {},
    }


def _vega_spec(frame, before, action_title, params, settings=None):
    settings = settings or {}
    x = "timestamp" if "timestamp" in frame else None
    y = params.get("column", "vwc_m3_m3")
    if y not in frame:
        numeric = frame.select_dtypes(include="number").columns.tolist(); y = numeric[0] if numeric else None
    if not y:
        return {"$schema": "https://vega.github.io/schema/vega-lite/v6.json", "description": "No numeric data to chart", "data": {"values": []}, "mark": "text", "encoding": {"text": {"value": "No numeric data"}}}
    values = []
    limit = int(params.get("visualization_max_points", 2000))
    stages = [("Before", before), ("After", frame)] if settings.get("show_before_after", True) else [("Result", frame)]
    for stage, source in stages:
        chosen = source.iloc[::max(1, math.ceil(len(source) / max(1, limit // 2)))]
        for _, row in chosen.iterrows():
            sensor = str(row.get("sensor_id", "All sensors"))
            depth = row.get("depth_cm")
            if depth is None or bool(pd.isna(depth)):
                series = sensor
            else:
                depth_label = f"{float(depth):g}" if isinstance(depth, (int, float, np.integer, np.floating)) else str(depth)
                series = f"{sensor} · {depth_label} cm"
            point = {"stage": stage, "value": _json_safe(row.get(y)), "sensor_id": sensor, "series": series}
            if "depth_cm" in row:
                point["depth_cm"] = _json_safe(row.get("depth_cm"))
            point["timestamp"] = _json_safe(row.get(x)) if x else int(_)
            values.append(point)
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json", "title": action_title,
        "data": {"values": values}, "mark": {"type": settings.get("chart_type", "line"), "tooltip": True, "clip": True},
        "encoding": {
            "x": {"field": "timestamp", "type": "temporal" if x else "quantitative", "title": "Time" if x else "Record"},
            "y": {"field": "value", "type": "quantitative", "title": y},
            "color": {"field": "stage", "type": "nominal", "scale": {"range": ["#b46f48", "#39756c"]}},
            "detail": {"field": "series", "type": "nominal"}, "opacity": {"value": 0.86},
        }, "width": "container", "height": 360,
    }


def _write_png(frame, before, params, target, settings=None):
    settings = settings or {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    column = params.get("column", "vwc_m3_m3")
    if column not in frame:
        numeric = frame.select_dtypes(include="number").columns.tolist(); column = numeric[0] if numeric else None
    fig, axis = plt.subplots(figsize=(10, 4.8), dpi=150)
    if column:
        points = settings.get("chart_type", "line") == "point"
        draw = axis.scatter if points else axis.plot

        def draw_stage(source, stage, color, alpha, draw_options):
            if column not in source:
                return
            group_columns = [item for item in ["sensor_id", "depth_cm"] if item in source]
            groups = source.groupby(group_columns, dropna=False, sort=False) if group_columns else [(None, source)]
            for group_index, (_, group) in enumerate(groups):
                if "timestamp" in group:
                    group = group.sort_values("timestamp")
                    x_values = pd.to_datetime(group["timestamp"], utc=True)
                else:
                    x_values = group.index
                draw(
                    x_values, group[column], color=color, alpha=alpha,
                    label=stage if group_index == 0 else "_nolegend_", **draw_options,
                )

        if settings.get("show_before_after", True):
            draw_stage(before, "Before", "#b46f48", .55, {"s": 8} if points else {"linewidth": 1.0})
        draw_stage(frame, "After", "#39756c", .9, {"s": 9} if points else {"linewidth": 1.5})
        axis.set_ylabel(column); axis.legend(frameon=False)
    else:
        axis.text(.5, .5, "No numeric data to chart", ha="center", va="center")
    axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(target, bbox_inches="tight"); plt.close(fig)


def _geojson(frame, settings=None):
    settings = settings or {}
    if not settings.get("map_enabled", True):
        return {"type": "FeatureCollection", "features": []}
    if not {"latitude", "longitude"}.issubset(frame.columns):
        return {"type": "FeatureCollection", "features": []}
    requested_value = settings.get("map_value", "vwc_m3_m3")
    columns = [item for item in ["sensor_id", "timestamp", requested_value, "depth_cm", "qc_flag"] if item in frame]
    columns = list(dict.fromkeys(columns))
    features = []
    for _, row in frame.dropna(subset=["latitude", "longitude"]).head(10000).iterrows():
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(row["longitude"]), float(row["latitude"])]}, "properties": _json_safe({column: row[column] for column in columns})})
    return {"type": "FeatureCollection", "features": features}


def _execute(action_slug, node_id, action_title, folder, input_names, output_names):
    started = datetime.now(timezone.utc)
    local_inputs = []
    for index, remote_name in enumerate(input_names):
        local_inputs.append(_download(folder, remote_name, f"input-{index}-{Path(remote_name).name}"))
    config_path = next((path for path in local_inputs if path.name.endswith("workflow-config.json")), None)
    configuration = json.loads(config_path.read_text()) if config_path else {}
    params = configuration.get("nodes", {}).get(node_id, configuration.get("parameters", {}))
    visualization_settings = configuration.get("visualization_settings", {}).get(node_id, {})
    data_inputs = [path for path in local_inputs if path != config_path]
    warnings = []
    metrics = {}
    if action_slug.endswith("-connector"):
        frame = _provider_frame(action_slug, params, warnings)
        before = frame.copy()
    else:
        if not data_inputs:
            raise ValueError(f"{action_title} needs an input dataset")
        frames = [_read_table(path, params) for path in data_inputs]
        frame, before = _apply_action(action_slug, frames[0], frames[1:], params, warnings, metrics)
    summary = _summary(frame, before, metrics, warnings)
    generated = {}
    if "data_parquet" in output_names:
        path = Path(output_names["data_parquet"]); frame.to_parquet(path, index=False); generated["data_parquet"] = path
    if "data_csv" in output_names:
        path = Path(output_names["data_csv"]); frame.to_csv(path, index=False); generated["data_csv"] = path
    if "data_netcdf" in output_names:
        import xarray as xr
        path = Path(output_names["data_netcdf"]); xr.Dataset.from_dataframe(frame.reset_index(drop=True)).to_netcdf(path); generated["data_netcdf"] = path
    if "summary_json" in output_names:
        path = Path(output_names["summary_json"]); path.write_text(json.dumps(_json_safe(summary), indent=2)); generated["summary_json"] = path
    if "visualization_json" in output_names:
        path = Path(output_names["visualization_json"]); path.write_text(json.dumps(_vega_spec(frame, before, action_title, params, visualization_settings), indent=2)); generated["visualization_json"] = path
    if "visualization_png" in output_names:
        path = Path(output_names["visualization_png"]); _write_png(frame, before, params, path, visualization_settings); generated["visualization_png"] = path
    if "map_geojson" in output_names:
        path = Path(output_names["map_geojson"]); path.write_text(json.dumps(_geojson(frame, visualization_settings), indent=2)); generated["map_geojson"] = path
    provenance = {
        "schema_version": "1.0", "action": action_slug, "node_id": node_id, "title": action_title,
        "action_version": configuration.get("action_versions", {}).get(node_id),
        "workflow": configuration.get("workflow", {}), "run": configuration.get("run", {}),
        "started_at": started.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat(),
        "parameters": _redact_export(params), "source_inputs": [{"name": path.name, "sha256": _sha256(path)} for path in data_inputs],
        "outputs": {}, "row_counts": {"before": len(before), "after": len(frame)}, "warnings": warnings,
        "software": _software_versions(configuration.get("software_dependencies", {}).get(node_id, [])),
        "configuration_sha256": configuration.get("sha256") if config_path else None,
        "configuration_file_sha256": _sha256(config_path) if config_path else None,
        "dataset_bindings": _redact_export(configuration.get("dataset_bindings", {}).get(node_id, {})),
        "citations": configuration.get("citations", {}).get(node_id, []),
    }
    for key, path in generated.items(): provenance["outputs"][key] = {"name": output_names[key], "sha256": _sha256(path)}
    if "provenance_json" in output_names:
        path = Path(output_names["provenance_json"]); path.write_text(json.dumps(_json_safe(provenance), indent=2)); generated["provenance_json"] = path
    if "processing_report" in output_names:
        path = Path(output_names["processing_report"])
        metric_lines = "\n".join(f"- {name}: {value}" for name, value in metrics.items()) or "- No scalar metrics were produced by this action."
        warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None"
        citation_lines = "\n".join(f"- {item.get('title')}: {item.get('url') or item.get('doi') or item.get('note', '')}" for item in provenance["citations"]) or "- No external method citation was declared."
        path.write_text(
            f"# {action_title}\n\n## Processing summary\n\n- Action: `{action_slug}`\n- Rows before: {len(before)}\n- Rows after: {len(frame)}\n- Started: {started.isoformat()}\n- Completed: {provenance['completed_at']}\n\n## Parameters\n\n```json\n{json.dumps(_redact_export(params), indent=2)}\n```\n\n## Metrics\n\n{metric_lines}\n\n## Warnings\n\n{warning_lines}\n\n## Citations\n\n{citation_lines}\n",
            encoding="utf-8",
        )
        generated["processing_report"] = path
    if "result_bundle" in output_names:
        path = Path(output_names["result_bundle"])
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            manifest = {artifact.name: {"sha256": _sha256(artifact), "bytes": artifact.stat().st_size} for artifact in generated.values()}
            for artifact in generated.values(): archive.write(artifact, arcname=artifact.name)
            sanitized_config = json.dumps(_redact_export(configuration), indent=2).encode()
            citations = json.dumps(_redact_export(provenance["citations"]), indent=2).encode()
            manifest["workflow-config.json"] = {"sha256": hashlib.sha256(sanitized_config).hexdigest(), "bytes": len(sanitized_config)}
            manifest["citations.json"] = {"sha256": hashlib.sha256(citations).hexdigest(), "bytes": len(citations)}
            archive.writestr("workflow-config.json", sanitized_config)
            archive.writestr("citations.json", citations)
            archive.writestr("manifest.json", json.dumps({"schema_version": "1.0", "files": manifest}, indent=2))
        generated["result_bundle"] = path
    for key, path in generated.items(): _upload(folder, path, output_names[key])
    _log(f"{action_title}: wrote {len(generated)} artifacts and {len(frame)} rows")
    # FaaSr's RPC return contract accepts only bool or None. Detailed action results
    # are already persisted in the summary/provenance artifacts and emitted above.
    return True


# The Studio compiler replaces this marker with a node-specific FaaSr entry point.
def rate_of_change_check_b0d36725(folder, input1, input2, output1, output2, output3, output4, output5, output6, output7, output8):
    return _execute("rate-of-change-check", "rate_of_change_check_b0d36725", "Rate-of-change check", folder, [input1, input2], {"data_parquet": output1, "data_csv": output2, "summary_json": output3, "visualization_json": output4, "visualization_png": output5, "provenance_json": output6, "processing_report": output7, "map_geojson": output8})

