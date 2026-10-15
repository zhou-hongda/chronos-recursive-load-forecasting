import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

try:
    from utils.timefeatures import time_features
except ImportError:
    time_features = None
    print("Warning: utils.timefeatures not found. TSLib datasets will use zero time features.")


DATA_PIPELINE_VERSION = "causal-observed-mask-v1"
DEFAULT_MISSING_THRESHOLD = 0.25
CAUSAL_SEASONAL_LAGS = (168, 24)


def _read_time_indexed_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}")

    df = pd.read_csv(file_path)
    time_col = "DATETIME" if "DATETIME" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    return df.apply(pd.to_numeric, errors="coerce")


def _make_time_features(dates, freq="h"):
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    if time_features is None:
        return np.zeros((len(dates), 4), dtype=np.float32)

    stamp = time_features(pd.to_datetime(dates.values), freq=freq)
    return stamp.transpose(1, 0).astype(np.float32)


def _metadata_path(output_path):
    return Path(output_path).with_suffix(".meta.json")


def _processed_cache_is_current(raw_path, output_path, start_date, end_date):
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    metadata_path = _metadata_path(output_path)
    if not output_path.is_file() or not metadata_path.is_file():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, ValueError):
        return False

    raw_stat = raw_path.stat()
    return (
        metadata.get("data_pipeline_version") == DATA_PIPELINE_VERSION
        and metadata.get("raw_size_bytes") == raw_stat.st_size
        and metadata.get("raw_mtime_ns") == raw_stat.st_mtime_ns
        and metadata.get("start_date") == str(start_date)
        and metadata.get("end_date") == str(end_date)
    )


def generate_processed_file(
    raw_path,
    output_path,
    target_ids=None,
    start_date="2022-01-01",
    end_date="2023-11-09",
):
    """Convert long-format records into an hourly observation table without imputation."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw data file does not exist: {raw_path}")
    if _processed_cache_is_current(raw_path, output_path, start_date, end_date):
        print(f"Using current observed-data cache: {output_path}")
        return

    print(f"Processing raw data: {raw_path}")
    df = pd.read_csv(raw_path)
    required_cols = {"TRANSFORMER_ID", "LOAD", "DATETIME"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing_cols)}")

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["LOAD"] = pd.to_numeric(df["LOAD"], errors="coerce")
    df = df.dropna(subset=["LOAD", "DATETIME", "TRANSFORMER_ID"])

    if target_ids is not None:
        df = df[df["TRANSFORMER_ID"].isin(target_ids)].copy()
        if df.empty:
            raise ValueError("No data remains after filtering by target_ids")

    print("Pivoting records to an hourly observation table...")
    observed = df.pivot_table(
        index="DATETIME",
        columns="TRANSFORMER_ID",
        values="LOAD",
        aggfunc="mean",
    ).resample("1h").asfreq()
    observed = observed.loc[start_date:end_date]
    if observed.empty:
        raise ValueError(f"No data in requested time range: {start_date} ~ {end_date}")
    if observed.shape[1] == 0:
        raise ValueError("No transformer columns remain after pivoting")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    observed.to_csv(temporary_path)
    temporary_path.replace(output_path)

    raw_stat = raw_path.stat()
    metadata = {
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "raw_path": str(raw_path),
        "raw_size_bytes": raw_stat.st_size,
        "raw_mtime_ns": raw_stat.st_mtime_ns,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "rows": int(observed.shape[0]),
        "transformers_before_training_filter": int(observed.shape[1]),
        "stored_values": "original_hourly_observations_with_missing_values_preserved",
    }
    metadata_path = _metadata_path(output_path)
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    with temporary_metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    temporary_metadata_path.replace(metadata_path)
    print(
        f"Saved {observed.shape[0]} hourly rows and {observed.shape[1]} raw transformer series "
        f"to {output_path}; missing values were preserved."
    )


def _select_training_series(train_observed, missing_threshold, min_active_ratio):
    observed_count = train_observed.notna().sum()
    missing_rate = train_observed.isna().mean()
    active_count = train_observed.abs().gt(1e-8).sum()
    active_ratio = active_count.div(observed_count.where(observed_count > 0))
    varying = train_observed.std(skipna=True).gt(1e-8)

    valid = (
        missing_rate.le(missing_threshold)
        & active_ratio.ge(min_active_ratio)
        & varying
        & observed_count.gt(0)
    ).fillna(False)
    reasons = {
        "missing_rate": train_observed.columns[missing_rate.gt(missing_threshold)].tolist(),
        "inactive": train_observed.columns[active_ratio.lt(min_active_ratio).fillna(True)].tolist(),
        "constant": train_observed.columns[~varying.fillna(False)].tolist(),
    }
    return train_observed.columns[valid].tolist(), reasons


def _causal_impute(observed, train_end):
    """Fill inputs using past values only, with a training-only fallback statistic."""
    filled = observed.copy()
    for lag in CAUSAL_SEASONAL_LAGS:
        filled = filled.fillna(filled.shift(lag))
    filled = filled.ffill()

    training_medians = observed.iloc[:train_end].median(skipna=True)
    filled = filled.fillna(training_medians)
    remaining = filled.columns[filled.isna().any()].tolist()
    if remaining:
        raise ValueError(
            "Causal imputation could not fill transformers without training observations: "
            f"{remaining[:10]}"
        )
    return filled


def load_multivariate_data(
    file_path,
    seq_len=168,
    train_ratio=0.8,
    validation_ratio=0.1,
    min_active_ratio=0.01,
    missing_threshold=DEFAULT_MISSING_THRESHOLD,
):
    """Select, impute, scale, and split all series using training information only."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < train_ratio:
        raise ValueError("validation_ratio must be positive and smaller than train_ratio")
    if not 0 <= missing_threshold < 1:
        raise ValueError("missing_threshold must be in [0, 1)")

    observed = _read_time_indexed_csv(file_path).sort_index()
    if not observed.index.is_monotonic_increasing:
        raise ValueError("Datetime index must be sorted in increasing order")
    if observed.index.has_duplicates:
        raise ValueError("Datetime index contains duplicate timestamps")

    total_steps = len(observed)
    test_start = int(total_steps * train_ratio)
    val_start = int(total_steps * (train_ratio - validation_ratio))
    if val_start <= seq_len or test_start - val_start < 24 or total_steps - test_start < 24:
        raise ValueError(
            f"Insufficient data for chronological split: total={total_steps}, "
            f"val_start={val_start}, test_start={test_start}"
        )

    valid_cols, rejection_reasons = _select_training_series(
        observed.iloc[:val_start],
        missing_threshold=missing_threshold,
        min_active_ratio=min_active_ratio,
    )
    if not valid_cols:
        raise ValueError("No transformer series passed the training-period quality criteria")
    original_feature_count = observed.shape[1]
    observed = observed.loc[:, valid_cols]
    observed_mask = observed.notna().to_numpy(dtype=bool)
    filled = _causal_impute(observed, train_end=val_start)

    train_raw = filled.iloc[:val_start].to_numpy(dtype=np.float32)
    val_raw = filled.iloc[val_start - seq_len:test_start].to_numpy(dtype=np.float32)
    test_raw = filled.iloc[test_start - seq_len:].to_numpy(dtype=np.float32)
    train_mask = observed_mask[:val_start]
    val_mask = observed_mask[val_start - seq_len:test_start]
    test_mask = observed_mask[test_start - seq_len:]

    # Each transformer is standardized only with optimization-period inputs.
    scaler = StandardScaler()
    scaler.fit(train_raw)
    train_data = scaler.transform(train_raw).astype(np.float32)
    val_data = scaler.transform(val_raw).astype(np.float32)
    test_data = scaler.transform(test_raw).astype(np.float32)

    target_masks = {
        "train": train_mask[seq_len:],
        "validation": val_mask[seq_len:],
        "test": test_mask[seq_len:],
    }
    rejected_transformers = sorted(set().union(*map(set, rejection_reasons.values())))
    data_quality = {
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "series_selection_period": "training_only",
        "missing_threshold": float(missing_threshold),
        "min_active_ratio": float(min_active_ratio),
        "causal_fill_order": [f"lag_{lag}h" for lag in CAUSAL_SEASONAL_LAGS]
        + ["forward_fill", "training_median_for_leading_values"],
        "transformers_before_filter": int(original_feature_count),
        "transformers_after_filter": int(len(valid_cols)),
        "rejected_transformers": rejected_transformers,
        "rejection_counts": {key: len(value) for key, value in rejection_reasons.items()},
        "observed_target_values": {
            name: int(mask.sum()) for name, mask in target_masks.items()
        },
        "total_target_values": {
            name: int(mask.size) for name, mask in target_masks.items()
        },
        "observed_target_rates": {
            name: float(mask.mean()) for name, mask in target_masks.items()
        },
    }
    split_info = {
        "train_end": val_start,
        "validation_start": val_start,
        "test_start": test_start,
        "train_index": observed.index[:val_start],
        "validation_index": observed.index[val_start - seq_len:test_start],
        "test_index": observed.index[test_start - seq_len:],
        "feature_names": observed.columns.tolist(),
        "train_observed_mask": train_mask,
        "validation_observed_mask": val_mask,
        "test_observed_mask": test_mask,
        "data_quality": data_quality,
    }
    print(
        "[Multivariate split] "
        f"train={train_data.shape}, validation={val_data.shape}, test={test_data.shape}, "
        f"features={observed.shape[1]}, test boundary={observed.index[test_start]}"
    )
    print(
        "[Observation masks] "
        f"train={target_masks['train'].mean():.2%}, "
        f"validation={target_masks['validation'].mean():.2%}, "
        f"test={target_masks['test'].mean():.2%} observed target values."
    )
    return train_data, val_data, test_data, scaler, observed, split_info


class MultivariateTransformerDataset(Dataset):
    """Ordered windows: [seq_len, features] -> [pred_len, features]."""

    def __init__(self, data, seq_len, pred_len, stride=1, target_mask=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.target_mask = (
            np.ones_like(self.data, dtype=bool)
            if target_mask is None
            else np.asarray(target_mask, dtype=bool)
        )
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.data.ndim != 2:
            raise ValueError(f"Expected [time, features], got {self.data.shape}")
        if self.target_mask.shape != self.data.shape:
            raise ValueError(
                f"target_mask must match data shape {self.data.shape}, got {self.target_mask.shape}"
            )
        self.num_windows = max(
            0,
            (len(self.data) - self.seq_len - self.pred_len) // self.stride + 1,
        )

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        start = index * self.stride
        split = start + self.seq_len
        end = split + self.pred_len
        return (
            torch.from_numpy(self.data[start:split]),
            torch.from_numpy(self.data[split:end]),
            torch.from_numpy(self.target_mask[split:end]),
        )


class MultivariateDataset_TSLib(Dataset):
    """Ordered multivariate windows for TSLib-compatible models."""

    def __init__(
        self,
        data,
        dates,
        seq_len,
        pred_len,
        label_len=None,
        freq="h",
        stride=1,
        use_time_features=False,
        target_mask=None,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.target_mask = (
            np.ones_like(self.data, dtype=bool)
            if target_mask is None
            else np.asarray(target_mask, dtype=bool)
        )
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.label_len = int(label_len if label_len is not None else seq_len // 2)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.data.ndim != 2:
            raise ValueError(f"Expected [time, features], got {self.data.shape}")
        if self.target_mask.shape != self.data.shape:
            raise ValueError(
                f"target_mask must match data shape {self.data.shape}, got {self.target_mask.shape}"
            )
        self.num_features = self.data.shape[1]
        self.num_windows = max(
            0,
            (len(self.data) - self.seq_len - self.pred_len) // self.stride + 1,
        )
        self.data_stamp = (
            _make_time_features(dates, freq=freq)
            if use_time_features
            else np.zeros((len(dates), 4), dtype=np.float32)
        )

    def __len__(self):
        return self.num_windows

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        decoder = np.concatenate(
            [
                self.data[r_begin:s_end],
                np.zeros((self.pred_len, self.num_features), dtype=np.float32),
            ],
            axis=0,
        )
        model_inputs = {
            "x_enc": torch.from_numpy(self.data[s_begin:s_end]),
            "x_mark_enc": torch.from_numpy(self.data_stamp[s_begin:s_end]),
            "x_dec": torch.from_numpy(decoder),
            "x_mark_dec": torch.from_numpy(self.data_stamp[r_begin:r_end]),
        }
        return (
            model_inputs,
            torch.from_numpy(self.data[s_end:r_end]),
            torch.from_numpy(self.target_mask[s_end:r_end]),
        )
