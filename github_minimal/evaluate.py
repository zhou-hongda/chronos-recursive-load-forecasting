import gc
import json
import math
import os
import re
import shutil
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

import preprocessing
from experiment_config import (
    CHRONOS_BASE_MODEL_ID,
    FINETUNED_CHRONOS,
    PAPER_OUTPUT_PATH,
    ZERO_SHOT_CHRONOS,
    chronos_checkpoint_path,
    model_checkpoint_path,
)
from model_utils import TSLIB_MODELS, forward_batch, match_target_shape


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def evaluate(model, dataloader, criterion, device, model_name="LSTM"):
    """Compute validation loss and aggregate metrics for training."""
    model.eval()
    total_loss = 0.0
    total_observed = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y, batch_mask in dataloader:
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            outputs = match_target_shape(forward_batch(model, batch_x, device, model_name), batch_y)
            loss = criterion(outputs, batch_y, batch_mask)
            observed_count = int(batch_mask.sum().item())
            total_loss += loss.item() * observed_count
            total_observed += observed_count
            predictions.append(outputs.cpu().numpy())
            actuals.append(
                np.where(
                    batch_mask.cpu().numpy(),
                    batch_y.cpu().numpy(),
                    np.nan,
                )
            )

    if not predictions:
        raise ValueError("Evaluation dataloader is empty")

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    metrics_values = evaluate_metrics(predictions, actuals)
    if total_observed == 0:
        raise ValueError("Evaluation dataloader contains no observed target values")
    metrics_values["loss"] = total_loss / total_observed
    return {"overall": metrics_values}, predictions, actuals


def evaluate_metrics(predictions, actuals, mape_threshold=1.0, allow_empty=False):
    """Compute aggregate metrics after flattening all finite prediction-target pairs."""
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    act = np.asarray(actuals, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(act)
    pred = pred[valid]
    act = act[valid]

    if pred.size == 0:
        if allow_empty:
            return {
                "mse": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "wape": np.nan,
                "smape": np.nan,
                "r2": np.nan,
            }
        raise ValueError("No finite prediction-target pairs available for metric computation")

    err = pred - act
    abs_err = np.abs(err)
    eps = 1e-8

    mse = mean_squared_error(act, pred)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))
    r2 = r2_score(act, pred) if pred.size > 1 else np.nan

    mape_mask = np.abs(act) >= mape_threshold
    mape = float(np.mean(abs_err[mape_mask] / np.abs(act[mape_mask])) * 100) if np.any(mape_mask) else np.nan
    wape = float(np.sum(abs_err) / max(np.sum(np.abs(act)), eps) * 100)
    smape = float(np.mean(2 * abs_err / (np.abs(act) + np.abs(pred) + eps)) * 100)

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "wape": wape,
        "smape": smape,
        "r2": float(r2),
    }


def _torch_load_checkpoint(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_trained_model(model_name, config, device):
    if model_name in {FINETUNED_CHRONOS, ZERO_SHOT_CHRONOS}:
        from chronos_adapter import ChronosAdapter

        adapter_config = dict(config)
        adapter_config["chronos_experiment_name"] = model_name
        if model_name == ZERO_SHOT_CHRONOS:
            model_id = CHRONOS_BASE_MODEL_ID
            print(f"\nLoading zero-shot Chronos-2 from {model_id}...")
        else:
            checkpoint_path = chronos_checkpoint_path(model_name)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Fine-tuned Chronos checkpoint not found: {checkpoint_path}. "
                    f"Train {model_name} first."
                )
            model_id = checkpoint_path
            print(f"\nLoading {model_name} from {model_id}...")
            strategy_path = Path(checkpoint_path) / "chronos_strategy.json"
            with strategy_path.open("r", encoding="utf-8") as file:
                experiment_metadata = json.load(file)
            checkpoint_pipeline = experiment_metadata.get("data_pipeline_version")
            current_pipeline = config.get("data_pipeline_version")
            if checkpoint_pipeline != current_pipeline:
                raise ValueError(
                    "Chronos checkpoint was trained with a different data pipeline "
                    f"({checkpoint_pipeline!r} != {current_pipeline!r}); retrain Chronos."
                )
            if experiment_metadata.get("feature_names") != config.get("feature_names"):
                raise ValueError(
                    "Chronos checkpoint transformer cohort/order differs from the current data; "
                    "retrain Chronos."
                )
            adapter_config["chronos_feature_group_size"] = experiment_metadata.get(
                "feature_group_size",
                adapter_config.get("chronos_feature_group_size", 32),
            )
            adapter_config["chronos_feature_grouping"] = experiment_metadata.get(
                "feature_grouping",
                "sequential",
            )
            if experiment_metadata.get("feature_groups") is not None:
                adapter_config["chronos_feature_groups"] = experiment_metadata[
                    "feature_groups"
                ]
        model = ChronosAdapter(adapter_config, model_id=model_id).to(device)
        if model_name == ZERO_SHOT_CHRONOS:
            model.experiment_metadata = {
                "model": model_name,
                "training": "zero-shot",
                "base_model": CHRONOS_BASE_MODEL_ID,
            }
        else:
            model.experiment_metadata = experiment_metadata
        return model

    import train

    model_path = model_checkpoint_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = _torch_load_checkpoint(model_path, device)
    checkpoint_mode = checkpoint.get("config", {}).get("data_mode", "multivariate")
    if checkpoint_mode != "multivariate":
        raise ValueError(
            f"{model_name} checkpoint protocol is {checkpoint_mode}, "
            "but current protocol is multivariate. Retrain the model."
        )

    checkpoint_pipeline = checkpoint.get("config", {}).get("data_pipeline_version")
    current_pipeline = config.get("data_pipeline_version")
    if checkpoint_pipeline != current_pipeline:
        raise ValueError(
            f"{model_name} checkpoint was trained with a different data pipeline "
            f"({checkpoint_pipeline!r} != {current_pipeline!r}); retrain the model."
        )
    checkpoint_features = checkpoint.get("config", {}).get("feature_names")
    if checkpoint_features != config.get("feature_names"):
        raise ValueError(
            f"{model_name} checkpoint transformer cohort/order differs from the current data; "
            "retrain the model."
        )

    checkpoint_ratio = checkpoint.get("config", {}).get("train_ratio")
    current_ratio = config.get("train_ratio")
    if checkpoint_ratio is not None and current_ratio is not None and checkpoint_ratio != current_ratio:
        print(
            f"[Warning] {model_name} checkpoint was trained with train_ratio={checkpoint_ratio}, "
            f"but current train_ratio={current_ratio}."
        )

    model_config = dict(config)
    model_config.update(checkpoint.get("config", {}))
    model = train.create_model(model_name, model_config, device=device)
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )
    model.load_state_dict(clean_state_dict)
    model.to(device).eval()
    model.experiment_metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    return model


def _safe_name(text):
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", str(text))
    return re.sub(r"[\\/:\*\?\"<>\|]", "_", text)


def _use_paper_font():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False


def _restore_default_font():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _save_paper_figure(fig, save_dir, stem):
    figures_dir = Path(save_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_multi_model_path_comparison(all_paths, actuals, model_names, feature_names, save_dir):
    _use_paper_font()
    display_limit = min(24 * 14, len(actuals))
    feature_count = min(2, actuals.shape[1])
    fig, axes = plt.subplots(feature_count, 1, figsize=(13, 4.2 * feature_count), sharex=True)
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("tab10").colors
    for feature_idx, ax in enumerate(axes):
        feature_name = str(feature_names[feature_idx])
        ax.plot(actuals[:display_limit, feature_idx], color="black", linewidth=2, label="Actual")
        for model_index, model_name in enumerate(model_names):
            if model_name in all_paths:
                ax.plot(
                    all_paths[model_name][:display_limit, feature_idx],
                    linewidth=1.15,
                    label=model_name,
                    color=colors[model_index % len(colors)],
                    alpha=0.9,
                )
        ax.set_title(f"Transformer: {feature_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Load")
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[-1].set_xlabel("Hours since test forecast origin")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Continuous Recursive Forecast Paths (First 14 Days)", y=0.995, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save_paper_figure(fig, save_dir, "forecast_path_comparison")
    plt.close(fig)

    _restore_default_font()


def _best_mse_reference(all_metrics):
    if not all_metrics:
        raise ValueError("At least one model metric is required")
    reference = min(float(metrics["mse"]) for metrics in all_metrics.values())
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Relative MSE visualization requires a positive finite MSE reference")
    return reference


def _add_relative_mse_columns(step_metrics, mse_reference):
    if not math.isfinite(mse_reference) or mse_reference <= 0:
        raise ValueError("Relative MSE requires a positive finite reference")

    normalized = step_metrics.copy()
    for scope in ("block", "cumulative"):
        source = f"{scope}_mse"
        if source in normalized:
            normalized[f"{scope}_relative_mse"] = normalized[source] / mse_reference
    return normalized


def plot_metrics_comparison_combined(all_metrics, save_dir):
    _use_paper_font()

    metrics_names = ["mse", "mae", "wape", "r2"]
    models = list(all_metrics.keys())
    use_relative_mse = len(models) > 1
    titles = [
        "(a) Relative MSE (Best = 1.0)" if use_relative_mse else "(a) MSE (Real Scale)",
        "(b) MAE",
        "(c) WAPE",
        "(d) R2",
    ]
    y_labels = ["Relative MSE" if use_relative_mse else "MSE", "MAE", "WAPE", "R2"]
    mse_reference = _best_mse_reference(all_metrics)

    print("Generating combined metrics comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.flatten()
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]
        values = [all_metrics[m][metric_name] for m in models]
        if metric_name == "mse" and use_relative_mse:
            values = [value / mse_reference for value in values]
        bars = ax.bar(
            models,
            values,
            color="#4C72B0",
            alpha=0.88,
            width=0.65,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_title(titles[i], fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(y_labels[i], fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        v_min, v_max = min(values), max(values)
        margin = max((v_max - v_min) * 0.2, max(abs(v_max), 1.0) * 0.08)
        if metric_name == "r2":
            ax.set_ylim(min(0.0, v_min - margin), v_max + margin)
        else:
            ax.set_ylim(0.0, v_max + margin)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + margin * 0.08,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout(pad=2.0)
    _save_paper_figure(fig, save_dir, "metrics_bar_chart")
    plt.close(fig)
    _restore_default_font()


def plot_recursive_horizon_comparison(step_metrics, save_dir):
    """Plot cumulative test performance as recursive forecast length increases."""
    _use_paper_font()
    metric_columns = [
        "cumulative_relative_mse",
        "cumulative_mae",
        "cumulative_wape",
        "cumulative_r2",
    ]
    titles = [
        "(a) Cumulative Relative MSE",
        "(b) Cumulative MAE",
        "(c) Cumulative WAPE",
        "(d) Cumulative R2",
    ]
    y_labels = ["Relative MSE", "MAE", "WAPE", "R2"]
    models = list(dict.fromkeys(step_metrics["Model"]))
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for metric_column, title, y_label, ax in zip(
        metric_columns, titles, y_labels, axes.flatten()
    ):
        for model_index, model_name in enumerate(models):
            rows = step_metrics[step_metrics["Model"] == model_name]
            ax.plot(
                rows["forecast_days"],
                rows[metric_column],
                label=model_name,
                color=colors[model_index % len(colors)],
                linewidth=1.4,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3, linestyle="--")
    for ax in axes[-1]:
        ax.set_xlabel("Recursive forecast length (days)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_paper_figure(fig, save_dir, "recursive_horizon_comparison")
    plt.close(fig)
    _restore_default_font()


def _inverse_multivariate_scaler(scaler, values):
    values = np.asarray(values)
    feature_count = values.shape[-1]
    restored = scaler.inverse_transform(values.reshape(-1, feature_count))
    return restored.reshape(values.shape)


def _make_recursive_multivariate_batch(context, config, model_name, device):
    context = np.asarray(context, dtype=np.float32)
    x_enc = torch.from_numpy(context[None, :, :]).to(device)
    if model_name not in TSLIB_MODELS:
        return x_enc

    label_len = config["seq_len"] // 2
    pred_len = config["pred_len"]
    num_features = context.shape[1]
    decoder = torch.cat(
        [
            x_enc[:, -label_len:, :],
            torch.zeros((1, pred_len, num_features), dtype=torch.float32, device=device),
        ],
        dim=1,
    )
    return {
        "x_enc": x_enc,
        "x_mark_enc": torch.zeros((1, config["seq_len"], 4), dtype=torch.float32, device=device),
        "x_dec": decoder,
        "x_mark_dec": torch.zeros((1, label_len + pred_len, 4), dtype=torch.float32, device=device),
    }


def _get_model_recursive_stride(model_name, config):
    return int(config["recursive_stride"])


def make_multivariate_recursive_predictions(
    model,
    model_name,
    test_data,
    test_dates,
    config,
    device,
    prediction_bounds=None,
):
    """Forecast the test tail by recursively feeding predicted steps back into history."""
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    recursive_stride = _get_model_recursive_stride(model_name, config)
    if recursive_stride < 1:
        raise ValueError(f"recursive_stride must be >= 1, got {recursive_stride}")
    if recursive_stride > pred_len:
        raise ValueError(
            f"recursive_stride cannot exceed pred_len because only the first pred_len outputs are available: "
            f"recursive_stride={recursive_stride}, pred_len={pred_len}"
        )
    if len(test_data) <= seq_len:
        raise ValueError(
            f"Test data must contain the initial {seq_len}h context plus future targets; "
            f"got {len(test_data)} rows"
        )

    forecast_length = len(test_data) - seq_len
    history = np.asarray(test_data[:seq_len], dtype=np.float32).copy()
    actuals = np.asarray(test_data[seq_len:], dtype=np.float32)
    predictions = []
    diagnostics = {
        "recursive_stride": recursive_stride,
        "nonfinite_values": 0,
        "fallback_blocks": 0,
        "clipped_values": 0,
    }
    lower_bound = upper_bound = None
    if prediction_bounds is not None:
        lower_bound, upper_bound = [np.asarray(bound, dtype=np.float32) for bound in prediction_bounds]
        if lower_bound.shape != (test_data.shape[1],) or upper_bound.shape != (test_data.shape[1],):
            raise ValueError(
                "prediction_bounds must contain per-feature arrays shaped "
                f"({test_data.shape[1]},), got {lower_bound.shape}/{upper_bound.shape}"
            )

    print(
        f"{model_name}: recursive multivariate forecast, "
        f"context={seq_len}, model_horizon={pred_len}, recursive_stride={recursive_stride}, "
        f"target_steps={forecast_length}."
    )
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, forecast_length, recursive_stride), desc=f"{model_name} Recursive Stride Test"):
            context = history[-seq_len:]
            batch_x = _make_recursive_multivariate_batch(context, config, model_name, device)
            expected = torch.empty((1, pred_len, test_data.shape[1]), dtype=torch.float32, device=device)
            output = match_target_shape(forward_batch(model, batch_x, device, model_name), expected)

            block = output[0].detach().cpu().numpy().astype(np.float32)
            keep = min(recursive_stride, forecast_length - start)
            used_block = block[:keep]

            nonfinite_mask = ~np.isfinite(used_block)
            diagnostics["nonfinite_values"] += int(np.sum(nonfinite_mask))
            if np.any(nonfinite_mask):
                fallback = np.repeat(context[-1:, :], keep, axis=0)
                used_block = np.where(nonfinite_mask, fallback, used_block)
                diagnostics["fallback_blocks"] += 1

            if lower_bound is not None:
                clipped_mask = (used_block < lower_bound[None, :]) | (used_block > upper_bound[None, :])
                diagnostics["clipped_values"] += int(np.sum(clipped_mask))
                used_block = np.clip(used_block, lower_bound[None, :], upper_bound[None, :])

            predictions.append(used_block)
            history = np.concatenate([history, used_block], axis=0)

    predictions = np.concatenate(predictions, axis=0)
    target_times = test_dates[seq_len:seq_len + len(predictions)]
    return predictions, actuals[:len(predictions)], target_times, diagnostics


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def _reset_paper_output(save_dir):
    root = Path(PAPER_OUTPUT_PATH).resolve()
    target = Path(save_dir).resolve()
    target.relative_to(root)
    if target == root:
        raise ValueError("Refusing to reset the paper output root")
    root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            shutil.rmtree(target)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot refresh result directory because a file is open: {target}. "
                "Close predictions.csv and other result files in Excel, editors, or preview panes, "
                "then rerun comparison; trained checkpoints are unaffected."
            ) from exc
    target.mkdir(parents=True, exist_ok=True)


def _comparison_output_dir(model_names):
    if len(model_names) == 1:
        return Path(PAPER_OUTPUT_PATH) / "single_models" / _safe_name(model_names[0])
    return Path(PAPER_OUTPUT_PATH) / "model_comparison"


def _build_recursive_step_metrics(actuals, all_predictions, all_diagnostics):
    rows = []
    for model_name, predictions in all_predictions.items():
        stride = int(all_diagnostics[model_name]["recursive_stride"])
        for step, start in enumerate(range(0, len(actuals), stride), start=1):
            end = min(start + stride, len(actuals))
            block_metrics = evaluate_metrics(
                predictions[start:end], actuals[start:end], allow_empty=True
            )
            cumulative_metrics = evaluate_metrics(
                predictions[:end], actuals[:end], allow_empty=True
            )
            row = {
                "Model": model_name,
                "recursive_step": step,
                "start_hour": start + 1,
                "end_hour": end,
                "forecast_days": end / 24.0,
            }
            row.update({f"block_{key}": value for key, value in block_metrics.items()})
            row.update({f"cumulative_{key}": value for key, value in cumulative_metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _export_multivariate_recursive_outputs(
    save_dir,
    config,
    feature_names,
    target_times,
    actuals,
    all_predictions,
    all_metrics,
    all_diagnostics,
    model_metadata,
    failed_models,
    data_quality,
):
    best_mse = _best_mse_reference(all_metrics)
    summary_rows = [
        {
            "Model": model_name,
            **metrics,
            "relative_mse": float(metrics["mse"]) / best_mse,
        }
        for model_name, metrics in all_metrics.items()
    ]
    pd.DataFrame(summary_rows).to_csv(
        Path(save_dir) / "metrics_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [{"Model": model_name, **diagnostics} for model_name, diagnostics in all_diagnostics.items()]
    ).to_csv(Path(save_dir) / "diagnostics.csv", index=False)

    per_feature_rows = []
    for model_name, predictions in all_predictions.items():
        for feature_idx, feature_name in enumerate(feature_names):
            per_feature_rows.append(
                {
                    "Model": model_name,
                    "Transformer": feature_name,
                    **evaluate_metrics(
                        predictions[:, feature_idx],
                        actuals[:, feature_idx],
                        allow_empty=True,
                    ),
                }
            )
    pd.DataFrame(per_feature_rows).to_csv(
        Path(save_dir) / "per_transformer_metrics.csv",
        index=False,
    )

    step_metrics = _build_recursive_step_metrics(actuals, all_predictions, all_diagnostics)
    step_metrics = _add_relative_mse_columns(step_metrics, best_mse)
    step_metrics.to_csv(Path(save_dir) / "recursive_step_metrics.csv", index=False)

    path_data = {"Time": target_times}
    for feature_idx, feature_name in enumerate(feature_names):
        path_data[f"{feature_name}_True"] = actuals[:, feature_idx]
        for model_name, predictions in all_predictions.items():
            path_data[f"{model_name}_{feature_name}_Pred"] = predictions[:, feature_idx]
    pd.DataFrame(path_data).to_csv(Path(save_dir) / "predictions.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_status": (
            "current_protocol_result" if not failed_models else "partial_model_failure"
        ),
        "protocol": "continuous_recursive_multivariate_forecast",
        "successful_models": list(all_predictions),
        "failed_models": failed_models,
        "evaluation_config": config,
        "dataset": {
            "path": config["data_path"],
            "train_ratio": config["train_ratio"],
            "validation_ratio": config["validation_ratio"],
            "context_hours": config["seq_len"],
            "model_horizon_hours": config["pred_len"],
            "forecast_steps": len(actuals),
            "num_transformers": len(feature_names),
            "forecast_start": str(target_times[0]),
            "forecast_end": str(target_times[-1]),
            "observed_forecast_values": int(np.isfinite(actuals).sum()),
            "total_forecast_values": int(actuals.size),
            "observed_forecast_rate": float(np.isfinite(actuals).mean()),
            "data_quality": data_quality,
        },
        "model_training_metadata": model_metadata,
        "metric_scope": (
            "All originally observed forecast targets across hours and transformers "
            "on the original load scale; imputed targets are excluded"
        ),
        "display_conventions": {
            "relative_mse": (
                "Each model's original-scale MSE divided by the minimum MSE among "
                "the models in this comparison; this is the MSE value reported in "
                "manuscript text, tables, and aggregate figures"
            ),
            "recursive_relative_mse": (
                "Every cumulative and block MSE value is divided by the same fixed "
                "comparison-level reference so the recursive curves remain comparable "
                "and their final values match the aggregate relative MSE"
            ),
            "relative_mse_reference": best_mse,
            "reported_metrics": (
                "CSV files retain original-scale MSE for reproducibility and add "
                "relative MSE columns used by the manuscript"
            ),
        },
        "files": {
            "aggregate_metrics": "metrics_summary.csv",
            "per_transformer_metrics": "per_transformer_metrics.csv",
            "recursive_step_metrics": "recursive_step_metrics.csv",
            "predictions": "predictions.csv",
            "diagnostics": "diagnostics.csv",
            "figures": "figures/",
        },
    }
    with (Path(save_dir) / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(manifest), file, indent=2, ensure_ascii=True)
    return step_metrics


def compare_models(model_names, config):
    model_strides = {model_name: _get_model_recursive_stride(model_name, config) for model_name in model_names}
    print(f"Starting recursive multivariate comparison: models={model_names}, strides={model_strides}")
    save_dir = _comparison_output_dir(model_names)
    _reset_paper_output(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, _, test_data, scaler, _, split_info = preprocessing.load_multivariate_data(
        file_path=config["data_path"],
        train_ratio=config.get("train_ratio", 0.8),
        validation_ratio=config.get("validation_ratio", 0.1),
        seq_len=config["seq_len"],
    )
    config["input_dim"] = test_data.shape[1]
    feature_names = split_info["feature_names"]
    test_dates = split_info["test_index"]
    test_observed_mask = split_info["test_observed_mask"]
    config["feature_names"] = feature_names
    config["data_quality"] = split_info["data_quality"]

    train_min = np.min(train_data, axis=0)
    train_max = np.max(train_data, axis=0)
    clip_margin = float(config.get("recursive_clip_margin", 0.1))
    margin = (train_max - train_min) * clip_margin + 1e-6
    prediction_bounds = (train_min - margin, train_max + margin)

    all_predictions = {}
    all_metrics = {}
    all_diagnostics = {}
    model_metadata = {}
    failed_models = {}
    actuals_orig = None
    target_times = None

    for model_name in model_names:
        model = None
        try:
            model = load_trained_model(model_name, config, device)
            pred_norm, act_norm, model_target_times, diagnostics = make_multivariate_recursive_predictions(
                model,
                model_name,
                test_data,
                test_dates,
                config,
                device,
                prediction_bounds=prediction_bounds,
            )
            pred_orig = _inverse_multivariate_scaler(scaler, pred_norm)
            act_orig = _inverse_multivariate_scaler(scaler, act_norm)
            target_mask = test_observed_mask[
                config["seq_len"]:config["seq_len"] + len(act_orig)
            ]
            act_orig = np.where(target_mask, act_orig, np.nan)
            actuals_orig = act_orig
            target_times = model_target_times
            all_predictions[model_name] = pred_orig
            all_metrics[model_name] = evaluate_metrics(pred_orig, act_orig)
            all_diagnostics[model_name] = diagnostics
            model_metadata[model_name] = getattr(model, "experiment_metadata", {})
            print(
                f"{model_name}: MAE={all_metrics[model_name]['mae']:.4f}, "
                f"WAPE={all_metrics[model_name]['wape']:.2f}%, "
                f"R2={all_metrics[model_name]['r2']:.4f}, "
                f"stride={diagnostics['recursive_stride']}, "
                f"nonfinite={diagnostics['nonfinite_values']}, "
                f"clipped={diagnostics['clipped_values']}"
            )
        except Exception as exc:
            print(f"Model {model_name} failed: {exc}")
            failed_models[model_name] = str(exc)
            traceback.print_exc()
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if actuals_orig is None:
        raise RuntimeError("No multivariate model produced predictions")

    step_metrics = _export_multivariate_recursive_outputs(
        save_dir,
        config,
        feature_names,
        target_times,
        actuals_orig,
        all_predictions,
        all_metrics,
        all_diagnostics,
        model_metadata,
        failed_models,
        split_info["data_quality"],
    )
    plot_multi_model_path_comparison(all_predictions, actuals_orig, list(all_predictions), feature_names, save_dir)
    plot_metrics_comparison_combined(all_metrics, save_dir)
    plot_recursive_horizon_comparison(step_metrics, save_dir)
    print(f"Recursive multivariate comparison outputs saved to {save_dir}")


def predict_mode(model_name, config):
    return compare_models([model_name], config)
