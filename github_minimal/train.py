import csv
import importlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import preprocessing
from experiment_config import TRAINING_OUTPUT_PATH, model_checkpoint_path
from model.GRUModel import GRUModel
from model.LSTMModel import LSTMModel
from model.MLPModel import MLPModel
from model.RNNModel import RNNModel
from model.TCNModel import TCNModel
from model_utils import TSLIB_MODELS, forward_batch, match_target_shape


TSLIB_MODULES = {
    "Autoformer": "model.Autoformer",
    "iTransformer": "model.iTransformer",
}

STANDARD_MODEL_CLASSES = {
    "MLP": MLPModel,
    "RNN": RNNModel,
    "GRU": GRUModel,
    "LSTM": LSTMModel,
    "TCN": TCNModel,
}

STANDARD_MODEL_OVERRIDES = {
    "MLP": {
        "hidden_dims": (512, 512, 256, 256, 128, 128, 64),
    },
    "RNN": {
        "hidden_dim": 128,
        "num_layers": 6,
    },
    "GRU": {
        "hidden_dim": 128,
        "num_layers": 6,
    },
    "LSTM": {
        "hidden_dim": 128,
        "num_layers": 6,
    },
    "TCN": {
        "num_channels": (64, 128, 256),
        "kernel_size": 3,
    },
}

TSLIB_MODEL_CLASSES = {}
TSLIB_IMPORT_ERRORS = {}
for _model_name, _module_name in TSLIB_MODULES.items():
    try:
        TSLIB_MODEL_CLASSES[_model_name] = importlib.import_module(_module_name).Model
    except ImportError as exc:
        TSLIB_MODEL_CLASSES[_model_name] = None
        TSLIB_IMPORT_ERRORS[_model_name] = exc

ALL_MODELS = list(STANDARD_MODEL_CLASSES) + list(TSLIB_MODELS)


class Configs:
    """Attribute wrapper expected by TSLib model constructors."""

    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, value)


class CombinedLoss(nn.Module):
    """Weighted MSE and MAE training objective."""

    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, prediction, target, target_mask=None):
        if target_mask is None:
            target_mask = torch.ones_like(target, dtype=torch.bool)
        else:
            target_mask = target_mask.to(device=target.device, dtype=torch.bool)
        valid = target_mask & torch.isfinite(target)
        if not torch.any(valid):
            raise ValueError("Training batch contains no observed target values")

        error = prediction[valid] - target[valid]
        return self.alpha * torch.mean(error.square()) + self.beta * torch.mean(error.abs())


def resolved_model_parameters(model_name, config, device=None):
    """Return the complete architecture arguments recorded with a checkpoint."""
    if model_name in TSLIB_MODELS:
        model_args = {
            "task_name": "long_term_forecast",
            "is_training": 1,
            "model_id": f"{model_name}_1",
            "enc_in": config["input_dim"],
            "dec_in": config["input_dim"],
            "c_out": config["input_dim"],
            "seq_len": config["seq_len"],
            "label_len": config["seq_len"] // 2,
            "pred_len": config["pred_len"],
            "d_model": 128,
            "n_heads": 4,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 256,
            "dropout": config["dropout"],
            "embed": "timeF",
            "activation": "gelu",
            "output_attention": False,
            "freq": "h",
            "factor": 3,
            "moving_avg": 25,
        }
        return model_args

    if model_name in STANDARD_MODEL_CLASSES:
        model_args = {
            "input_dim": config["input_dim"],
            "seq_len": config["seq_len"],
            "pred_len": config["pred_len"],
            "dropout": config["dropout"],
        }
        model_args.update(STANDARD_MODEL_OVERRIDES[model_name])
        return model_args
    raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")


def create_model(model_name, config, device=None):
    """Create a standard or TSLib forecasting model."""
    model_args = resolved_model_parameters(model_name, config, device=device)
    if model_name in TSLIB_MODELS:
        model_class = TSLIB_MODEL_CLASSES.get(model_name)
        if model_class is None:
            raise ImportError(
                f"Could not import TSLib model {model_name}: {TSLIB_IMPORT_ERRORS.get(model_name)}"
            )

        args = Configs(model_args)
        return model_class(args)

    model_class = STANDARD_MODEL_CLASSES.get(model_name)
    model = model_class(**model_args)
    if model_name == "TCN":
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = None
    return model


def _serializable(value):
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def _torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(model, model_name, epoch, val_loss, metrics, config):
    """Save the best checkpoint with its full experiment configuration."""
    path = Path(model_checkpoint_path(model_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {
        "model_name": model_name,
        "best_epoch": int(epoch) + 1,
        "val_loss": float(val_loss),
        "metrics": metrics,
        "config": dict(config),
        "model_parameters": resolved_model_parameters(model_name, config),
        "model_state_dict": state_dict,
    }
    temporary_path = path.with_suffix(".tmp")
    torch.save(_serializable(checkpoint), temporary_path)
    temporary_path.replace(path)
    print(f"Saved best {model_name} checkpoint to {path} (validation loss={val_loss:.6f})")
    return path


def train_epoch(model, dataloader, optimizer, criterion, device, model_name, gradient_clip_norm):
    model.train()
    total_loss = 0.0
    total_observed = 0
    for batch_x, batch_y, batch_mask in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)
        outputs = match_target_shape(forward_batch(model, batch_x, device, model_name), batch_y)
        loss = criterion(outputs, batch_y, batch_mask)
        loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        observed_count = int(batch_mask.sum().item())
        total_loss += loss.item() * observed_count
        total_observed += observed_count
    if total_observed == 0:
        raise ValueError("Training dataloader contains no observed target values")
    return total_loss / total_observed


def build_datasets(
    config,
    train_data,
    val_data,
    train_dates,
    validation_dates,
    train_mask,
    validation_mask,
    model_name,
):
    stride = int(config["train_window_stride"])
    if model_name in TSLIB_MODELS:
        train_dataset = preprocessing.MultivariateDataset_TSLib(
            train_data,
            train_dates,
            config["seq_len"],
            config["pred_len"],
            stride=stride,
            target_mask=train_mask,
        )
        val_dataset = preprocessing.MultivariateDataset_TSLib(
            val_data,
            validation_dates,
            config["seq_len"],
            config["pred_len"],
            stride=stride,
            target_mask=validation_mask,
        )
    else:
        train_dataset = preprocessing.MultivariateTransformerDataset(
            train_data,
            config["seq_len"],
            config["pred_len"],
            stride=stride,
            target_mask=train_mask,
        )
        val_dataset = preprocessing.MultivariateTransformerDataset(
            val_data,
            config["seq_len"],
            config["pred_len"],
            stride=stride,
            target_mask=validation_mask,
        )

    if not train_dataset or not val_dataset:
        raise ValueError(
            f"Insufficient windows: train={len(train_dataset)}, validation={len(val_dataset)}"
        )
    return train_dataset, val_dataset


def train_model(config, model_name):
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_data, val_data, _, scaler, _, split_info = preprocessing.load_multivariate_data(
        file_path=config["data_path"],
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
        seq_len=config["seq_len"],
    )
    config["input_dim"] = train_data.shape[1]
    config["feature_names"] = split_info["feature_names"]
    config["data_quality"] = split_info["data_quality"]
    train_dataset, val_dataset = build_datasets(
        config,
        train_data,
        val_data,
        split_info["train_index"],
        split_info["validation_index"],
        split_info["train_observed_mask"],
        split_info["validation_observed_mask"],
        model_name,
    )

    batch_size = int(config["batch_size"])
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    print(
        f"Ordered windows: shuffle=False, batch_size={batch_size}, "
        f"train={len(train_dataset)}, validation={len(val_dataset)}, "
        f"stride={config['train_window_stride']}"
    )
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    model = create_model(model_name, config, device=device).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = CombinedLoss()
    best_val_loss = float("inf")
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    patience = max(1, int(config["early_stopping_patience"]))
    min_delta = max(0.0, float(config["early_stopping_min_delta"]))
    gradient_clip_norm = max(0.0, float(config["gradient_clip_norm"]))

    training_dir = Path(TRAINING_OUTPUT_PATH) / model_name
    training_root = Path(TRAINING_OUTPUT_PATH).resolve()
    resolved_training_dir = training_dir.resolve()
    resolved_training_dir.relative_to(training_root)
    if training_dir.exists():
        shutil.rmtree(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)

    from evaluate import evaluate as evaluate_model

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_name,
            gradient_clip_norm,
        )
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device, model_name)
        val_loss = val_metrics["overall"]["loss"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                **{f"validation_{key}": value for key, value in val_metrics["overall"].items()},
            }
        )
        print(
            f"Train loss={train_loss:.6f}, validation loss={val_loss:.6f}, "
            f"MSE={val_metrics['overall']['mse']:.6f}, R2={val_metrics['overall']['r2']:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(model, model_name, epoch, val_loss, val_metrics, config)
        else:
            epochs_without_improvement += 1
            print(
                f"Early stopping: {epochs_without_improvement}/{patience} epochs "
                f"without at least {min_delta:g} validation-loss improvement."
            )
            if epochs_without_improvement >= patience:
                print(f"Stopped at epoch {epoch + 1}; best epoch was {best_epoch}.")
                break

    if best_epoch == 0:
        raise RuntimeError(f"{model_name} did not produce a finite validation checkpoint")

    history_path = training_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(_serializable(history))

    with (training_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            _serializable(
                {
                    "model": model_name,
                    "ordered_training": True,
                    "shuffle": False,
                    "max_epochs": int(config["epochs"]),
                    "completed_epochs": len(history),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "early_stopping_patience": patience,
                    "early_stopping_min_delta": min_delta,
                    "gradient_clip_norm": gradient_clip_norm,
                    "training_config": config,
                    "model_parameters": resolved_model_parameters(model_name, config),
                }
            ),
            file,
            indent=2,
        )

    plt.figure(figsize=(10, 5))
    plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Train Loss")
    plt.plot(
        [row["epoch"] for row in history],
        [row["validation_loss"] for row in history],
        label="Validation Loss",
    )
    plt.axvline(best_epoch, color="black", linestyle="--", linewidth=1, label="Best Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(training_dir / "training_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    checkpoint = _torch_load_checkpoint(Path(model_checkpoint_path(model_name)), device)
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, best_metrics
