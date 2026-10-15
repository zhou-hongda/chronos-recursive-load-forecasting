"""Proj6 data and training adapter around Amazon's official Chronos-2 package.

The forecasting architecture and fit/predict APIs come from the installed
``chronos`` package; this module defines only the project experiment protocol.
"""

import csv
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from chronos import Chronos2Pipeline

from chronos_grouping import build_feature_groups, group_member_names
from experiment_config import TRAINING_OUTPUT_PATH


def _resolve_complete_pass_schedule(full_pass_steps, requested_steps):
    """Round the training budget up to complete ordered dataset passes."""
    full_pass_steps = int(full_pass_steps)
    requested_steps = int(requested_steps)
    if full_pass_steps <= 0:
        raise ValueError("A complete ordered pass must contain at least one step")

    ordered_passes = 1 if requested_steps <= 0 else math.ceil(
        requested_steps / full_pass_steps
    )
    return ordered_passes, ordered_passes * full_pass_steps


class ChronosWindowSequence(Sequence):
    """Lazy, time-ordered Chronos windows over a [time, series] array."""

    def __init__(
        self,
        data,
        context_length,
        prediction_length,
        stride,
        group_size,
        feature_groups=None,
        include_tail=False,
        target_mask=None,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.target_mask = (
            np.ones_like(self.data, dtype=bool)
            if target_mask is None
            else np.asarray(target_mask, dtype=bool)
        )
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.window_length = self.context_length + self.prediction_length
        self.stride = max(1, int(stride))
        self.group_size = max(1, int(group_size))

        if self.data.ndim != 2:
            raise ValueError(f"Expected [time, series] data, got {self.data.shape}")
        if self.target_mask.shape != self.data.shape:
            raise ValueError(
                f"target_mask must match data shape {self.data.shape}, got {self.target_mask.shape}"
            )
        if len(self.data) < self.window_length:
            raise ValueError(
                f"Need at least {self.window_length} rows for one Chronos window, got {len(self.data)}"
            )

        last_start = len(self.data) - self.window_length
        starts = list(range(0, last_start + 1, self.stride))
        if include_tail and starts[-1] != last_start:
            starts.append(last_start)
        self.window_starts = tuple(starts)

        if feature_groups is None:
            feature_groups = tuple(
                tuple(range(start, min(start + self.group_size, self.data.shape[1])))
                for start in range(0, self.data.shape[1], self.group_size)
            )
        self.feature_groups = tuple(
            tuple(int(feature_index) for feature_index in group)
            for group in feature_groups
        )
        flattened = [index for group in self.feature_groups for index in group]
        if sorted(flattened) != list(range(self.data.shape[1])):
            raise ValueError("Chronos feature groups must cover every series exactly once")
        if any(not group or len(group) > self.group_size for group in self.feature_groups):
            raise ValueError("Chronos feature groups contain an invalid group size")
        self.group_sizes = tuple(len(group) for group in self.feature_groups)
        self.sample_indices = tuple(
            (window_index, group_index)
            for window_index, time_start in enumerate(self.window_starts)
            for group_index, feature_indices in enumerate(self.feature_groups)
            if self.target_mask[
                time_start + self.context_length:time_start + self.window_length,
            ][:, feature_indices].any()
        )
        if not self.sample_indices:
            raise ValueError("Chronos windows contain no observed target values")

    @property
    def num_time_windows(self):
        return len(self.window_starts)

    @property
    def num_groups(self):
        return len(self.feature_groups)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        window_index, group_index = self.sample_indices[index]
        time_start = self.window_starts[window_index]
        feature_indices = self.feature_groups[group_index]
        window = self.data[
            time_start:time_start + self.window_length,
        ][:, feature_indices].copy()
        future_mask = self.target_mask[
            time_start + self.context_length:time_start + self.window_length,
        ][:, feature_indices]
        window[self.context_length:] = np.where(
            future_mask,
            window[self.context_length:],
            np.nan,
        )
        context = torch.from_numpy(np.ascontiguousarray(window.T))
        n_targets = context.shape[0]
        return {
            "context": context,
            "future_covariates": torch.full(
                (n_targets, self.prediction_length),
                fill_value=torch.nan,
                dtype=torch.float32,
            ),
            "n_targets": n_targets,
            "n_covariates": 0,
            "n_future_covariates": 0,
        }

    def batches_per_pass(self, batch_size):
        """Match Chronos batching, where batch size counts individual series."""
        batch_size = max(1, int(batch_size))
        batches = 0
        current_size = 0
        for _, group_index in self.sample_indices:
            current_size += self.group_sizes[group_index]
            if current_size >= batch_size:
                batches += 1
                current_size = 0
        return batches + int(current_size > 0)


class ChronosAdapter:
    """Chronos-2 LoRA fine-tuning and median multivariate forecasts."""

    def __init__(self, config, model_id="amazon/chronos-2"):
        self.seq_len = int(config["seq_len"])
        self.pred_len = int(config["pred_len"])
        self.train_stride = max(1, int(config.get("train_window_stride", 1)))
        self.validation_stride = max(1, int(config.get("chronos_validation_stride", 12)))
        self.feature_group_size = max(1, int(config.get("chronos_feature_group_size", 32)))
        self.feature_grouping = str(
            config.get("chronos_feature_grouping", "sequential")
        )
        self.transformer_meta_path = config.get("transformer_meta_path")
        self.feature_names = tuple(config.get("feature_names", []))
        self.saved_feature_group_members = config.get("chronos_feature_groups")
        self.feature_groups = None
        self.experiment_name = str(config.get("chronos_experiment_name", "Chronos"))
        self.inference_batch_size = max(
            1, int(config.get("chronos_inference_batch_size", 16))
        )
        self.recursive_stride = max(1, int(config.get("recursive_stride", 24)))
        self.recursive_clip_margin = max(
            0.0, float(config.get("recursive_clip_margin", 0.5))
        )
        if self.recursive_stride > self.pred_len:
            raise ValueError(
                f"recursive_stride={self.recursive_stride} exceeds pred_len={self.pred_len}"
            )
        self.seed = int(config.get("seed", 42))
        self.data_pipeline_version = config.get("data_pipeline_version")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = {
            "base_model": model_id,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "train_window_stride": self.train_stride,
            "validation_window_stride": self.validation_stride,
            "feature_group_size": self.feature_group_size,
            "feature_grouping": self.feature_grouping,
            "experiment_name": self.experiment_name,
            "inference_batch_size": self.inference_batch_size,
            "recursive_stride": self.recursive_stride,
            "recursive_clip_margin": self.recursive_clip_margin,
            "point_forecast_quantile": 0.5,
            "ordered_training": True,
            "seed": self.seed,
            "data_pipeline_version": self.data_pipeline_version,
            "data_quality": config.get("data_quality", {}),
            "feature_names": list(self.feature_names),
        }
        if self.feature_names:
            self._resolve_feature_groups(len(self.feature_names))

        device_map = "cuda" if self.device.type == "cuda" else "cpu"
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"Loading Chronos-2 ({model_id}) on {device_map}...")

        adapter_config_path = Path(model_id) / "adapter_config.json"
        if adapter_config_path.is_file():
            self.pipeline = self._load_lora_adapter(model_id, adapter_config_path, device_map, dtype)
        else:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=device_map,
                dtype=dtype,
            )
        print("Chronos-2 loaded.")

    def _resolve_feature_groups(self, num_features):
        num_features = int(num_features)
        if self.feature_groups is not None:
            if sum(map(len, self.feature_groups)) != num_features:
                raise ValueError(
                    "Chronos feature grouping does not match the current feature count"
                )
            return self.feature_groups
        if len(self.feature_names) != num_features:
            raise ValueError(
                "Chronos grouping requires feature_names in the same order as the data"
            )

        self.feature_groups = build_feature_groups(
            self.feature_names,
            max_group_size=self.feature_group_size,
            strategy=self.feature_grouping,
            metadata_path=self.transformer_meta_path,
            saved_group_members=self.saved_feature_group_members,
        )
        members = group_member_names(self.feature_names, self.feature_groups)
        self.strategy.update(
            {
                "feature_group_sizes": [len(group) for group in self.feature_groups],
                "feature_groups": members,
                "num_groups": len(self.feature_groups),
            }
        )
        return self.feature_groups

    @staticmethod
    def _load_lora_adapter(model_id, adapter_config_path, device_map, dtype):
        from peft import PeftModel

        with adapter_config_path.open("r", encoding="utf-8") as file:
            adapter_config = json.load(file)
        base_model_id = adapter_config.get("base_model_name_or_path", "amazon/chronos-2")
        pipeline = Chronos2Pipeline.from_pretrained(
            base_model_id,
            device_map=device_map,
            dtype=dtype,
        )
        pipeline.model = PeftModel.from_pretrained(
            pipeline.model,
            model_id,
            is_trainable=False,
        ).merge_and_unload()
        return pipeline

    @staticmethod
    def _use_ordered_batches(enabled):
        """Make Chronos cycle through its prepared inputs without random sampling."""
        from chronos.chronos2.dataset import Chronos2Dataset

        original_name = "_proj6_original_generate_train_batches"
        if not hasattr(Chronos2Dataset, original_name):
            setattr(Chronos2Dataset, original_name, Chronos2Dataset._generate_train_batches)

        original = getattr(Chronos2Dataset, original_name)
        if not enabled:
            Chronos2Dataset._generate_train_batches = original
            return

        def generate_ordered_batches(dataset):
            while True:
                yield from dataset._generate_sequential_batches()

        Chronos2Dataset._generate_train_batches = generate_ordered_batches

    def to(self, device):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.pipeline.model.to(self.device)
        return self

    @staticmethod
    def _reset_directory(target, root):
        resolved_root = Path(root).resolve()
        resolved_target = Path(target).resolve()
        resolved_target.relative_to(resolved_root)
        if resolved_target == resolved_root:
            raise ValueError("Refusing to reset the training output root")
        if resolved_target.exists():
            shutil.rmtree(resolved_target)
        resolved_target.mkdir(parents=True, exist_ok=True)

    def _recursive_validation_metrics(
        self,
        pipeline,
        train_data,
        validation_data,
        scaler=None,
        validation_observed_mask=None,
    ):
        """Score one candidate using the same recursive protocol as final testing."""
        if len(validation_data) <= self.seq_len:
            raise ValueError(
                f"Validation data needs more than {self.seq_len} rows, got {len(validation_data)}"
            )

        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        margin = (train_max - train_min) * self.recursive_clip_margin + 1e-6
        lower_bound = train_min - margin
        upper_bound = train_max + margin

        history = validation_data[:self.seq_len].copy()
        actuals = validation_data[self.seq_len:]
        predictions = []
        nonfinite_values = 0
        fallback_blocks = 0
        clipped_values = 0

        pipeline.model.eval()
        with torch.no_grad():
            for start in range(0, len(actuals), self.recursive_stride):
                context = history[-self.seq_len:]
                model_input = torch.from_numpy(context[None]).to(self.device)
                block = self._predict_chunked(model_input, pipeline=pipeline)[0].cpu().numpy()
                keep = min(self.recursive_stride, len(actuals) - start)
                used_block = block[:keep]

                nonfinite_mask = ~np.isfinite(used_block)
                nonfinite_values += int(np.sum(nonfinite_mask))
                if np.any(nonfinite_mask):
                    fallback = np.repeat(context[-1:], keep, axis=0)
                    used_block = np.where(nonfinite_mask, fallback, used_block)
                    fallback_blocks += 1

                clipped_mask = (used_block < lower_bound[None]) | (used_block > upper_bound[None])
                clipped_values += int(np.sum(clipped_mask))
                used_block = np.clip(used_block, lower_bound[None], upper_bound[None])
                predictions.append(used_block)
                history = np.concatenate([history, used_block], axis=0)

        predictions = np.concatenate(predictions, axis=0)
        actuals = actuals[:len(predictions)]
        if validation_observed_mask is None:
            target_mask = np.ones_like(actuals, dtype=bool)
        else:
            validation_observed_mask = np.asarray(validation_observed_mask, dtype=bool)
            if validation_observed_mask.shape != validation_data.shape:
                raise ValueError(
                    "validation_observed_mask must match validation_data shape "
                    f"{validation_data.shape}, got {validation_observed_mask.shape}"
                )
            target_mask = validation_observed_mask[
                self.seq_len:self.seq_len + len(predictions)
            ]
        if scaler is not None:
            metric_predictions = scaler.inverse_transform(predictions)
            metric_actuals = scaler.inverse_transform(actuals)
        else:
            metric_predictions = predictions
            metric_actuals = actuals
        valid = target_mask & np.isfinite(metric_predictions) & np.isfinite(metric_actuals)
        if not np.any(valid):
            raise ValueError("Recursive validation contains no observed target values")
        metric_predictions = metric_predictions.astype(np.float64)[valid]
        metric_actuals = metric_actuals.astype(np.float64)[valid]
        residual = metric_predictions - metric_actuals
        absolute_residual = np.abs(residual)
        total_variation = np.sum((metric_actuals - np.mean(metric_actuals)) ** 2)
        metrics = {
            "mse": float(np.mean(residual ** 2)),
            "mae": float(np.mean(absolute_residual)),
            "wape": float(
                np.sum(absolute_residual) / max(np.sum(np.abs(metric_actuals)), 1e-8) * 100
            ),
            "r2": float(1.0 - np.sum(residual ** 2) / max(total_variation, 1e-8)),
            "forecast_steps": int(len(predictions)),
            "evaluated_values": int(valid.sum()),
            "excluded_missing_values": int(valid.size - valid.sum()),
            "nonfinite_values": nonfinite_values,
            "fallback_blocks": fallback_blocks,
            "clipped_values": clipped_values,
        }
        return metrics

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fit(
        self,
        train_data,
        validation_data,
        batch_size=0,
        learning_rate=1e-5,
        num_steps=0,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        scaler=None,
        train_observed_mask=None,
        validation_observed_mask=None,
    ):
        """Fine-tune ordered windows and keep the best direct-validation checkpoint."""
        train_data = np.asarray(train_data, dtype=np.float32)
        validation_data = np.asarray(validation_data, dtype=np.float32)
        if train_data.ndim != 2 or validation_data.ndim != 2:
            raise ValueError("Chronos train and validation data must be [time, series] arrays")
        if train_data.shape[1] != validation_data.shape[1]:
            raise ValueError("Train and validation feature counts do not match")
        train_observed_mask = (
            np.ones_like(train_data, dtype=bool)
            if train_observed_mask is None
            else np.asarray(train_observed_mask, dtype=bool)
        )
        validation_observed_mask = (
            np.ones_like(validation_data, dtype=bool)
            if validation_observed_mask is None
            else np.asarray(validation_observed_mask, dtype=bool)
        )
        if train_observed_mask.shape != train_data.shape:
            raise ValueError("train_observed_mask must match train_data")
        if validation_observed_mask.shape != validation_data.shape:
            raise ValueError("validation_observed_mask must match validation_data")

        feature_groups = self._resolve_feature_groups(train_data.shape[1])
        train_inputs = ChronosWindowSequence(
            train_data,
            self.seq_len,
            self.pred_len,
            stride=self.train_stride,
            group_size=self.feature_group_size,
            feature_groups=feature_groups,
            target_mask=train_observed_mask,
        )
        validation_inputs = ChronosWindowSequence(
            validation_data,
            self.seq_len,
            self.pred_len,
            stride=self.validation_stride,
            group_size=self.feature_group_size,
            feature_groups=feature_groups,
            include_tail=True,
            target_mask=validation_observed_mask,
        )

        num_series = train_data.shape[1]
        effective_batch_size = num_series if int(batch_size) <= 0 else int(batch_size)
        full_pass_steps = train_inputs.batches_per_pass(effective_batch_size)
        requested_steps = int(num_steps)
        planned_ordered_passes, total_steps = _resolve_complete_pass_schedule(
            full_pass_steps,
            requested_steps,
        )
        eval_steps = full_pass_steps
        save_steps = full_pass_steps
        patience_ordered_passes = max(1, int(early_stopping_patience))
        early_stopping_threshold = max(0.0, float(early_stopping_min_delta))
        lora_config = {
            "r": max(1, int(lora_rank)),
            "lora_alpha": max(1, int(lora_alpha)),
            "lora_dropout": float(lora_dropout),
            "target_modules": [
                "self_attention.q",
                "self_attention.k",
                "self_attention.v",
                "self_attention.o",
                "output_patch_embedding.output_layer",
            ],
        }

        print(
            "Chronos fine-tuning: "
            f"series={num_series}, grouping={self.feature_grouping}, "
            f"groups={train_inputs.num_groups}, max_group_size={self.feature_group_size}, "
            f"train_windows={train_inputs.num_time_windows}, "
            f"validation_windows={validation_inputs.num_time_windows}, "
            f"batch_size={effective_batch_size}, steps_per_pass={full_pass_steps}, "
            f"ordered_passes={planned_ordered_passes}, total_steps={total_steps}, "
            f"validate/save=pass_boundary, "
            f"early_stopping_patience={patience_ordered_passes} passes"
        )
        if requested_steps > 0 and requested_steps != total_steps:
            print(
                "Chronos step budget aligned to complete ordered passes: "
                f"requested={requested_steps}, actual={total_steps}."
            )
        print(
            f"LoRA: rank={lora_config['r']}, alpha={lora_config['lora_alpha']}, "
            f"dropout={lora_config['lora_dropout']}, learning_rate={learning_rate}"
        )

        self.strategy.update(
            {
                "num_series": num_series,
                "num_groups": train_inputs.num_groups,
                "train_windows": train_inputs.num_time_windows,
                "validation_windows": validation_inputs.num_time_windows,
                "train_samples_with_observed_targets": len(train_inputs),
                "validation_samples_with_observed_targets": len(validation_inputs),
                "validation_tail_anchored": True,
                "batch_size": effective_batch_size,
                "learning_rate": float(learning_rate),
                "requested_num_steps": requested_steps,
                "num_steps": total_steps,
                "full_ordered_pass_steps": full_pass_steps,
                "planned_ordered_passes": planned_ordered_passes,
                "minimum_training_coverage": "one_complete_ordered_pass",
                "eval_steps": eval_steps,
                "validation_schedule": "complete_ordered_pass_boundaries",
                "planned_pass_boundary_validations": planned_ordered_passes,
                "checkpoint_save_steps": save_steps,
                "checkpoint_selection": (
                    "minimum_chronos_direct_validation_loss_at_complete_pass_boundaries"
                ),
                "direct_validation_objective": "chronos_native_eval_loss",
                "early_stopping_patience_ordered_passes": patience_ordered_passes,
                "early_stopping_min_delta": early_stopping_threshold,
                "finetune_mode": "lora",
                "lora_config": lora_config,
                "lr_scheduler_type": str(lr_scheduler_type),
                "warmup_ratio": float(warmup_ratio),
                "weight_decay": float(weight_decay),
                "max_grad_norm": float(max_grad_norm),
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from transformers import EarlyStoppingCallback, TrainerCallback

        class ValidationHistoryCallback(TrainerCallback):
            def __init__(self):
                self.history = []
                self.trainer_best_checkpoint = None
                self.trainer_best_metric = None
                self.completed_steps = 0

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                metrics = metrics or {}
                step = int(state.global_step)
                if step == 0:
                    return
                if step % full_pass_steps != 0:
                    raise RuntimeError(
                        "Chronos validation was triggered inside an ordered pass"
                    )
                self.history.append(
                    {
                        "step": step,
                        "ordered_pass": step // full_pass_steps,
                        "eval_loss": float(metrics.get("eval_loss", float("nan"))),
                    }
                )

            def on_train_end(self, args, state, control, **kwargs):
                self.completed_steps = int(state.global_step)
                if state.best_model_checkpoint:
                    self.trainer_best_checkpoint = Path(state.best_model_checkpoint).name
                if state.best_metric is not None:
                    self.trainer_best_metric = float(state.best_metric)

        validation_history = ValidationHistoryCallback()
        training_dir = Path(TRAINING_OUTPUT_PATH) / self.experiment_name
        self._reset_directory(training_dir, TRAINING_OUTPUT_PATH)
        work_dir = training_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        self._use_ordered_batches(True)
        try:
            finetuned_pipeline = self.pipeline.fit(
                inputs=train_inputs,
                validation_inputs=validation_inputs,
                prediction_length=self.pred_len,
                finetune_mode="lora",
                lora_config=lora_config,
                context_length=self.seq_len,
                learning_rate=float(learning_rate),
                num_steps=total_steps,
                batch_size=effective_batch_size,
                min_past=self.seq_len,
                output_dir=str(work_dir),
                finetuned_ckpt_name="trainer_best",
                convert_inputs=False,
                eval_steps=eval_steps,
                save_steps=save_steps,
                logging_steps=eval_steps,
                callbacks=[
                    validation_history,
                    EarlyStoppingCallback(
                        early_stopping_patience=patience_ordered_passes,
                        early_stopping_threshold=early_stopping_threshold,
                    ),
                ],
                lr_scheduler_type=str(lr_scheduler_type),
                warmup_ratio=float(warmup_ratio),
                weight_decay=float(weight_decay),
                max_grad_norm=float(max_grad_norm),
                save_total_limit=1,
                greater_is_better=False,
                overwrite_output_dir=True,
                remove_printer_callback=True,
                seed=self.seed,
                data_seed=self.seed,
            )
        finally:
            self._use_ordered_batches(False)

        finite_history = [
            item for item in validation_history.history if math.isfinite(item["eval_loss"])
        ]
        self.strategy["validation_history"] = validation_history.history
        self.strategy["completed_pass_boundary_validations"] = len(
            validation_history.history
        )
        self.strategy["completed_steps"] = validation_history.completed_steps
        if validation_history.completed_steps % full_pass_steps != 0:
            raise RuntimeError(
                "Chronos training ended inside an ordered pass; refusing to select "
                "a partially covered checkpoint"
            )
        self.strategy["completed_ordered_passes"] = (
            validation_history.completed_steps // full_pass_steps
        )
        self.strategy["early_stopped"] = validation_history.completed_steps < total_steps
        if not finite_history:
            raise RuntimeError("Chronos training did not produce a finite direct-validation loss")
        best_validation = min(finite_history, key=lambda item: item["eval_loss"])
        self.strategy["best_validation_step"] = best_validation["step"]
        self.strategy["best_validation_loss"] = best_validation["eval_loss"]
        selected_checkpoint = validation_history.trainer_best_checkpoint
        selected_loss = validation_history.trainer_best_metric
        if selected_checkpoint is None:
            selected_checkpoint = f"checkpoint-{best_validation['step']}"
        if selected_loss is None:
            selected_loss = best_validation["eval_loss"]
        selected_step = int(best_validation["step"])
        if selected_step < full_pass_steps or selected_step % full_pass_steps != 0:
            raise RuntimeError(
                "Chronos selected a checkpoint without complete ordered-data coverage"
            )

        self.pipeline = finetuned_pipeline
        self.strategy["selected_checkpoint"] = selected_checkpoint
        self.strategy["selected_checkpoint_step"] = selected_step
        self.strategy["selected_ordered_pass"] = selected_step // full_pass_steps
        self.strategy["selected_direct_validation_loss"] = selected_loss
        self.strategy["recursive_validation_scale"] = (
            "original_load_scale" if scaler is not None else "standardized_training_scale"
        )

        diagnostic = {
            "checkpoint": selected_checkpoint,
            "direct_validation_loss": selected_loss,
            "role": "diagnostic_only",
            "status": "failed",
            "error": "",
            "mse": None,
            "mae": None,
            "wape": None,
            "r2": None,
            "forecast_steps": None,
            "evaluated_values": None,
            "excluded_missing_values": None,
            "nonfinite_values": None,
            "fallback_blocks": None,
            "clipped_values": None,
        }
        try:
            recursive_metrics = self._recursive_validation_metrics(
                self.pipeline,
                train_data,
                validation_data,
                scaler=scaler,
                validation_observed_mask=validation_observed_mask,
            )
            diagnostic.update({"status": "ok", **recursive_metrics})
            print(
                "Chronos recursive validation diagnostic: "
                f"MSE={recursive_metrics['mse']:.6f}, "
                f"MAE={recursive_metrics['mae']:.6f}, "
                f"R2={recursive_metrics['r2']:.4f}, "
                f"clipped={recursive_metrics['clipped_values']}"
            )
        except Exception as exc:
            diagnostic["error"] = str(exc)
            print(f"Chronos recursive validation diagnostic failed: {exc}")
        self.strategy["recursive_validation_role"] = "diagnostic_only"
        self.strategy["recursive_validation_diagnostic"] = diagnostic

        self._write_csv(training_dir / "validation_history.csv", validation_history.history)
        self._write_csv(training_dir / "recursive_validation_diagnostics.csv", [diagnostic])
        with (training_dir / "training_summary.json").open("w", encoding="utf-8") as file:
            json.dump(self.strategy, file, indent=2)
        shutil.rmtree(work_dir)
        print(
            "Chronos fine-tuning complete; selected "
            f"{selected_checkpoint} by native direct-validation loss."
        )

    def save_pretrained(self, save_directory):
        """Merge LoRA weights and atomically save a standalone Chronos model."""
        model = self.pipeline.model
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
            self.pipeline = Chronos2Pipeline(model=model)

        target = Path(save_directory)
        temporary = target.with_name(f"{target.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        self.pipeline.save_pretrained(temporary)
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)

        with (target / "chronos_strategy.json").open("w", encoding="utf-8") as file:
            json.dump(self.strategy, file, indent=2)

    def _predict_chunked(self, x, pipeline=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, features], got {tuple(x.shape)}")

        batch_size, _, num_features = x.shape
        device = x.device
        contexts = x.detach().permute(0, 2, 1).cpu()
        feature_groups = self._resolve_feature_groups(num_features)
        grouped_inputs = []
        for batch_index in range(batch_size):
            for feature_indices in feature_groups:
                grouped_inputs.append(contexts[batch_index, list(feature_indices)])

        active_pipeline = self.pipeline if pipeline is None else pipeline
        if active_pipeline is None:
            raise RuntimeError("Chronos pipeline is not loaded")
        quantile_forecasts, _ = active_pipeline.predict_quantiles(
            inputs=grouped_inputs,
            prediction_length=self.pred_len,
            quantile_levels=[0.5],
            batch_size=self.inference_batch_size,
            context_length=self.seq_len,
        )

        group_predictions = []
        for forecast_item in quantile_forecasts:
            if isinstance(forecast_item, torch.Tensor):
                forecast_item = forecast_item.detach().cpu().numpy()
            quantile = np.asarray(forecast_item, dtype=np.float32)
            if quantile.ndim == 3:
                if quantile.shape[-1] != 1:
                    raise RuntimeError(f"Expected one quantile, got shape {quantile.shape}")
                quantile = quantile[..., 0]
            if quantile.shape[-1] != self.pred_len:
                raise RuntimeError(f"Invalid Chronos forecast shape: {quantile.shape}")
            group_predictions.append(quantile.reshape(-1, self.pred_len))

        expected_groups = batch_size * len(feature_groups)
        if len(group_predictions) != expected_groups:
            raise RuntimeError(
                f"Chronos returned {len(group_predictions)} groups for {expected_groups} inputs"
            )

        predictions = np.empty(
            (batch_size, num_features, self.pred_len),
            dtype=np.float32,
        )
        prediction_index = 0
        for batch_index in range(batch_size):
            for feature_indices in feature_groups:
                group_prediction = group_predictions[prediction_index]
                prediction_index += 1
                if len(group_prediction) != len(feature_indices):
                    raise RuntimeError(
                        "Chronos returned a feature count that does not match its input group"
                    )
                predictions[batch_index, list(feature_indices)] = group_prediction
        return torch.from_numpy(predictions).permute(0, 2, 1).to(device)

    def predict(self, x):
        return self._predict_chunked(x)

    def eval(self):
        self.pipeline.model.eval()
        return self

    def __call__(self, x, *args, **kwargs):
        return self.predict(x)
