import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch

import evaluate
import preprocessing
import train
from experiment_config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    FINETUNED_CHRONOS,
    GRADIENT_CLIP_NORM,
    NON_CHRONOS_BATCH_SIZE,
    NON_CHRONOS_EPOCHS,
    NON_CHRONOS_LEARNING_RATE,
    PRED_LEN,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RECURSIVE_STRIDE,
    SEED,
    SEQ_LEN,
    SUPPORTED_EXPERIMENT_MODELS,
    TRAIN_RATIO,
    TRAIN_WINDOW_STRIDE,
    VALIDATION_RATIO,
    ZERO_SHOT_CHRONOS,
    chronos_checkpoint_path,
)


warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(description="Chronological multivariate load forecasting")
    parser.add_argument(
        "--model",
        default="LSTM",
        choices=SUPPORTED_EXPERIMENT_MODELS,
        help="Current audited model entry",
    )
    parser.add_argument("--mode", default="train", choices=["train", "predict", "compare"])
    parser.add_argument(
        "--compare_models",
        nargs="+",
        choices=SUPPORTED_EXPERIMENT_MODELS,
        help="Current audited entries included in comparison mode",
    )
    parser.add_argument("--epochs", type=int, default=NON_CHRONOS_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=NON_CHRONOS_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=NON_CHRONOS_LEARNING_RATE)
    parser.add_argument("--early_stopping_patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--early_stopping_min_delta", type=float, default=EARLY_STOPPING_MIN_DELTA)
    parser.add_argument("--gradient_clip_norm", type=float, default=GRADIENT_CLIP_NORM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--recursive_stride", type=int, default=RECURSIVE_STRIDE)
    parser.add_argument("--recursive_clip_margin", type=float, default=0.5)

    chronos = parser.add_argument_group("Chronos LoRA")
    chronos.add_argument("--chronos_learning_rate", type=float, default=1e-5)
    chronos.add_argument(
        "--chronos_batch_size",
        type=int,
        default=0,
        help="Number of series per batch; 0 uses all series from one time window",
    )
    chronos.add_argument(
        "--chronos_num_steps",
        type=int,
        default=0,
        help=(
            "Requested optimization steps; 0 performs one complete ordered pass, "
            "and positive values are rounded up to a complete-pass boundary"
        ),
    )
    chronos.add_argument("--chronos_lora_rank", type=int, default=8)
    chronos.add_argument("--chronos_lora_alpha", type=int, default=16)
    chronos.add_argument("--chronos_lora_dropout", type=float, default=0.05)
    chronos.add_argument(
        "--chronos_validation_stride",
        type=int,
        default=12,
        help="Stride between validation windows; 12 gives 50%% overlap for 24h targets",
    )
    chronos.add_argument("--chronos_feature_group_size", type=int, default=32)
    chronos.add_argument("--chronos_inference_batch_size", type=int, default=16)
    chronos.add_argument(
        "--chronos_lr_scheduler_type",
        default="linear",
        choices=["linear", "cosine", "constant_with_warmup"],
    )
    chronos.add_argument("--chronos_warmup_ratio", type=float, default=0.05)
    chronos.add_argument("--chronos_weight_decay", type=float, default=0.0)
    chronos.add_argument("--chronos_max_grad_norm", type=float, default=1.0)
    return parser


def build_config(args):
    return {
        "data_mode": "multivariate",
        "data_pipeline_version": preprocessing.DATA_PIPELINE_VERSION,
        "data_path": PROCESSED_DATA_PATH,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "train_window_stride": TRAIN_WINDOW_STRIDE,
        "recursive_stride": args.recursive_stride,
        "recursive_clip_margin": args.recursive_clip_margin,
        "dropout": 0.1,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-5,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
        "chronos_learning_rate": args.chronos_learning_rate,
        "chronos_batch_size": args.chronos_batch_size,
        "chronos_num_steps": args.chronos_num_steps,
        "chronos_lora_rank": args.chronos_lora_rank,
        "chronos_lora_alpha": args.chronos_lora_alpha,
        "chronos_lora_dropout": args.chronos_lora_dropout,
        "chronos_validation_stride": args.chronos_validation_stride,
        "chronos_feature_group_size": args.chronos_feature_group_size,
        "chronos_feature_grouping": "sequential",
        "chronos_experiment_name": args.model,
        "chronos_inference_batch_size": args.chronos_inference_batch_size,
        "chronos_lr_scheduler_type": args.chronos_lr_scheduler_type,
        "chronos_warmup_ratio": args.chronos_warmup_ratio,
        "chronos_weight_decay": args.chronos_weight_decay,
        "chronos_max_grad_norm": args.chronos_max_grad_norm,
    }


def prepare_data():
    preprocessing.generate_processed_file(
        raw_path=Path(RAW_DATA_PATH),
        output_path=Path(PROCESSED_DATA_PATH),
    )


def train_chronos(config, device, model_name):
    from chronos_adapter import ChronosAdapter

    train_data, val_data, _, scaler, _, split_info = preprocessing.load_multivariate_data(
        file_path=config["data_path"],
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
        seq_len=config["seq_len"],
    )
    config["input_dim"] = train_data.shape[1]
    config["feature_names"] = split_info["feature_names"]
    config["data_quality"] = split_info["data_quality"]
    adapter = ChronosAdapter(config).to(device)
    adapter.fit(
        train_data=train_data,
        validation_data=val_data,
        batch_size=config["chronos_batch_size"],
        learning_rate=config["chronos_learning_rate"],
        num_steps=config["chronos_num_steps"],
        lora_rank=config["chronos_lora_rank"],
        lora_alpha=config["chronos_lora_alpha"],
        lora_dropout=config["chronos_lora_dropout"],
        lr_scheduler_type=config["chronos_lr_scheduler_type"],
        warmup_ratio=config["chronos_warmup_ratio"],
        weight_decay=config["chronos_weight_decay"],
        max_grad_norm=config["chronos_max_grad_norm"],
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_min_delta=config["early_stopping_min_delta"],
        scaler=scaler,
        train_observed_mask=split_info["train_observed_mask"],
        validation_observed_mask=split_info["validation_observed_mask"],
    )
    checkpoint_path = chronos_checkpoint_path(model_name)
    adapter.save_pretrained(checkpoint_path)
    print(f"[{model_name}] Saved direct-validation best model to {checkpoint_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    config = build_config(args)
    prepare_data()

    if args.mode == "compare":
        if not args.compare_models:
            parser.error("--compare_models is required in compare mode")
        evaluate.compare_models(args.compare_models, config)
        return

    if args.mode == "predict":
        evaluate.predict_mode(args.model, config)
        return

    if args.model == ZERO_SHOT_CHRONOS:
        print(f"[{ZERO_SHOT_CHRONOS}] No training is required; use --mode predict or compare.")
    elif args.model == FINETUNED_CHRONOS:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_chronos(config, device, args.model)
    else:
        train.train_model(config, args.model)


if __name__ == "__main__":
    main()
