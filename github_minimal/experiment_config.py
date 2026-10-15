RAW_DATA_PATH = "data/transformer_raw.csv"
PROCESSED_DATA_PATH = "data/representative_data.csv"
RESULTS_PATH = "results"
CHECKPOINTS_PATH = f"{RESULTS_PATH}/checkpoints"
TRAINING_OUTPUT_PATH = f"{RESULTS_PATH}/training"
PAPER_OUTPUT_PATH = f"{RESULTS_PATH}/paper"
CHRONOS_CHECKPOINT_PATH = f"{CHECKPOINTS_PATH}/Chronos"
CHRONOS_BASE_MODEL_ID = "amazon/chronos-2"

FINETUNED_CHRONOS = "Chronos"
ZERO_SHOT_CHRONOS = "Chronos-ZeroShot"

# Main-paper entries backed by three official forecasting architectures.
# Chronos-2 is evaluated in both zero-shot and fine-tuned settings.
OFFICIAL_SOTA_EXPERIMENTS = (
    ZERO_SHOT_CHRONOS,
    FINETUNED_CHRONOS,
    "iTransformer",
    "Autoformer",
)
BASELINE_EXPERIMENTS = (
    "MLP",
    "RNN",
    "GRU",
    "LSTM",
    "TCN",
)
DEFAULT_EXPERIMENT_MODELS = (
    *OFFICIAL_SOTA_EXPERIMENTS,
    *BASELINE_EXPERIMENTS,
)

SUPPORTED_EXPERIMENT_MODELS = DEFAULT_EXPERIMENT_MODELS

SEQ_LEN = 168
PRED_LEN = 24
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TRAIN_WINDOW_STRIDE = 1
RECURSIVE_STRIDE = 24
SEED = 42

NON_CHRONOS_EPOCHS = 15
NON_CHRONOS_BATCH_SIZE = 8
NON_CHRONOS_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4
GRADIENT_CLIP_NORM = 1.0


def model_checkpoint_path(model_name):
    return f"{CHECKPOINTS_PATH}/{model_name}/best_model.pth"


def chronos_checkpoint_path(model_name):
    if model_name == FINETUNED_CHRONOS:
        return CHRONOS_CHECKPOINT_PATH
    raise ValueError(f"No fine-tuned Chronos checkpoint is defined for {model_name!r}")
