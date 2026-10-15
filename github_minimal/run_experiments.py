import argparse
import subprocess
import sys

from experiment_config import (
    DEFAULT_EXPERIMENT_MODELS,
    SUPPORTED_EXPERIMENT_MODELS,
    ZERO_SHOT_CHRONOS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed paper protocol: train selected models, then compare them"
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Models to compare; defaults to the current audited benchmark suite",
    )
    parser.add_argument("--compare-only", action="store_true", help="Use existing checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    invalid_models = [
        model_name
        for model_name in args.models
        if model_name not in SUPPORTED_EXPERIMENT_MODELS
    ]
    if invalid_models:
        parser.error(
            f"unknown model(s): {', '.join(invalid_models)}; "
            f"choose from {', '.join(SUPPORTED_EXPERIMENT_MODELS)}"
        )
    return args


def run_main(*arguments):
    command = [sys.executable, "main.py", *arguments]
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    requested_models = args.models or DEFAULT_EXPERIMENT_MODELS
    models = list(dict.fromkeys(requested_models))

    if not args.compare_only:
        for model_name in models:
            if model_name == ZERO_SHOT_CHRONOS:
                continue
            run_main(
                "--mode",
                "train",
                "--model",
                model_name,
                "--seed",
                str(args.seed),
            )

    run_main(
        "--mode",
        "compare",
        "--compare_models",
        *models,
        "--seed",
        str(args.seed),
    )


if __name__ == "__main__":
    main()
