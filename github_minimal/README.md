# Recursive Load Forecasting with Chronos-2

Code for **Extending Short-Term Load Forecasting to Long Horizons through
Recursive Prediction with Chronos-2**.

This minimal repository contains the nine manuscript experiment entries:
`Chronos-ZeroShot`, `Chronos`, `iTransformer`, `Autoformer`, `MLP`, `RNN`, `GRU`,
`LSTM`, and `TCN`. The first two share the official Chronos-2 architecture;
`Chronos` denotes LoRA adaptation. Historical optional experiments are excluded.

## Install

Use Python 3.10. Install a suitable PyTorch 2.8.0 CPU/CUDA build first, then:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -B -m unittest discover -s tests -v
```

Chronos is pinned to official commit
`f951d9aefa06f5389b2ed6b0e51fd5a1a4cf194b`. Its version string remains 2.2.2,
but its training interface differs from the tagged/PyPI 2.2.2 release; keep the
source URL in `requirements.txt` rather than replacing it with that version pin.
The package supplies the architecture and native training loss. The local
adapter controls the experiment protocol, not a substitute forecasting network.

## Data

The private observations, trained weights, and original experiment outputs are
not distributed. Reproducing the paper requires authorized access to those data.
The unit tests use synthetic inputs and do not reproduce numerical paper results.

Create `data/` and place the authorized observations at `data/transformer_raw.csv`
with columns `DATETIME`, `TRANSFORMER_ID`, and `LOAD`. For example, the schema is:

```text
DATETIME,TRANSFORMER_ID,LOAD
2022-01-01 00:00:00,example_device,42.0
```

This example is illustrative, not a real observation. The default date range is
2022-01-01 through 2023-11-09. `main.py` creates a wide observed-data cache,
retains missing observations, selects series using training data only, fills
inputs causally, and fits per-series scaling on training inputs. Missing targets
are excluded from the losses and evaluation metrics.

## Run

Run commands from the repository root:

```bash
# Train and compare all nine entries
python run_experiments.py --seed 42

# Train Chronos only
python main.py --mode train --model Chronos --seed 42

# Compare existing compatible checkpoints without training
python run_experiments.py --compare-only --seed 42

# Plot Figures 2-4 after the full manuscript evaluation
python plot_paper_figures.py
```

Chronos needs access to `amazon/chronos-2` or its local cache. The original entry
point has an HF_ENDPOINT mirror default; set `HF_ENDPOINT=https://huggingface.co`
before running to use Hugging Face directly. Outputs go to `results/`, which is
excluded from Git. Repeated training/evaluation can replace previous outputs.

## Protocol

- Chronological split: 70% training, 10% validation, 20% test. In configuration,
  `train_ratio=0.8` specifies the test boundary and includes validation.
- Input 168 h, target 24 h, ordered training windows with stride 1, no shuffle.
- Test stride 24: append predicted blocks without refreshing true observations.
- Non-Chronos models: at most 15 epochs, batch 8, learning rate 1e-3, patience 3;
  loss is `0.7 * masked MSE + 0.3 * masked MAE` on standardized targets.
- Chronos: rank 8, alpha 16, LoRA dropout 0.05, learning rate 1e-5, one complete
  ordered pass by default, consecutive variable groups of at most 32 series.
- Direct validation uses observed history without prediction feedback; window
  stride is 1 for other models and 12 for Chronos. Each family keeps its training
  loss for checkpoint selection. Recursive validation is diagnostic only.
- Point forecasts use the Chronos 0.5 quantile. All models use training-derived
  recursive clipping bounds with margin 0.5; clipping events are recorded.
- Metrics are computed on observed targets after inverse scaling. Relative MSE
  uses one common divisor, not per-device standardization.

The original study retained 366 series, with 11,390/1,627/3,255 hourly target
positions across the three intervals. Manuscript plotting checks these study
dimensions and example IDs; it needs adaptation for a different dataset.

## Layout

| Path | Purpose |
|---|---|
| `main.py`, `run_experiments.py` | Single and batch experiment entry points |
| `experiment_config.py` | Paths, model registry, and defaults |
| `preprocessing.py` | Selection, filling, scaling, masks, and windows |
| `chronos_adapter.py`, `chronos_grouping.py` | Official Chronos integration and sequential groups |
| `train.py`, `model_utils.py` | Model construction, training, validation, early stopping |
| `evaluate.py` | Recursive inference, metric exports, and experiment manifest |
| `model/`, `layers/`, `utils/` | Models and their required shared components |
| `plot_paper_figures.py`, `paper_figures/` | Matplotlib manuscript figures |
| `tests/test_core.py` | Synthetic core checks; no model download |
| `third_party/licenses/` | Upstream license texts |

## License and Sources

Original project code is MIT-licensed; see [LICENSE](LICENSE). Third-party
modules retain their own terms and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This code license does not grant rights to private data, weights, or manuscripts.
