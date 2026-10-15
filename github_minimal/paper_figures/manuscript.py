"""Reproducible manuscript Figures 2-4, using existing experiment outputs.

No model is trained and no test forecast is changed by this script.
Exploratory data clustering uses the training period only.
"""

from pathlib import Path
import hashlib
from importlib.metadata import version
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper_figures.style import (
    BAR_BLUE, CLUSTER_COLORS, CLUSTER_MARKERS, GRID, INK, LABELS, LEGEND_SIZE,
    MODELS, NPG_COLORS, TITLE_SIZE, STYLES, apply_style, display_mse, export_figure,
    mse_description, mse_label, style_axes,
)

OUT = ROOT / "results/paper/manuscript_figures"
DATA = OUT / "source_data"
QA = OUT / "_qa"
RESULTS = ROOT / "results/paper/model_comparison"
SEED = 42
EXPORT_DPI = 600
EXPORT_FORMATS = ("png", "pdf", "svg")
MSE_DISPLAY = "relative"
MSE_REFERENCE = None
FONT = apply_style()


def title(ax, text, size=TITLE_SIZE):
    ax.set_title(text, loc="left", fontsize=size, fontweight="bold", pad=11)


def plot_style(ax):
    style_axes(ax)


def save(fig, stem):
    export_figure(fig, OUT / stem, dpi=EXPORT_DPI, formats=EXPORT_FORMATS, preview_dir=QA)
    plt.close(fig)


def evidence(figures):
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text())
    names = manifest["evaluation_config"]["feature_names"]
    raw = train = predictions = None
    if 2 in figures:
        config = manifest["evaluation_config"]
        raw = pd.read_csv(ROOT / config["data_path"], usecols=["DATETIME"] + names,
                          parse_dates=["DATETIME"]).set_index("DATETIME").sort_index()[names]
        raw = raw.apply(pd.to_numeric, errors="coerce")
        assert raw.shape == (16272, 366) and not raw.index.has_duplicates
        # Use the same boundary formula as preprocessing.load_multivariate_data.
        train_end = int(len(raw) * (config["train_ratio"] - config["validation_ratio"]))
        train = raw.iloc[:train_end]
        assert train_end == 11390
    steps = pd.read_csv(RESULTS / "recursive_step_metrics.csv")
    metrics = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("Model")
    if 4 in figures:
        columns = ["Time"]
        for name in ["0-0-0", "0-0-1"]:
            columns += [f"{name}_True"] + [f"{m}_{name}_Pred" for m in MODELS]
        predictions = pd.read_csv(RESULTS / "predictions.csv", usecols=columns, parse_dates=["Time"])
        assert len(predictions) == manifest["dataset"]["forecast_steps"] == 3255
        assert predictions.Time.iloc[0] == pd.Timestamp(manifest["dataset"]["forecast_start"])
    for model in MODELS:
        last = steps.loc[steps.Model == model].sort_values("end_hour").iloc[-1]
        for key in ["mse", "mae", "wape", "r2"]:
            assert np.isclose(last[f"cumulative_{key}"], metrics.loc[model, key])
    return raw, train, steps, metrics, predictions


def mse_reference(metrics):
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text())
    reference = manifest["display_conventions"]["relative_mse_reference"]
    assert np.isclose(reference, metrics.loc[MODELS, "mse"].min())
    assert np.allclose(metrics.loc[MODELS, "relative_mse"], metrics.loc[MODELS, "mse"] / reference)
    return float(reference)


def cluster_training_profiles(weekly, daily):
    """Describe load-profile heterogeneity using a four-cluster Ward partition."""
    if (not weekly.index.equals(daily.index) or not np.isfinite(weekly.to_numpy()).all()
            or not np.isfinite(daily.to_numpy()).all()):
        raise ValueError("Clustering requires aligned, finite training profiles")
    x = StandardScaler().fit_transform(weekly)
    diagnostics, stability = [], []
    with threadpool_limits(limits=1):
        labels = None
        for k in range(2, 7):
            partition = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
            counts = np.bincount(partition, minlength=k)
            diagnostics.append({
                "method": "Ward", "k": k,
                "silhouette": float(silhouette_score(x, partition)),
                "min_cluster_size": int(counts.min()), "max_cluster_size": int(counts.max()),
                "used": k == 4,
            })
            if k == 4:
                labels = partition
        order = sorted(range(4), key=lambda c: (int(daily.iloc[labels == c].median().idxmax()), c))
        label_map = {old: new for new, old in enumerate(order)}
        labels = np.array([label_map[int(c)] for c in labels])
        # Ward has no random initialization; check sensitivity to row ordering only.
        for seed in (0, 1, 7, 99):
            permutation = np.random.default_rng(seed).permutation(len(x))
            repeated = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(x[permutation])
            restored = repeated[np.argsort(permutation)]
            stability.append({"permutation_seed": seed,
                              "adjusted_rand_index": float(adjusted_rand_score(labels, restored))})
    return x, labels, pd.DataFrame(diagnostics), pd.DataFrame(stability)


def analyze_training_data(raw, train):
    stats = pd.DataFrame({
        "aligned_length_hours": len(raw),
        "observed_length_hours": raw.notna().sum(),
        "observed_fraction": raw.notna().mean(),
        "training_mean_load": train.mean(),
        "training_std_load": train.std(),
        "training_cv": train.std() / train.mean(),
        "training_observed_hours": train.notna().sum(),
    })
    assert np.isfinite(stats.to_numpy()).all() and (stats.training_mean_load > 0).all()
    hour_of_week = train.index.dayofweek * 24 + train.index.hour
    weekly = train.groupby(hour_of_week).mean().T.div(train.mean(), axis=0)
    daily = train.groupby(train.index.hour).mean().T.div(train.mean(), axis=0)
    missing_weekly = int(weekly.isna().sum().sum())
    missing_daily = int(daily.isna().sum().sum())
    # Fill missing aggregated profile features only; model data remain untouched.
    for profiles in [weekly, daily]:
        for name in profiles.index[profiles.isna().any(axis=1)]:
            values = profiles.loc[name].to_numpy()
            observed = np.isfinite(values)
            assert observed.any()
            profiles.loc[name] = np.interp(np.arange(len(values)), np.flatnonzero(observed),
                                           values[observed], period=len(values))
    assert weekly.shape == (366, 168) and weekly.notna().all().all()
    x, labels, selection, stability = cluster_training_profiles(weekly, daily)
    best_k = 4
    with threadpool_limits(limits=2):
        pca = PCA(n_components=2, svd_solver="full").fit(x)
        pca_xy = pca.transform(x)
    stats["exploratory_cluster"] = labels + 1
    stats["pca_1"], stats["pca_2"] = pca_xy.T
    corr = train.corr(min_periods=5000)
    assert np.isfinite(np.diag(corr)).all()
    stats.to_csv(DATA / "transformer_descriptors.csv", index_label="Transformer")
    weekly.to_csv(DATA / "training_weekly_profiles.csv", index_label="Transformer")
    daily.to_csv(DATA / "training_daily_profiles.csv", index_label="Transformer")
    selection.to_csv(DATA / "cluster_selection.csv", index=False)
    stability.to_csv(DATA / "cluster_stability.csv", index=False)
    cluster_summary = []
    for c in range(best_k):
        mask = labels == c
        profile = daily.iloc[mask].median()
        cluster_summary.append({
            "cluster": c + 1,
            "transformers": int(mask.sum()), "median_profile_peak_hour": int(profile.idxmax()),
            "median_profile_trough_hour": int(profile.idxmin()),
            "median_training_mean_load": float(stats.training_mean_load.iloc[mask].median()),
            "median_training_cv": float(stats.training_cv.iloc[mask].median()),
        })
    pd.DataFrame(cluster_summary).to_csv(DATA / "cluster_summary.csv", index=False)
    summary = {
        "transformers": 366, "aligned_length_hours": 16272,
        "train_hours": 11390, "validation_hours": 1627, "test_hours": 3255,
        "statistics_period": "Training observations, except full-series observation counts",
        "clustering_features": "168 hourly-of-week mean loads divided by each transformer's training mean; dimensions standardized across transformers",
        "missing_aggregated_profile_cells": {"weekly": missing_weekly, "daily": missing_daily},
        "profile_feature_filling": "Circular linear interpolation within the aggregated training-only profile; model data and original observations are not changed",
        "clustering": "Agglomerative hierarchical clustering with Ward linkage and Euclidean distance; one shared feature scaling; four clusters",
        "cluster_count_rationale": "Four groups are a specified descriptive resolution, not the silhouette-maximizing count; same-scale Ward partitions for k=2..6 are exported as diagnostics",
        "selected_clusters": best_k,
        "cluster_counts": {str(c + 1): int((labels == c).sum()) for c in range(best_k)},
        "cluster_id_order": "Ascending peak hour of each cluster's median daily profile",
        "selected_silhouette": float(selection.loc[selection.used, "silhouette"].iloc[0]),
        "row_order_consistency": {"permutation_seeds": [0, 1, 7, 99],
                                  "minimum_adjusted_rand_index": float(stability.adjusted_rand_index.min()),
                                  "scope": "Row-order consistency only; not temporal or sampling stability"},
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "correlation": "Pairwise Pearson correlation from original observed training values; minimum 5000 joint observations",
        "mean_observed_fraction": float(stats.observed_fraction.mean()),
        "observed_hours_range": [int(stats.observed_length_hours.min()), int(stats.observed_length_hours.max())],
        "training_cv_median": float(stats.training_cv.median()),
        "training_mean_load_range": [float(stats.training_mean_load.min()), float(stats.training_mean_load.max())],
        "undefined_correlation_cells": int(corr.isna().sum().sum()),
        "pairwise_correlation_median": float(np.nanmedian(corr.to_numpy()[np.triu_indices(366, k=1)])),
        "interpretation": "Exploratory load-profile clusters, not geographic regions or validated discrete populations; no model is trained or regrouped using these clusters",
    }
    (DATA / "data_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return stats, daily, corr, labels, pca, best_k


def figure2(raw, train):
    stats, daily, corr, labels, pca, k = analyze_training_data(raw, train)
    fig = plt.figure(figsize=(14.4, 9.2), dpi=140)
    grid = fig.add_gridspec(2, 1, left=0.072, right=0.950, bottom=0.09, top=0.94,
                           hspace=0.64)
    top = grid[0, 0].subgridspec(1, 3, wspace=0.56)
    middle = grid[1, 0].subgridspec(1, 2, wspace=0.32)
    axes = [fig.add_subplot(top[0, c]) for c in range(3)]
    axes += [fig.add_subplot(middle[0, c]) for c in range(2)]
    ax = axes[0]
    ax.hist(stats.observed_length_hours, bins=18, color=BAR_BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(16272, color=INK, ls="--", lw=1.1)
    ax.set_xlim(min(stats.observed_length_hours.min() - 250, 15000), 16800)
    ax.set_xticks([14000, 15000, 16272], ["14,000", "15,000", "16,272"])
    ax.set_xlabel("Observed length (h)")
    ax.set_ylabel("Transformers")
    title(ax, "a. Valid observations")
    ax.text(0.05, 0.94, "n = 366", transform=ax.transAxes, fontsize=16, va="top")
    plot_style(ax)
    ax = axes[1]
    for c in range(k):
        subset = stats.iloc[labels == c]
        ax.scatter(subset.training_mean_load, subset.training_cv, s=28, color=CLUSTER_COLORS[c],
                   marker=CLUSTER_MARKERS[c], alpha=0.85, edgecolors="white", linewidths=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean load (log scale)")
    ax.set_ylabel("CV (log scale)")
    title(ax, "b. Load variability")
    plot_style(ax)
    ax = axes[2]
    order = np.lexsort((stats.training_mean_load.to_numpy(), labels))
    matrix = corr.iloc[order, order].to_numpy()
    cmap = LinearSegmentedColormap.from_list("signed_correlation", [NPG_COLORS[0], "#FAFAFA", NPG_COLORS[3]])
    cmap.set_bad("#B8BFC7")
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
    counts = [(labels == c).sum() for c in range(k)]
    boundaries = np.cumsum(counts)
    for boundary in boundaries[:-1]:
        ax.axvline(boundary - 0.5, color="white", lw=0.6)
        ax.axhline(boundary - 0.5, color="white", lw=0.6)
    # Numeric ordered indices avoid overlapping labels for the small subgroups.
    ax.set_xticks([0, 100, 200, 300], ["1", "101", "201", "301"])
    ax.set_yticks([0, 100, 200, 300], ["1", "101", "201", "301"])
    ax.set_xlabel("Transformers")
    ax.set_ylabel("Transformers")
    ax.tick_params(labelsize=14)
    title(ax, "c. Load correlations")
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035, ticks=[-1, 0, 1])
    cb.set_label("Pearson r", fontsize=14)
    cb.ax.tick_params(labelsize=13)
    ax = axes[3]
    for c in range(k):
        subset = stats.iloc[labels == c]
        ax.scatter(subset.pca_1, subset.pca_2, s=30, color=CLUSTER_COLORS[c],
                   marker=CLUSTER_MARKERS[c], edgecolors="white", linewidths=0.35, alpha=0.9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    title(ax, "d. PCA")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    plot_style(ax)
    ax = axes[4]
    for c in range(k):
        values = daily.iloc[labels == c]
        ax.plot(np.arange(24), values.median(), color=CLUSTER_COLORS[c], lw=1.8,
                marker=CLUSTER_MARKERS[c], markersize=4, markevery=3,
                label=f"C{c + 1} (n={len(values)})")
        ax.fill_between(np.arange(24), values.quantile(0.25), values.quantile(0.75),
                        color=CLUSTER_COLORS[c], alpha=0.10, linewidth=0)
    ax.set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Load / train mean")
    ax.legend(loc="upper left", fontsize=13.5, frameon=False, ncol=2,
              columnspacing=0.9, handlelength=1.4, borderaxespad=0.1)
    title(ax, "e. Daily profiles")
    plot_style(ax)
    save(fig, "Fig2_data_and_model")


def radar_data(steps, metrics):
    records = []
    for horizon in [24, 168, 720, 2160, 3255]:
        rows = steps.loc[steps.end_hour == horizon].set_index("Model").loc[MODELS]
        assert len(rows) == 9
        frame = pd.DataFrame(index=MODELS)
        for key in ["mse", "mae", "wape", "r2"]:
            frame[key] = rows[f"cumulative_{key}"]
        for key in ["mse", "mae", "wape"]:
            assert frame[key].min() > 0
            frame[f"{key}_score"] = frame[key].min() / frame[key]
        residual = 1 - frame.r2
        assert (residual > 0).all()
        frame["r2_score"] = residual.min() / residual
        for model, row in frame.iterrows():
            records.append({"Model": model, "horizon_hours": horizon, **row.to_dict(),
                            "mse_div_1000": row.mse / 1000,
                            "relative_mse": row.mse / MSE_REFERENCE})
    result = pd.DataFrame(records)
    scores = result[[f"{k}_score" for k in ["mse", "mae", "wape", "r2"]]]
    assert ((scores >= 0) & (scores <= 1 + 1e-8)).all().all()
    # Same observed targets make these paired axes algebraically redundant.
    assert np.allclose(result.mse_score, result.r2_score)
    assert np.allclose(result.mae_score, result.wape_score)
    result.to_csv(DATA / "radar_metrics_and_scores.csv", index=False)
    return result


def radar_axis(ax, rows, heading, big=False):
    theta = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    closed = np.r_[theta, theta[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(theta)
    ax.set_xticklabels(["MSE", "MAE", "WAPE", r"$R^2$"], fontsize=19 if big else 16)
    ax.tick_params(axis="x", pad=10 if big else 7)
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(["0.5", "1"], fontsize=13 if big else 11.5)
    ax.set_rlabel_position(43)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_edgecolor("#B7BDC3")
    ax.spines["polar"].set_linewidth(0.65)
    for model in [m for m in MODELS if m != "Chronos"] + ["Chronos"]:
        row = rows.loc[model]
        values = row[[f"{key}_score" for key in ["mse", "mae", "wape", "r2"]]].to_numpy(dtype=float)
        values = np.r_[values, values[0]]
        color, ls, marker = STYLES[model]
        ax.plot(closed, values, color=color, ls=ls, marker=marker,
                markersize=4.5 if big else 3.5, lw=1.6 if big else 1.3,
                markerfacecolor="white", markeredgewidth=1, alpha=1)
    ax.set_title(heading, fontsize=19 if big else 18, fontweight="bold", pad=20 if big else 10)


def model_legend():
    return [Line2D([0], [0], color=STYLES[m][0], ls=STYLES[m][1], marker=STYLES[m][2],
                   markersize=4.5, markerfacecolor="white", markeredgewidth=1, lw=1.5,
                   label=LABELS.get(m, m)) for m in MODELS]


def comparison_table_data(steps, metrics):
    """Combine the two manuscript tables without repeating the full-test errors."""
    table = metrics.loc[MODELS, ["relative_mse", "mae", "wape", "r2"]].copy()
    table.columns = ["full_relative_mse", "full_mae", "full_wape", "full_r2"]
    for horizon in (24, 168, 720):
        rows = steps.loc[steps.end_hour == horizon].set_index("Model").loc[MODELS]
        for key in ("mae", "wape"):
            table[f"h{horizon}_{key}"] = rows[f"cumulative_{key}"]
    table.to_csv(DATA / "Table1_overall_and_horizons.csv", index_label="Model")
    return table


def horizon_bar_axis(ax, rows, heading, show_models=True):
    values = rows.loc[MODELS, "mae"].to_numpy(dtype=float)
    bars = ax.barh(np.arange(len(MODELS)), values, height=0.66,
                   color=[STYLES[m][0] for m in MODELS], edgecolor="none")
    ax.set_yticks(np.arange(len(MODELS)), [LABELS.get(m, m) for m in MODELS]
                 if show_models else [])
    ax.invert_yaxis()
    # A shared zero-based scale makes magnitudes comparable across all four panels.
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 30, 60])
    ax.set_xlabel("Cumulative MAE", fontsize=18)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=16)
    style_axes(ax, tick_size=17)
    ax.tick_params(axis="y", length=0, pad=7)
    title(ax, heading, size=20)


def figure3(steps, metrics):
    data = radar_data(steps, metrics)
    comparison_table_data(steps, metrics)
    data.loc[data.horizon_hours != 3255, ["Model", "horizon_hours", "mae"]].to_csv(
        DATA / "horizon_cumulative_mae.csv", index=False)
    fig = plt.figure(figsize=(14.4, 10.8), dpi=140)
    ax = fig.add_axes([0.058, 0.525, 0.30, 0.36], projection="polar")
    radar_axis(ax, data.loc[data.horizon_hours == 3255].set_index("Model"),
               "a. Full test (3,255 h)", big=True)
    fig.text(0.208, 0.46, "Relative score (best = 1)", ha="center", fontsize=17)
    fig.legend(handles=model_legend(), loc="center", bbox_to_anchor=(0.208, 0.255),
               ncol=1, frameon=False, fontsize=18, handlelength=2.0, labelspacing=0.7)
    specifications = [
        ([0.535, 0.58, 0.185, 0.32], 24, "b. 24 h", True),
        ([0.785, 0.58, 0.185, 0.32], 168, "c. 168 h", False),
        ([0.535, 0.11, 0.185, 0.32], 720, "d. 720 h", True),
        ([0.785, 0.11, 0.185, 0.32], 2160, "e. 2,160 h", False),
    ]
    for rect, horizon, heading, show_models in specifications:
        ax = fig.add_axes(rect)
        horizon_bar_axis(ax, data.loc[data.horizon_hours == horizon].set_index("Model"),
                         heading, show_models)
    save(fig, "Fig3_performance_radars")


def figure4(steps, predictions):
    fig, axes = plt.subplots(3, 2, figsize=(14.4, 11.0), dpi=140,
                              gridspec_kw={"height_ratios": [1, 1, 1.1]})
    fig.subplots_adjust(left=0.075, right=0.965, top=0.855, bottom=0.105, hspace=0.61, wspace=0.24)
    handles = model_legend() + [Line2D([0], [0], color="#22272D", lw=1.8, label="Observed")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.986),
               ncol=5, frameon=False, fontsize=LEGEND_SIZE, columnspacing=1.1, handlelength=1.8)
    specifications = [("mse", f"a. Cumulative {mse_label(MSE_DISPLAY)}", mse_label(MSE_DISPLAY)),
                      ("mae", "b. Cumulative MAE", "MAE"),
                      ("wape", "c. Cumulative WAPE", "WAPE (%)"),
                      ("r2", r"d. Cumulative $R^2$", r"$R^2$")]
    for ax, (key, heading, ylabel) in zip(axes[:2].flat, specifications):
        for model in [m for m in MODELS if m != "Chronos"] + ["Chronos"]:
            series = steps.loc[steps.Model == model]
            color, ls, _ = STYLES[model]
            values = series[f"cumulative_{key}"]
            if key == "mse":
                values = display_mse(values, MSE_DISPLAY, MSE_REFERENCE)
                if MSE_DISPLAY == "relative":
                    assert np.allclose(values, series.cumulative_relative_mse)
            ax.plot(series.end_hour / 24, values, color=color, ls=ls,
                    lw=2 if model == "Chronos" else 1.15, alpha=1 if model == "Chronos" else 0.85)
        ax.axvline(7, ls=":", color="#A3ADB7", lw=0.9)
        ax.set(xlim=(0, 136), xticks=[0, 30, 60, 90, 135])
        ax.set_xlabel("Horizon (d)")
        ax.set_ylabel(ylabel)
        title(ax, heading)
        plot_style(ax)
    for ax, name, letter in zip(axes[2], ["0-0-0", "0-0-1"], ["e", "f"]):
        days = np.arange(336) / 24
        ax.axvspan(7, 14, color="#F0F4F8", zorder=0)
        for model in [m for m in MODELS if m != "Chronos"] + ["Chronos"]:
            color, ls, _ = STYLES[model]
            ax.plot(days, predictions[f"{model}_{name}_Pred"].iloc[:336], color=color, ls=ls,
                    lw=1.65 if model == "Chronos" else 0.95,
                    alpha=1 if model == "Chronos" else 0.8)
        ax.plot(days, predictions[f"{name}_True"].iloc[:336], color="#22272D", lw=1.6)
        ax.axvline(7, ls=":", color="#A3ADB7", lw=0.9)
        ax.set(xlim=(0, 14), xticks=[0, 2, 4, 7, 10, 12, 14])
        ax.set_xlabel("Horizon (d)")
        ax.set_ylabel("Load")
        title(ax, f"{letter}. Transformer {name}")
        plot_style(ax)
    fig.text(0.5, 0.022, "Dotted line: 7 d; prediction-only context thereafter", ha="center", fontsize=15)
    save(fig, "Fig4_recursive_curves_and_paths")


def write_captions():
    captions = {
        "Figure 2": "Data characteristics. (a) Valid observed sequence lengths. (b) Mean load and coefficient of variation. (c) Pearson correlations ordered by load-profile cluster; undefined cells are gray. (d) PCA of normalized weekly profiles. (e) Cluster median normalized daily profiles and interquartile ranges. Statistics use the training period except for sequence lengths in (a).",
        "Figure 3": "Overall performance and cumulative errors across forecast horizons. (a) Relative metric scores over the full 3,255 h test period; larger radii indicate better overall performance. (b-e) Cumulative MAE at 24, 168, 720, and 2,160 h, shown on a common zero-based scale; lower values indicate lower error. Colors and model order are consistent across panels. Chronos-2 base and Chronos-2 + LoRA denote Chronos-ZeroShot and Chronos, respectively. Scores in (a) are defined in Section 2.4.",
        "Figure 4": "Recursive error evolution and example load trajectories. (a-d) Cumulative MSE, MAE, WAPE, and R-squared throughout the 3,255 h recursive test for all nine models. MSE is divided by 1000 solely for display. (e,f) Measured loads and recursive forecasts for transformers 0-0-0 and 0-0-1 during the first 14 d. The dotted 168 h boundary marks complete replacement of the initial observed context by model-generated values; shaded portions of the example trajectories denote the same stage. Observed values after the origin are used only for evaluation, not input refresh. Examples follow the existing manuscript and are not selected to maximize Chronos's apparent advantage.",
        "Table 1": "Overall forecasting performance and cumulative errors across forecast horizons. Each paired cell reports cumulative MAE / WAPE (%). Full-test relative MSE uses the fixed reference specified in Section 2.4. Bold values indicate the best result in each column.",
        "Table 2": "Dynamic fidelity diagnostics of the main models.",
    }
    description = mse_description(MSE_DISPLAY, MSE_REFERENCE)
    captions["Figure 4"] = captions["Figure 4"].replace("MSE is divided by 1000 solely for display.", description)
    (OUT / "captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    text = "# Manuscript Figures 2-4\n\nFigure 1 is the author's separately drawn graphical abstract and is not generated by this code. This script generates Figures 2-4 only. No model is retrained and no numerical forecast is changed.\n\n"
    text += f"## MSE Display\n\n{description}\n\n"
    text += "| New item | Purpose | Placement | Replaces |\n|---|---|---|---|\n"
    text += "| Figure 1 | Author-supplied graphical abstract | End of Introduction | Overview of the study |\n"
    text += "| Figure 2 | Data statistics and training-only PCA and cluster profiles | Section 2.1 | New descriptive data figure |\n"
    text += "| Figure 3 | Overall radar and four cumulative-MAE bar charts | Section 3.1, cross-reference in 3.2 | Previous five-radar comparison |\n"
    text += "| Figure 4 | Full recursive curves and example trajectories | Sections 3.2 and 3.3 | Former Figures 2 and 3 combined |\n"
    text += "| Table 1 | Overall metrics and horizon-specific cumulative errors | Section 3.1, cross-reference in 3.2 | Previous Tables 1 and 2 merged without repeating full-test MAE/WAPE |\n"
    text += "| Table 2 | Dynamic fidelity diagnostics | Section 3.3 | Previous Table 3 renumbered |\n\n"
    text += "No Figure 5 is included. The previous charts remain in the source experiment directories as reproducibility evidence; they are not part of the new manuscript figure set.\n\n"
    text += "## Comparison Layout\n\nThe radar uses full-test relative scores only. The four bar panels use cumulative MAE from the existing exports at 24, 168, 720, and 2,160 h, with the same model order and 0-70 axis. This allows direct comparison of error magnitudes without repeated radar-score transformations. All nine models remain visible.\n\n"
    text += "The merged Table 1 retains cumulative MAE and WAPE at 24, 168, 720, and 3,255 h, plus full-test relative MSE and R-squared. Full-test MAE/WAPE are reported once. Table 2 retains the original dynamic-fidelity values. Section 3.4 reports LoRA adaptation only; historical grouping experiments are not included in this manuscript figure set.\n\n"
    text += "Overall radar scores are relative to the best model, not absolute accuracy or a composite score. The MSE/R-squared and MAE/WAPE axes remain algebraically related under common evaluation targets.\n\n"
    text += "## Captions\n\n" + "\n\n".join(f"**{key}.** {value}" for key, value in captions.items())
    text += "## Methods Addition for Figure 2\n\n"
    text += ("For descriptive analysis, each transformer's mean weekly profile was estimated from valid training observations "
             "and divided by its training-period mean load. The 42 missing weekly-profile cells and six missing daily-profile "
             "cells across three transformers were filled by circular linear interpolation within their training-only aggregated "
             "profiles, without changing observed data or the forecasting pipeline. The resulting 168-dimensional profiles were standardized across "
             "transformers. Ward hierarchical clustering with Euclidean distance divided the profiles into four groups "
             "to describe differences in load shape. The same standardized features were used for all transformers. "
             "Two-to-six-group silhouette diagnostics and row-order checks are exported with the cluster statistics. "
             "PCA was applied to the globally standardized training-only "
             "features. The embeddings and cluster profiles characterize load heterogeneity; they do not identify physical "
             "regions, demonstrate forecasting mechanisms, or alter model inputs or grouping.\n\n")
    text += f"## Regeneration\n\n```powershell\npython plot_paper_figures.py --mse-display {MSE_DISPLAY}\n```\n"
    (OUT / "FIGURE_PLAN_AND_CAPTIONS.md").write_text(text, encoding="utf-8")


def write_plot_manifest(figures):
    paths = [RESULTS / "metrics_summary.csv", RESULTS / "recursive_step_metrics.csv",
             RESULTS / "experiment_manifest.json"]
    if 2 in figures:
        paths.append(ROOT / "data/representative_data.csv")
    if 4 in figures:
        paths.append(RESULTS / "predictions.csv")
    author_image = ROOT / "results/paper/manuscript_figures/Fig1_graphical_abstract_author.png"
    if author_image.is_file():
        paths.append(author_image)
    sources = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest()})
    payload = {
        "generated_figures": figures, "font": FONT, "dpi": EXPORT_DPI,
        "graphical_abstract": "Author-supplied Figure 1; not generated or overwritten",
        "formats": list(EXPORT_FORMATS), "models": MODELS, "seed": SEED,
        "mse_display": MSE_DISPLAY, "mse_reference": MSE_REFERENCE,
        "mse_description": mse_description(MSE_DISPLAY, MSE_REFERENCE),
        "sources": sources,
        "figure_layout": "Concise panel titles, enlarged typography, shared legends; details in captions",
        "figure2_panels": ["a. Valid observations", "b. Load variability", "c. Load correlations",
                           "d. PCA", "e. Daily profiles"] if 2 in figures else [],
        "palette": {"name": "NPG (nrc)", "colors": NPG_COLORS,
                    "source": "https://nanx.me/ggsci/reference/pal_npg.html",
                    "model_colors": {m: STYLES[m][0] for m in MODELS},
                    "cluster_colors": CLUSTER_COLORS},
        "radar_style": "Unfilled profiles; common line weights; distinct colors, line styles and markers",
        "figure3_panels": ["a. Full-test radar (3255 h)", "b. Cumulative MAE (24 h)",
                           "c. Cumulative MAE (168 h)", "d. Cumulative MAE (720 h)",
                           "e. Cumulative MAE (2160 h)"] if 3 in figures else [],
        "figure3_bar_axis": {"metric": "cumulative_mae", "limits": [0, 70],
                             "model_order": MODELS, "all_nine_models": True},
        "font_sizes_pt": {"body": 16, "axis_ticks": 14, "panel_titles": 18, "legend": 15},
        "plotting_code": ["plot_paper_figures.py", "paper_figures/style.py",
                          "paper_figures/manuscript.py"],
        "metrics_source": "evaluate.py exports; aggregate and final cumulative values are checked",
        "forecast_source": "Existing predictions.csv; no forecasts generated or changed",
        "clustering_scope": "Training-only descriptive analysis; not model training or grouping",
        "clustering_method": "Ward hierarchical clustering with Euclidean distance; one shared scaling; four descriptive clusters" if 2 in figures else None,
        "radar_scope": "Relative visualization scores only; not new evaluation metrics or composite scores",
        "software": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": version("scikit-learn")},
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(figures=(2, 3, 4), output_dir=None, dpi=600, formats=("png", "pdf", "svg"), mse_display="relative"):
    global OUT, DATA, QA, EXPORT_DPI, EXPORT_FORMATS, MSE_DISPLAY, MSE_REFERENCE
    figures = sorted(set(figures))
    if not figures or set(figures) - {2, 3, 4}:
        raise ValueError("Select figures 2, 3, or 4; Figure 1 is supplied separately")
    OUT = Path(output_dir).resolve() if output_dir is not None else ROOT / "results/paper/manuscript_figures"
    DATA, QA = OUT / "source_data", OUT / "_qa"
    EXPORT_DPI, EXPORT_FORMATS, MSE_DISPLAY = dpi, tuple(formats), mse_display
    DATA.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    raw, train, steps, metrics, predictions = evidence(figures)
    MSE_REFERENCE = mse_reference(metrics)
    print(f"Using existing experiment exports; MSE display: {MSE_DISPLAY}", flush=True)
    if 2 in figures:
        figure2(raw, train)
        print("Figure 2 complete", flush=True)
    if 3 in figures:
        figure3(steps, metrics)
        print("Figure 3 complete", flush=True)
    if 4 in figures:
        figure4(steps, predictions)
        print("Figure 4 complete", flush=True)
    write_captions()
    write_plot_manifest(figures)
    print(f"Saved requested figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
