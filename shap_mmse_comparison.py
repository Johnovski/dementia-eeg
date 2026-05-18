"""
SHAP Severity vs MMSE — Per-Subject Comparison
================================================
Computes actual SHAP-based severity scores per subject
and creates a clear side-by-side comparison with MMSE.

The SHAP severity score is defined as:
    SHAP_severity = mean(|SHAP_toward_AD|) / (mean(|SHAP_toward_AD|) + mean(|SHAP_toward_CN|))

This gives a [0, 1] score where:
    - 0.0 = SHAP attributes all features toward CN (healthy)
    - 1.0 = SHAP attributes all features toward AD (most severe)

Usage:
    python shap_mmse_comparison.py
    python shap_mmse_comparison.py --fold 0
    python shap_mmse_comparison.py --windows-per-subject 10
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import h5py
import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
FEATURES_FILE = BASE_DIR / "features" / "features.h5"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
PARTICIPANTS_TSV = BASE_DIR / "ds004504" / "participants.tsv"

LABEL_NAMES = {0: "AD", 1: "FTD", 2: "CN"}
N_CLASSES = 3

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_features():
    with h5py.File(FEATURES_FILE, "r") as hf:
        X = hf["X"][:]
        y = hf["y"][:]
        subject_ids = hf["subject_ids"][:]
        if isinstance(subject_ids[0], bytes):
            subject_ids = np.array([s.decode("utf-8") for s in subject_ids])
    return X, y, subject_ids


def load_participants():
    participants = {}
    with open(PARTICIPANTS_TSV, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            row = dict(zip(header, fields))
            participants[row["participant_id"]] = {
                "gender": row["Gender"],
                "age": int(row["Age"]),
                "group": row["Group"],
                "mmse": int(row["MMSE"]),
            }
    return participants


def find_best_fold():
    best_fold, best_acc = 0, 0.0
    for cp_file in sorted(CHECKPOINT_DIR.glob("fold_*.json")):
        with open(cp_file) as f:
            result = json.load(f)
        if result["subject_accuracy"] > best_acc:
            best_acc = result["subject_accuracy"]
            best_fold = result["fold"]
    return best_fold, best_acc


def get_fold_data(fold_idx):
    from sklearn.model_selection import StratifiedKFold
    X, y, subject_ids = load_features()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        if i == fold_idx:
            return (X[train_idx], y[train_idx], subject_ids[train_idx],
                    X[val_idx], y[val_idx], subject_ids[val_idx])
    raise ValueError("Fold not found")


# ─── Per-Subject SHAP Computation ────────────────────────────────────────────

def compute_per_subject_shap(model, X_train, X_val, y_val, subject_ids_val,
                              windows_per_subject=10, n_background=100):
    """Compute actual SHAP attribution scores per subject.

    For each subject:
      1. Sample `windows_per_subject` windows
      2. Run SHAP GradientExplainer
      3. Compute SHAP severity = |SHAP_AD| / (|SHAP_AD| + |SHAP_CN|)

    Returns dict: {subject_id: {"shap_severity": float, "shap_ad": float,
                                  "shap_ftd": float, "shap_cn": float, ...}}
    """
    import shap

    # Create background set (balanced sample from training data)
    np.random.seed(42)
    bg_indices = np.random.choice(len(X_train), min(n_background, len(X_train)),
                                   replace=False)
    X_bg = X_train[bg_indices].astype(np.float32)

    explainer = shap.GradientExplainer(model, X_bg)

    unique_subjects = np.unique(subject_ids_val)
    subject_results = {}

    total = len(unique_subjects)
    for idx, sub_id in enumerate(sorted(unique_subjects)):
        sub_mask = subject_ids_val == sub_id
        sub_indices = np.where(sub_mask)[0]
        sub_label = y_val[sub_indices[0]]

        # Sample windows
        n_sample = min(windows_per_subject, len(sub_indices))
        sampled = np.random.choice(sub_indices, n_sample, replace=False)
        X_sub = X_val[sampled].astype(np.float32)

        # Compute SHAP
        raw_shap = explainer.shap_values(X_sub)
        raw_shap = np.array(raw_shap)

        # Parse shape — GradientExplainer may return different shapes
        if raw_shap.ndim == 5 and raw_shap.shape[0] == N_CLASSES:
            # (3, n_sub, 19, 19, 3) — class-first
            shap_ad = raw_shap[0]    # (n_sub, 19, 19, 3)
            shap_ftd = raw_shap[1]
            shap_cn = raw_shap[2]
        elif raw_shap.ndim == 5 and raw_shap.shape[-1] == N_CLASSES:
            # (n_sub, 19, 19, 3, 3) — class-last
            shap_ad = raw_shap[:, :, :, :, 0]
            shap_ftd = raw_shap[:, :, :, :, 1]
            shap_cn = raw_shap[:, :, :, :, 2]
        else:
            print("    ⚠ Unexpected SHAP shape for {}: {}".format(sub_id, raw_shap.shape))
            continue

        # Compute mean absolute SHAP per class (averaged over windows)
        mag_ad = float(np.mean(np.abs(shap_ad)))
        mag_ftd = float(np.mean(np.abs(shap_ftd)))
        mag_cn = float(np.mean(np.abs(shap_cn)))

        # SHAP severity: proportion of attribution toward AD vs CN
        # 1.0 = all attribution toward AD, 0.0 = all toward CN
        total_mag = mag_ad + mag_cn + 1e-10
        shap_severity = mag_ad / total_mag

        subject_results[sub_id] = {
            "shap_severity": shap_severity,
            "shap_ad_magnitude": mag_ad,
            "shap_ftd_magnitude": mag_ftd,
            "shap_cn_magnitude": mag_cn,
            "true_label": int(sub_label),
            "n_windows": n_sample,
        }

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print("    Processed {}/{} subjects".format(idx + 1, total))

    return subject_results


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_side_by_side(subject_results, participants, save_dir):
    """Create a side-by-side bar chart: MMSE vs SHAP severity per subject.

    Subjects are sorted by MMSE score so the comparison is visually clear.
    MMSE is normalized to [0, 1] where 1 = healthy (MMSE=30), 0 = severe.
    SHAP severity is already [0, 1] where 1 = AD, 0 = CN.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect data
    subjects = []
    for sub_id, res in subject_results.items():
        if sub_id in participants:
            subjects.append({
                "id": sub_id,
                "mmse": participants[sub_id]["mmse"],
                "shap": res["shap_severity"],
                "label": res["true_label"],
                "group": participants[sub_id]["group"],
            })

    # Sort by MMSE score (ascending = most severe first)
    subjects.sort(key=lambda s: s["mmse"])

    n = len(subjects)
    ids = [s["id"].replace("sub-", "S") for s in subjects]
    mmse_raw = np.array([s["mmse"] for s in subjects])
    shap_scores = np.array([s["shap"] for s in subjects])
    labels = [s["label"] for s in subjects]

    # Normalize MMSE to [0,1]: 0 = MMSE of 0 (severe), 1 = MMSE of 30 (healthy)
    # INVERT so that higher = more severe (to match SHAP direction)
    mmse_inverted = 1.0 - (mmse_raw / 30.0)

    # Color by true class
    color_map = {0: "#E53935", 1: "#FB8C00", 2: "#43A047"}
    label_map = {0: "AD", 1: "FTD", 2: "CN"}
    colors = [color_map[l] for l in labels]

    # ── Figure 1: Side-by-side bars per subject ──
    fig, ax = plt.subplots(figsize=(22, 8))

    x = np.arange(n)
    width = 0.38

    bars_mmse = ax.bar(x - width / 2, mmse_inverted, width,
                        label="Clinical Severity\n(1 − MMSE/30)",
                        color=[c + "88" for c in colors],  # transparent
                        edgecolor=colors, linewidth=1.2)
    bars_shap = ax.bar(x + width / 2, shap_scores, width,
                        label="SHAP Severity Score",
                        color=colors, alpha=0.85,
                        edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Subject (sorted by MMSE, most severe → least severe)",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Severity Score (0 = Healthy, 1 = Severe AD)",
                  fontsize=13, fontweight="bold")
    ax.set_title(
        "SHAP Severity Score vs Clinical MMSE — Per Subject Comparison",
        fontsize=16, fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, fontsize=6)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add class labels at bottom
    for i, s in enumerate(subjects):
        ax.text(i, -0.08, label_map[s["label"]], ha="center", fontsize=5,
                color=color_map[s["label"]], fontweight="bold",
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    path = save_dir / "shap_vs_mmse_per_subject.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Per-subject comparison saved: {}".format(path))

    # ── Figure 2: Direct scatter with Spearman ──
    from scipy.stats import spearmanr, pearsonr

    spearman_r, spearman_p = spearmanr(shap_scores, mmse_raw)
    pearson_r, pearson_p = pearsonr(shap_scores, mmse_raw)

    print("  SHAP Severity vs MMSE:")
    print("    Spearman ρ = {:.4f} (p = {:.2e})".format(spearman_r, spearman_p))
    print("    Pearson  r = {:.4f} (p = {:.2e})".format(pearson_r, pearson_p))

    fig2, ax2 = plt.subplots(figsize=(10, 7))

    markers = {0: "s", 1: "^", 2: "o"}
    for cls in [2, 1, 0]:
        mask = np.array(labels) == cls
        ax2.scatter(mmse_raw[mask], shap_scores[mask],
                    c=color_map[cls], label=label_map[cls],
                    marker=markers[cls], s=90, alpha=0.85,
                    edgecolors="white", linewidth=0.5, zorder=3 - cls)

    # Trend line
    z = np.polyfit(mmse_raw, shap_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(mmse_raw.min() - 1, mmse_raw.max() + 1, 100)
    ax2.plot(x_line, p(x_line), "--", color="gray", alpha=0.7, linewidth=2,
             label="Linear trend")

    ax2.set_xlabel("MMSE Score (Clinical)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("SHAP Severity Score\n(|SHAP→AD| / (|SHAP→AD| + |SHAP→CN|))",
                   fontsize=13, fontweight="bold")
    ax2.set_title(
        "Actual SHAP Attribution Severity vs Clinical MMSE\n"
        "Spearman ρ = {:.3f} (p = {:.2e})".format(spearman_r, spearman_p),
        fontsize=15, fontweight="bold", pad=15
    )
    ax2.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Shade regions
    ax2.axhspan(0.0, 0.4, alpha=0.04, color="green", zorder=0)
    ax2.axhspan(0.4, 0.6, alpha=0.04, color="orange", zorder=0)
    ax2.axhspan(0.6, 1.0, alpha=0.04, color="red", zorder=0)

    plt.tight_layout()
    path2 = save_dir / "shap_severity_vs_mmse_scatter.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ SHAP severity scatter saved: {}".format(path2))

    # ── Figure 3: Paired bar — show each subject's MMSE and SHAP score ──
    # This is the clearest "layman" view
    fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(22, 10),
                                           sharex=True, gridspec_kw={"hspace": 0.05})

    # Top panel: MMSE scores
    ax_top.bar(x, mmse_raw, color=colors, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax_top.set_ylabel("MMSE Score\n(Clinical)", fontsize=13, fontweight="bold")
    ax_top.set_ylim(0, 35)
    ax_top.axhline(y=24, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax_top.text(n - 1, 24.5, "MMSE ≤ 24 = cognitive impairment",
                fontsize=9, color="gray", ha="right")
    ax_top.set_title(
        "Per-Subject Comparison: Clinical MMSE vs SHAP Severity",
        fontsize=16, fontweight="bold", pad=15
    )
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    # Bottom panel: SHAP severity (inverted axis so high severity is at bottom)
    ax_bot.bar(x, shap_scores, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    ax_bot.set_ylabel("SHAP Severity\n(0=Healthy, 1=AD)", fontsize=13, fontweight="bold")
    ax_bot.set_ylim(0, 1.1)
    ax_bot.set_xlabel("Subject (sorted by MMSE, most severe → least severe)",
                      fontsize=13, fontweight="bold")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(ids, rotation=90, fontsize=6)
    ax_bot.grid(axis="y", alpha=0.3)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    # Add class labels
    for i, s in enumerate(subjects):
        ax_bot.text(i, -0.12, label_map[s["label"]], ha="center", fontsize=5,
                    color=color_map[s["label"]], fontweight="bold",
                    transform=ax_bot.get_xaxis_transform())

    # Add legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#E53935", alpha=0.85, label="AD"),
        mpatches.Patch(color="#FB8C00", alpha=0.85, label="FTD"),
        mpatches.Patch(color="#43A047", alpha=0.85, label="CN"),
    ]
    ax_top.legend(handles=legend_handles, fontsize=11, loc="upper left")

    plt.tight_layout()
    path3 = save_dir / "shap_mmse_paired_comparison.png"
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ Paired comparison saved: {}".format(path3))

    # Save results
    results = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_subjects": n,
        "method": "SHAP GradientExplainer — actual attribution severity",
        "per_subject": {
            s["id"]: {
                "mmse": s["mmse"],
                "shap_severity": float(shap_scores[i]),
                "true_class": label_map[s["label"]],
            }
            for i, s in enumerate(subjects)
        }
    }
    results_path = RESULTS_DIR / "shap_mmse_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("  ✓ Results saved: {}".format(results_path))


# ─── Main ────────────────────────────────────────────────────────────────────

def main(fold_idx=None, windows_per_subject=10):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  SHAP Severity vs MMSE — Per-Subject Analysis")
    print("=" * 60)

    # Find best fold
    if fold_idx is None:
        fold_idx, best_acc = find_best_fold()
        print("  Best fold: {} (acc: {:.1%})".format(fold_idx, best_acc))

    # Load model
    model_path = MODELS_DIR / "fold_{:03d}".format(fold_idx) / "model.keras"
    if not model_path.exists():
        print("  ✗ Model not found: {}".format(model_path))
        return
    print("  Loading model...")
    model = tf.keras.models.load_model(str(model_path))

    # Load data
    print("  Loading fold {} data...".format(fold_idx))
    X_train, y_train, sids_train, X_val, y_val, sids_val = get_fold_data(fold_idx)

    # Standardize (using training stats)
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 1, 2), keepdims=True), 1e-8)
    X_train_n = ((X_train - mean) / std).astype(np.float32)
    X_val_n = ((X_val - mean) / std).astype(np.float32)

    # Compute per-subject SHAP
    print()
    print("  Computing SHAP values per subject ({} windows each)...".format(
        windows_per_subject))
    print("  Unique subjects in validation: {}".format(
        len(np.unique(sids_val))))

    subject_results = compute_per_subject_shap(
        model, X_train_n, X_val_n, y_val, sids_val,
        windows_per_subject=windows_per_subject,
        n_background=100,
    )

    # Load MMSE and plot
    print()
    print("  Generating comparison plots...")
    participants = load_participants()
    plot_side_by_side(subject_results, participants, FIGURES_DIR)

    print()
    print("  ✓ Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP Severity vs MMSE Per-Subject Comparison"
    )
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--windows-per-subject", type=int, default=10,
                        help="Windows to sample per subject for SHAP (default: 10)")
    args = parser.parse_args()

    main(fold_idx=args.fold, windows_per_subject=args.windows_per_subject)
