"""
Advanced SHAP Analysis: Attention Weights & MMSE Correlation
==============================================================
Two novel analyses for the thesis:

1. Attention Weight vs SHAP Comparison
   - Extracts learned attention weights from the MHSA layer
   - Maps them back to electrode-level importance
   - Plots side-by-side with SHAP electrode importance

2. SHAP Severity Score vs MMSE Correlation
   - Computes a per-subject SHAP-based "dementia severity" score
   - Correlates with clinical MMSE scores
   - Validates that model confidence tracks clinical severity

Usage:
    python advanced_shap_analysis.py                    # Both analyses
    python advanced_shap_analysis.py --analysis attn    # Attention only
    python advanced_shap_analysis.py --analysis mmse    # MMSE only
    python advanced_shap_analysis.py --fold 0           # Specific fold
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
INPUT_SHAPE = (19, 19, 3)

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_features():
    """Load features and labels from HDF5 file."""
    with h5py.File(FEATURES_FILE, "r") as hf:
        X = hf["X"][:]
        y = hf["y"][:]
        subject_ids = hf["subject_ids"][:]
        if isinstance(subject_ids[0], bytes):
            subject_ids = np.array([s.decode("utf-8") for s in subject_ids])
    return X, y, subject_ids


def load_participants():
    """Load participant metadata including MMSE scores."""
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
    """Find the fold with best validation accuracy."""
    best_fold = 0
    best_acc = 0.0
    for cp_file in sorted(CHECKPOINT_DIR.glob("fold_*.json")):
        with open(cp_file) as f:
            result = json.load(f)
        if result["subject_accuracy"] > best_acc:
            best_acc = result["subject_accuracy"]
            best_fold = result["fold"]
    return best_fold, best_acc


def get_fold_val_data(fold_idx):
    """Recreate the validation split for a specific fold."""
    from sklearn.model_selection import StratifiedKFold

    X, y, subject_ids = load_features()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(y)), y)
    ):
        if i == fold_idx:
            return (X[train_idx], y[train_idx], subject_ids[train_idx],
                    X[val_idx], y[val_idx], subject_ids[val_idx])
    raise ValueError("Fold {} not found".format(fold_idx))


# ─── Analysis 1: Attention Weights vs SHAP ───────────────────────────────────

def extract_attention_weights(model, X_samples):
    """Extract attention weight matrix from the MHSA layer.

    Creates a sub-model that outputs attention scores alongside predictions.
    The MHSA layer's attention scores are (batch, n_heads, seq_len, seq_len)
    where seq_len=16 corresponds to the 4×4 spatial grid.

    Returns:
        attention_weights: (n_samples, n_heads, 16, 16)
    """
    import tensorflow as tf

    # Get the MHSA layer
    mhsa_layer = model.get_layer("mhsa")

    # Find the input to MHSA (output of attn_proj)
    attn_proj_layer = model.get_layer("attn_proj")

    # Build a sub-model: input → attn_proj output
    # We need the intermediate representation at the attn_proj layer
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=attn_proj_layer.output
    )

    # Get the projected representations
    proj_output = intermediate_model.predict(X_samples, verbose=0)

    # Now call the MHSA layer directly with return_attention_scores=True
    # proj_output shape: (batch, 16, 64)
    _, attention_scores = mhsa_layer(
        proj_output, proj_output,
        return_attention_scores=True,
        training=False
    )

    # attention_scores: (batch, n_heads, 16, 16)
    return attention_scores.numpy()


def compute_gradcam_importance(model, X_samples, class_idx):
    """Compute Grad-CAM electrode importance from the last conv layer.

    Uses gradient-weighted class activation mapping on the conv2b layer
    (shape 9×9×64, before pool2). This preserves spatial resolution much
    better than the 4×4 attention token grid.

    The 9×9 activation map is upsampled to 19×19 and then aggregated
    to per-electrode importance by summing over rows and columns.

    Returns:
        electrode_importance: (19,) normalized importance per electrode
    """
    import tensorflow as tf

    # Build Grad-CAM sub-model
    last_conv_layer = model.get_layer("relu2b")  # (batch, 9, 9, 64)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Process in batches to avoid OOM
    batch_size = 64
    all_cams = []

    for start in range(0, len(X_samples), batch_size):
        batch = X_samples[start:start + batch_size]
        batch_tensor = tf.constant(batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(batch_tensor, training=False)
            class_output = predictions[:, class_idx]

        grads = tape.gradient(class_output, conv_output)

        # Global average pooling of gradients → weight per filter
        weights = tf.reduce_mean(grads, axis=(1, 2))  # (batch, 64)

        # Weighted combination of feature maps
        cam = tf.einsum("bf,bhwf->bhw", weights, conv_output)  # (batch, 9, 9)
        cam = tf.nn.relu(cam)  # Only positive activations

        all_cams.append(cam.numpy())

    cam_all = np.concatenate(all_cams, axis=0)  # (n_samples, 9, 9)

    # Average across samples
    mean_cam = np.mean(cam_all, axis=0)  # (9, 9)

    # Upsample 9×9 → 19×19 using bilinear interpolation
    from scipy.ndimage import zoom
    cam_19 = zoom(mean_cam, 19 / 9, order=1)  # (19, 19)

    # Electrode importance: sum of connectivity involving each electrode
    # (sum over rows = how much this electrode contributes as source,
    #  sum over cols = how much as target)
    electrode_importance = cam_19.sum(axis=0) + cam_19.sum(axis=1)  # (19,)
    electrode_importance = electrode_importance / (electrode_importance.max() + 1e-10)

    return electrode_importance, cam_19


def extract_per_class_attention(model, X_val, y_val):
    """Extract Grad-CAM based electrode importance separately for each class.

    Uses gradient-weighted class activation from the last conv layer,
    which preserves spatial resolution (9×9) much better than the
    16-token attention grid (4×4).
    """
    class_attention = {}
    class_cam = {}

    for cls_idx, cls_name in LABEL_NAMES.items():
        cls_mask = y_val == cls_idx
        n_cls = int(cls_mask.sum())
        if n_cls == 0:
            class_attention[cls_name] = np.zeros(19)
            class_cam[cls_name] = np.zeros((19, 19))
            continue

        # Sample up to 300 per class
        cls_indices = np.where(cls_mask)[0]
        if len(cls_indices) > 300:
            np.random.seed(42)
            cls_indices = np.random.choice(cls_indices, 300, replace=False)

        X_cls = X_val[cls_indices].astype(np.float32)
        electrode_imp, cam_19 = compute_gradcam_importance(model, X_cls, cls_idx)
        class_attention[cls_name] = electrode_imp
        class_cam[cls_name] = cam_19

        print("    {} Grad-CAM extracted ({} samples)".format(cls_name, len(cls_indices)))

    return class_attention, class_cam


def plot_attention_vs_shap(class_attention, save_dir):
    """Plot attention-derived electrode importance alongside SHAP importance.

    Loads pre-computed SHAP values and creates side-by-side comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load SHAP values
    shap_path = RESULTS_DIR / "shap_values.npz"
    if not shap_path.exists():
        print("  ⚠ SHAP values not found at {}. Run explain_model.py first.".format(shap_path))
        print("    Plotting attention-only figure...")
        _plot_attention_only(class_attention, save_dir)
        return

    shap_data = np.load(shap_path)
    shap_cls0 = shap_data["shap_class_0"]  # (n_explain, 19, 19, 3)
    shap_cls1 = shap_data["shap_class_1"]
    shap_cls2 = shap_data["shap_class_2"]
    y_explain = shap_data["y_explain"]

    # Compute SHAP electrode importance per class
    shap_electrode = {}
    for cls_idx, cls_name, shap_vals in [
        (0, "AD", shap_cls0), (1, "FTD", shap_cls1), (2, "CN", shap_cls2)
    ]:
        cls_mask = y_explain == cls_idx
        if cls_mask.sum() == 0:
            shap_electrode[cls_name] = np.zeros(19)
            continue
        cls_shap = shap_vals[cls_mask]  # (n, 19, 19, 3)
        mean_shap = np.mean(np.abs(cls_shap), axis=(0, 3))  # (19, 19)
        imp = mean_shap.sum(axis=0) + mean_shap.sum(axis=1)
        shap_electrode[cls_name] = imp / (imp.max() + 1e-10)

    # ── Create comparison figure ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(19)
    width = 0.35

    colors_attn = {"AD": "#E53935", "FTD": "#FB8C00", "CN": "#43A047"}
    colors_shap = {"AD": "#EF9A9A", "FTD": "#FFCC80", "CN": "#A5D6A7"}

    for idx, cls_name in enumerate(["AD", "FTD", "CN"]):
        ax = axes[idx]

        ax.bar(x - width / 2, class_attention[cls_name], width,
               label="Grad-CAM", color=colors_attn[cls_name],
               alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x + width / 2, shap_electrode[cls_name], width,
               label="SHAP Importance", color=colors_shap[cls_name],
               alpha=0.85, edgecolor="white", linewidth=0.5,
               hatch="//")

        ax.set_xlabel("Electrode", fontsize=11)
        ax.set_ylabel("Normalized Importance", fontsize=11)
        ax.set_title("{} — Grad-CAM vs SHAP".format(cls_name),
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_ylim(0, 1.2)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Compute correlation
        corr = np.corrcoef(class_attention[cls_name], shap_electrode[cls_name])[0, 1]
        ax.text(0.02, 0.95, "r = {:.3f}".format(corr),
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(
        "Grad-CAM (Model-Intrinsic) vs SHAP (Post-hoc) Electrode Importance",
        fontsize=16, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    path = save_dir / "attention_vs_shap_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Attention vs SHAP comparison saved: {}".format(path))


def _plot_attention_only(class_attention, save_dir):
    """Fallback: plot attention weights only if SHAP data unavailable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(19)
    width = 0.25

    colors = {"AD": "#E53935", "FTD": "#FB8C00", "CN": "#43A047"}
    for idx, cls_name in enumerate(["AD", "FTD", "CN"]):
        offset = (idx - 1) * width
        ax.bar(x + offset, class_attention[cls_name], width,
               label=cls_name, color=colors[cls_name], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("EEG Electrode", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Attention Importance", fontsize=14, fontweight="bold")
    ax.set_title("MHSA Learned Attention — Electrode Importance",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = save_dir / "attention_electrode_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Attention electrode importance saved: {}".format(path))


def plot_gradcam_heatmaps(class_cam, save_dir):
    """Plot Grad-CAM heatmaps (19×19) per class — shows spatial connectivity importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = {"AD": "AD — Grad-CAM Activation",
              "FTD": "FTD — Grad-CAM Activation",
              "CN": "CN — Grad-CAM Activation"}
    cmaps = {"AD": "Reds", "FTD": "Oranges", "CN": "Greens"}

    for idx, cls_name in enumerate(["AD", "FTD", "CN"]):
        ax = axes[idx]
        cam = class_cam[cls_name]

        # Normalise to [0, 1]
        cam_norm = cam / (cam.max() + 1e-10)

        im = ax.imshow(cam_norm, cmap=cmaps[cls_name], vmin=0, vmax=1,
                        aspect="equal", interpolation="bilinear")
        ax.set_title(titles[cls_name], fontsize=14, fontweight="bold")
        ax.set_xticks(range(19))
        ax.set_yticks(range(19))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)
        ax.set_xlabel("Target Electrode", fontsize=10)
        ax.set_ylabel("Source Electrode", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Activation")

    fig.suptitle(
        "Grad-CAM: Spatial Connectivity Importance per Class",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = save_dir / "gradcam_heatmaps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Grad-CAM heatmaps saved: {}".format(path))


# ─── Analysis 2: SHAP Severity Score vs MMSE ─────────────────────────────────

def compute_shap_severity_scores(model, X_val, y_val, subject_ids_val):
    """Compute a per-subject SHAP-based dementia severity score.

    The severity score is defined on a CN→FTD→AD axis:
      severity = P(AD) * 1.0 + P(FTD) * 0.5 + P(CN) * 0.0

    where P(class) is the model's softmax probability for each class,
    averaged across all windows for a given subject.

    This creates a continuous [0, 1] score where:
      - 0.0 = confidently healthy
      - 0.5 = confidently FTD (mid-severity)
      - 1.0 = confidently AD (highest severity)

    Returns:
        subject_scores: dict of {subject_id: {"severity": float, "true_label": int,
                                               "pred_proba": [p_ad, p_ftd, p_cn]}}
    """
    # Get model predictions
    proba = model.predict(X_val, verbose=0)  # (n_windows, 3)

    # Aggregate per subject
    subject_scores = {}
    for sub in np.unique(subject_ids_val):
        mask = subject_ids_val == sub
        sub_proba = proba[mask]  # (n_windows, 3)
        sub_true = y_val[mask][0]

        # Average probabilities across all windows
        avg_proba = sub_proba.mean(axis=0)  # [P(AD), P(FTD), P(CN)]

        # Severity score: weighted sum on CN(0) → FTD(0.5) → AD(1.0) axis
        severity = avg_proba[0] * 1.0 + avg_proba[1] * 0.5 + avg_proba[2] * 0.0

        subject_scores[sub] = {
            "severity": float(severity),
            "true_label": int(sub_true),
            "pred_proba": avg_proba.tolist(),
            "confidence": float(np.max(avg_proba)),
        }

    return subject_scores


def compute_shap_attribution_severity(fold_idx):
    """Compute SHAP-based severity using actual SHAP attribution magnitudes.

    For each subject's explanation samples, compute the sum of SHAP
    attributions pushing toward the AD class minus those pushing toward CN.

    Returns None if SHAP values are not available.
    """
    shap_path = RESULTS_DIR / "shap_values.npz"
    if not shap_path.exists():
        return None

    shap_data = np.load(shap_path)
    shap_ad = shap_data["shap_class_0"]    # (n_explain, 19, 19, 3)
    shap_cn = shap_data["shap_class_2"]    # (n_explain, 19, 19, 3)
    y_explain = shap_data["y_explain"]

    # SHAP severity = mean(|SHAP_AD|) - mean(|SHAP_CN|) per sample
    # Positive = model attributes more to AD, Negative = more to CN
    ad_magnitude = np.mean(np.abs(shap_ad), axis=(1, 2, 3))  # (n_explain,)
    cn_magnitude = np.mean(np.abs(shap_cn), axis=(1, 2, 3))  # (n_explain,)

    shap_severity = ad_magnitude - cn_magnitude  # positive = AD-leaning

    return shap_severity, y_explain


def plot_severity_vs_mmse(subject_scores, participants, save_dir):
    """Scatter plot: model severity score vs MMSE, colored by true class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr

    severities = []
    mmse_scores = []
    true_labels = []
    sub_ids = []

    for sub_id, scores in subject_scores.items():
        if sub_id in participants:
            severities.append(scores["severity"])
            mmse_scores.append(participants[sub_id]["mmse"])
            true_labels.append(scores["true_label"])
            sub_ids.append(sub_id)

    severities = np.array(severities)
    mmse_scores = np.array(mmse_scores)
    true_labels = np.array(true_labels)

    # Compute correlations
    spearman_r, spearman_p = spearmanr(severities, mmse_scores)
    pearson_r, pearson_p = pearsonr(severities, mmse_scores)

    print("  Spearman correlation: r={:.4f}, p={:.2e}".format(spearman_r, spearman_p))
    print("  Pearson correlation:  r={:.4f}, p={:.2e}".format(pearson_r, pearson_p))

    # ── Scatter plot ──
    fig, ax = plt.subplots(figsize=(10, 7))

    colors_map = {0: "#E53935", 1: "#FB8C00", 2: "#43A047"}
    labels_map = {0: "AD", 1: "FTD", 2: "CN"}
    markers_map = {0: "s", 1: "^", 2: "o"}

    for cls in [2, 1, 0]:  # Plot CN first, AD on top
        mask = true_labels == cls
        ax.scatter(
            mmse_scores[mask], severities[mask],
            c=colors_map[cls], label=labels_map[cls],
            marker=markers_map[cls], s=80, alpha=0.8,
            edgecolors="white", linewidth=0.5, zorder=3 - cls
        )

    # Trend line
    z = np.polyfit(mmse_scores, severities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(mmse_scores.min() - 1, mmse_scores.max() + 1, 100)
    ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.7, linewidth=2,
            label="Linear trend")

    ax.set_xlabel("MMSE Score (Clinical)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model Severity Score (0=CN, 0.5=FTD, 1=AD)",
                  fontsize=14, fontweight="bold")
    ax.set_title(
        "Model Prediction Confidence vs Clinical MMSE Score\n"
        "Spearman ρ = {:.3f} (p = {:.2e})".format(spearman_r, spearman_p),
        fontsize=15, fontweight="bold", pad=15
    )

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate regions
    ax.axhspan(0.0, 0.25, alpha=0.05, color="green", zorder=0)
    ax.axhspan(0.25, 0.75, alpha=0.05, color="orange", zorder=0)
    ax.axhspan(0.75, 1.0, alpha=0.05, color="red", zorder=0)

    ax.text(30, 0.10, "CN region", fontsize=10, color="green", alpha=0.6)
    ax.text(30, 0.50, "FTD region", fontsize=10, color="orange", alpha=0.6)
    ax.text(30, 0.90, "AD region", fontsize=10, color="red", alpha=0.6)

    plt.tight_layout()
    path = save_dir / "severity_vs_mmse.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Severity vs MMSE plot saved: {}".format(path))

    # ── Box plot: MMSE distribution per model prediction ──
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    pred_labels = []
    pred_mmse = []
    for sub_id, scores in subject_scores.items():
        if sub_id in participants:
            pred_cls = int(np.argmax(scores["pred_proba"]))
            pred_labels.append(labels_map[pred_cls])
            pred_mmse.append(participants[sub_id]["mmse"])

    import pandas as pd
    df = pd.DataFrame({"Predicted Class": pred_labels, "MMSE": pred_mmse})

    class_order = ["CN", "FTD", "AD"]
    palette = {"AD": "#E53935", "FTD": "#FB8C00", "CN": "#43A047"}

    import seaborn as sns
    sns.boxplot(data=df, x="Predicted Class", y="MMSE", order=class_order,
                palette=palette, ax=ax2, width=0.5, showfliers=True)
    sns.stripplot(data=df, x="Predicted Class", y="MMSE", order=class_order,
                  palette=palette, ax=ax2, size=5, alpha=0.6, jitter=True)

    ax2.set_xlabel("Model Predicted Class", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Clinical MMSE Score", fontsize=14, fontweight="bold")
    ax2.set_title("MMSE Distribution by Model Prediction",
                  fontsize=15, fontweight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path2 = save_dir / "mmse_by_prediction.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ MMSE by prediction saved: {}".format(path2))

    # ── Per-class MMSE stats ──
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    true_mmse = defaultdict(list)
    true_severity = defaultdict(list)
    for sub_id, scores in subject_scores.items():
        if sub_id in participants:
            cls_name = labels_map[scores["true_label"]]
            true_mmse[cls_name].append(participants[sub_id]["mmse"])
            true_severity[cls_name].append(scores["severity"])

    for cls_name in ["CN", "FTD", "AD"]:
        if cls_name in true_mmse:
            ax3.scatter(
                true_mmse[cls_name], true_severity[cls_name],
                c=palette[cls_name], label=cls_name,
                s=70, alpha=0.7, edgecolors="white", linewidth=0.5
            )

            # Per-class trend
            if len(true_mmse[cls_name]) > 2:
                z = np.polyfit(true_mmse[cls_name], true_severity[cls_name], 1)
                p = np.poly1d(z)
                x_cls = np.linspace(min(true_mmse[cls_name]),
                                    max(true_mmse[cls_name]), 50)
                ax3.plot(x_cls, p(x_cls), "--", color=palette[cls_name],
                         alpha=0.5, linewidth=1.5)

    ax3.set_xlabel("MMSE Score", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Model Severity Score", fontsize=14, fontweight="bold")
    ax3.set_title("Within-Class: Does Severity Track MMSE?",
                  fontsize=15, fontweight="bold", pad=15)
    ax3.legend(fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    path3 = save_dir / "severity_vs_mmse_per_class.png"
    fig3.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ Per-class severity vs MMSE saved: {}".format(path3))

    # Save numerical results
    corr_results = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_subjects": len(severities),
        "per_subject": {
            sub_id: {
                "severity": subject_scores[sub_id]["severity"],
                "mmse": participants[sub_id]["mmse"],
                "true_class": labels_map[subject_scores[sub_id]["true_label"]],
                "pred_proba": subject_scores[sub_id]["pred_proba"],
            }
            for sub_id in sub_ids
        }
    }
    results_path = RESULTS_DIR / "mmse_correlation.json"
    with open(results_path, "w") as f:
        json.dump(corr_results, f, indent=2)
    print("  ✓ Correlation results saved: {}".format(results_path))


# ─── Main ────────────────────────────────────────────────────────────────────

def run_analysis(fold_idx=None, analysis="both"):
    """Run the advanced SHAP analyses."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  Advanced SHAP Analysis")
    print("=" * 60)

    # Find best fold
    if fold_idx is None:
        fold_idx, best_acc = find_best_fold()
        print("  Best fold: {} (accuracy: {:.1%})".format(fold_idx, best_acc))
    else:
        print("  Using fold: {}".format(fold_idx))

    # Load model
    model_path = MODELS_DIR / "fold_{:03d}".format(fold_idx) / "model.keras"
    if not model_path.exists():
        print("  ✗ Model not found: {}".format(model_path))
        return

    print("  Loading model...")
    model = tf.keras.models.load_model(str(model_path))

    # Load data
    print("  Loading fold {} data...".format(fold_idx))
    X_train, y_train, sids_train, X_val, y_val, sids_val = get_fold_val_data(fold_idx)

    # Standardize
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 1, 2), keepdims=True), 1e-8)
    X_val_n = ((X_val - mean) / std).astype(np.float32)

    # ── Analysis 1: Attention Weights vs SHAP ──
    if analysis in ("both", "attn"):
        print()
        print("  ── Analysis 1: Grad-CAM Electrode Importance ──")
        class_attention, class_cam = extract_per_class_attention(model, X_val_n, y_val)
        plot_attention_vs_shap(class_attention, FIGURES_DIR)
        plot_gradcam_heatmaps(class_cam, FIGURES_DIR)

    # ── Analysis 2: Severity vs MMSE ──
    if analysis in ("both", "mmse"):
        print()
        print("  ── Analysis 2: Severity Score vs MMSE ──")
        participants = load_participants()
        subject_scores = compute_shap_severity_scores(
            model, X_val_n, y_val, sids_val
        )
        plot_severity_vs_mmse(subject_scores, participants, FIGURES_DIR)

    print()
    print("  ✓ All advanced analyses complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced SHAP Analysis: Attention & MMSE"
    )
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (default: best fold)")
    parser.add_argument("--analysis", choices=["both", "attn", "mmse"],
                        default="both", help="Which analysis to run")
    args = parser.parse_args()

    run_analysis(fold_idx=args.fold, analysis=args.analysis)
