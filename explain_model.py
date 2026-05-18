"""
SHAP Explainability for Hybrid DS-CNN + MHSA
==============================================
Generates publication-quality SHAP explanations for the trained
dementia classification model.

Produces:
    - shap_summary.png           – Overall feature importance heatmap
    - shap_electrode_importance.png – Per-electrode importance bar chart
    - shap_band_importance.png   – Alpha vs Beta vs Gamma contribution
    - shap_class_comparison.png  – Per-class SHAP patterns

Usage:
    python explain_model.py                          # Use best fold
    python explain_model.py --fold 0                 # Specific fold
    python explain_model.py --n-background 100       # Background samples
"""

import argparse
import json
import os
import warnings
from pathlib import Path

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

LABEL_NAMES = {0: "AD", 1: "FTD", 2: "CN"}
N_CLASSES = 3
INPUT_SHAPE = (19, 19, 3)

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]
BAND_NAMES = ["Alpha", "Beta", "Gamma"]


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


# ─── SHAP Analysis ──────────────────────────────────────────────────────────

def compute_shap_values(model, X_background, X_explain):
    """Compute SHAP values using GradientExplainer.
    
    Returns:
        shap_values: numpy array of shape (n_classes, n_explain, 19, 19, 3)
    """
    import shap
    
    print("  Computing SHAP values ({} background, {} explain)...".format(
        len(X_background), len(X_explain)
    ))
    
    explainer = shap.GradientExplainer(model, X_background)
    raw_shap = explainer.shap_values(X_explain)
    
    # Normalize output to consistent shape: (n_classes, n_explain, 19, 19, 3)
    raw_shap = np.array(raw_shap)
    print("    Raw SHAP shape: {}".format(raw_shap.shape))
    
    if raw_shap.ndim == 5 and raw_shap.shape[0] == N_CLASSES:
        # Shape: (3, n_explain, 19, 19, 3) — already correct
        shap_values = raw_shap
    elif raw_shap.ndim == 4 and raw_shap.shape[0] == len(X_explain):
        # Shape: (n_explain, 19, 19, 3) — single output, duplicate for all classes
        shap_values = np.stack([raw_shap] * N_CLASSES, axis=0)
    elif raw_shap.ndim == 5 and raw_shap.shape[0] == len(X_explain):
        # Shape: (n_explain, 19, 19, 3, 3) — last dim is classes
        shap_values = np.moveaxis(raw_shap[:, :, :, :, :N_CLASSES], -1, 0)
    else:
        # Try to handle any other shape by reshaping
        print("    ⚠ Unexpected SHAP shape: {}".format(raw_shap.shape))
        shap_values = raw_shap
    
    print("    ✓ SHAP values computed: {}".format(shap_values.shape))
    return shap_values


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_shap_summary(shap_values, X_explain, y_explain, save_dir):
    """Plot SHAP summary: mean absolute SHAP as heatmaps per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for cls_idx, cls_name in LABEL_NAMES.items():
        ax = axes[cls_idx]
        
        cls_mask = y_explain == cls_idx
        n_cls = int(cls_mask.sum())
        if n_cls == 0:
            ax.set_title("{} (no samples)".format(cls_name))
            continue
        
        # shap_values[cls_idx] is (n_explain, 19, 19, 3)
        cls_shap = shap_values[cls_idx][cls_mask]  # (n_cls, 19, 19, 3)
        mean_shap = np.mean(np.abs(cls_shap), axis=(0, 3))  # → (19, 19)
        
        im = ax.imshow(mean_shap, cmap="Reds", aspect="equal")
        ax.set_title("{} — Mean |SHAP|".format(cls_name),
                     fontsize=14, fontweight="bold")
        ax.set_xticks(range(19))
        ax.set_yticks(range(19))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=90, fontsize=7)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle("SHAP Feature Importance — Electrode Connectivity Patterns",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = save_dir / "shap_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ SHAP summary saved: {}".format(path))


def plot_electrode_importance(shap_values, y_explain, save_dir):
    """Plot per-electrode importance aggregated across all connections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {"AD": "#E53935", "FTD": "#FB8C00", "CN": "#43A047"}
    x = np.arange(19)
    width = 0.25
    
    for cls_idx, cls_name in LABEL_NAMES.items():
        cls_mask = y_explain == cls_idx
        n_cls = int(cls_mask.sum())
        if n_cls == 0:
            continue
        
        cls_shap = shap_values[cls_idx][cls_mask]  # (n_cls, 19, 19, 3)
        mean_shap = np.mean(np.abs(cls_shap), axis=(0, 3))  # → (19, 19)
        
        electrode_imp = mean_shap.sum(axis=0) + mean_shap.sum(axis=1)
        electrode_imp = electrode_imp / (electrode_imp.max() + 1e-10)
        
        offset = (cls_idx - 1) * width
        ax.bar(x + offset, electrode_imp, width, label=cls_name,
               color=colors[cls_name], alpha=0.85, edgecolor="white",
               linewidth=0.5)
    
    ax.set_xlabel("EEG Electrode", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized SHAP Importance", fontsize=14, fontweight="bold")
    ax.set_title("Electrode Importance for Dementia Classification",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, fontsize=10, rotation=45, ha="right")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    path = save_dir / "shap_electrode_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Electrode importance saved: {}".format(path))


def plot_band_importance(shap_values, y_explain, save_dir):
    """Plot frequency band importance (alpha vs beta vs gamma)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_band = ["#1E88E5", "#FDD835", "#E53935"]
    x = np.arange(3)
    width = 0.22
    
    for band_idx, band_name in enumerate(BAND_NAMES):
        band_importances = []
        for cls_idx in range(N_CLASSES):
            cls_mask = y_explain == cls_idx
            n_cls = int(cls_mask.sum())
            if n_cls == 0:
                band_importances.append(0.0)
                continue
            
            cls_shap = shap_values[cls_idx][cls_mask]  # (n_cls, 19, 19, 3)
            mean_val = float(np.mean(np.abs(cls_shap[:, :, :, band_idx])))
            band_importances.append(mean_val)
        
        offset = (band_idx - 1) * width
        bars = ax.bar(x + offset, band_importances, width, label=band_name,
                      color=colors_band[band_idx], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        
        for bar, val in zip(bars, band_importances):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        "{:.4f}".format(val), ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    
    ax.set_xlabel("Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean |SHAP| Value", fontsize=14, fontweight="bold")
    ax.set_title("Frequency Band Importance per Class",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(["AD", "FTD", "CN"], fontsize=13)
    ax.legend(fontsize=12, title="Band", title_fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    path = save_dir / "shap_band_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Band importance saved: {}".format(path))


def plot_class_comparison(shap_values, y_explain, save_dir):
    """Plot per-class SHAP difference heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    class_shap = {}
    for cls_idx, cls_name in LABEL_NAMES.items():
        cls_mask = y_explain == cls_idx
        n_cls = int(cls_mask.sum())
        if n_cls == 0:
            class_shap[cls_name] = np.zeros((19, 19))
            continue
        cls_shap_data = shap_values[cls_idx][cls_mask]  # (n_cls, 19, 19, 3)
        class_shap[cls_name] = np.mean(cls_shap_data, axis=(0, 3))  # → (19, 19)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    comparisons = [
        ("AD vs CN", "AD", "CN"),
        ("FTD vs CN", "FTD", "CN"),
        ("AD vs FTD", "AD", "FTD"),
    ]
    
    for i, (title, cls_a, cls_b) in enumerate(comparisons):
        ax = axes[i]
        diff = class_shap[cls_a] - class_shap[cls_b]
        
        vmax = max(abs(diff.min()), abs(diff.max())) or 1e-6
        
        sns.heatmap(
            diff, ax=ax, cmap="RdBu_r", center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            square=True, linewidths=0.1, linecolor="gray",
            cbar_kws={"shrink": 0.8, "label": "SHAP diff"},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=7)
    
    fig.suptitle(
        "Class-Discriminative Connectivity Patterns (SHAP Differences)",
        fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = save_dir / "shap_class_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Class comparison saved: {}".format(path))


# ─── Main ────────────────────────────────────────────────────────────────────

def run_explanation(fold_idx=None, n_background=100, n_explain=200):
    """Run full SHAP explanation pipeline."""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print()
    print("=" * 55)
    print("  SHAP Explainability Analysis")
    print("=" * 55)
    
    if fold_idx is None:
        fold_idx, best_acc = find_best_fold()
        print("  Best fold: {} (subject accuracy: {:.1%})".format(fold_idx, best_acc))
    else:
        print("  Using fold: {}".format(fold_idx))
    
    model_path = MODELS_DIR / "fold_{:03d}".format(fold_idx) / "model.keras"
    if not model_path.exists():
        print("  ✗ Model not found: {}".format(model_path))
        print("  Run training first: python train_model.py")
        return
    
    print("  Loading model from {}...".format(model_path))
    model = tf.keras.models.load_model(str(model_path))
    
    print("  Loading fold {} data...".format(fold_idx))
    X_train, y_train, _, X_val, y_val, _ = get_fold_val_data(fold_idx)
    
    # Stratified background sample
    np.random.seed(42)
    bg_per_class = max(n_background // N_CLASSES, 10)
    bg_indices = []
    for cls in range(N_CLASSES):
        cls_indices = np.where(y_train == cls)[0]
        n_pick = min(bg_per_class, len(cls_indices))
        chosen = np.random.choice(cls_indices, n_pick, replace=False)
        bg_indices.extend(chosen.tolist())
    X_background = X_train[bg_indices].astype(np.float32)
    
    # Stratified explanation sample
    exp_per_class = max(n_explain // N_CLASSES, 10)
    exp_indices = []
    for cls in range(N_CLASSES):
        cls_indices = np.where(y_val == cls)[0]
        n_pick = min(exp_per_class, len(cls_indices))
        chosen = np.random.choice(cls_indices, n_pick, replace=False)
        exp_indices.extend(chosen.tolist())
    X_explain = X_val[exp_indices].astype(np.float32)
    y_explain = y_val[exp_indices]
    
    print("  Background: {} | Explain: {} ({})".format(
        len(X_background), len(X_explain),
        ", ".join("{}: {}".format(LABEL_NAMES[c], int((y_explain == c).sum()))
                  for c in range(N_CLASSES))
    ))
    
    # Compute SHAP
    shap_values = compute_shap_values(model, X_background, X_explain)
    
    # Generate plots
    print()
    print("  Generating thesis-quality plots...")
    plot_shap_summary(shap_values, X_explain, y_explain, FIGURES_DIR)
    plot_electrode_importance(shap_values, y_explain, FIGURES_DIR)
    plot_band_importance(shap_values, y_explain, FIGURES_DIR)
    plot_class_comparison(shap_values, y_explain, FIGURES_DIR)
    
    # Save raw values
    shap_path = RESULTS_DIR / "shap_values.npz"
    np.savez_compressed(
        shap_path,
        shap_class_0=shap_values[0],
        shap_class_1=shap_values[1],
        shap_class_2=shap_values[2],
        X_explain=X_explain,
        y_explain=y_explain,
    )
    
    print()
    print("  ✓ SHAP values saved: {}".format(shap_path))
    print("  ✓ All figures saved to: {}".format(FIGURES_DIR))
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP Explainability for Dementia CNN"
    )
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index to explain (default: best fold)")
    parser.add_argument("--n-background", type=int, default=100,
                        help="Number of background samples for SHAP")
    parser.add_argument("--n-explain", type=int, default=200,
                        help="Number of samples to explain")
    args = parser.parse_args()
    
    run_explanation(
        fold_idx=args.fold,
        n_background=args.n_background,
        n_explain=args.n_explain,
    )
