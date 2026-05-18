"""
Results Visualization for K-Fold CV Training
=============================================
Generates thesis-quality plots from K-Fold CV results.

Usage:
    python plot_results.py                         # Plot from default results
    python plot_results.py --results path/to.json  # Custom results path
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import seaborn as sns

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LABEL_NAMES = {0: "AD", 1: "FTD", 2: "CN"}


# ─── Plot Functions ──────────────────────────────────────────────────────────

def plot_confusion_matrix(results, save_dir):
    """Plot confusion matrix heatmap."""
    cm = np.array(results["confusion_matrix"])
    labels = [LABEL_NAMES[i] for i in range(3)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 18, "weight": "bold"},
        linewidths=0.5, linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    acc_key = "overall_subject_accuracy" if "overall_subject_accuracy" in results else "subject_accuracy"
    ax.set_title(
        "CV Confusion Matrix (Subject-Level)\nAccuracy: {:.1%}".format(
            results[acc_key]
        ),
        fontsize=16, fontweight="bold", pad=15,
    )
    ax.tick_params(labelsize=13)
    
    plt.tight_layout()
    path = save_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Confusion matrix saved: {}".format(path))


def plot_per_class_metrics(results, save_dir):
    """Plot per-class precision, recall, F1 bar chart."""
    if "classification_report" in results:
        report = results["classification_report"]
    else:
        from sklearn.metrics import classification_report
        all_true = []
        all_pred = []
        for fold in results.get("per_fold", []):
            for sr in fold.get("subject_results", []):
                all_true.append(sr["true_label"])
                all_pred.append(sr["pred_label"])
        report = classification_report(all_true, all_pred, labels=[0, 1, 2],
                                       target_names=["AD", "FTD", "CN"],
                                       output_dict=True, zero_division=0)
    
    classes = ["AD", "FTD", "CN"]
    metrics = ["precision", "recall", "f1-score"]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    
    for i, metric in enumerate(metrics):
        values = [report[c][metric] for c in classes]
        bars = ax.bar(x + i * width, values, width, label=metric.title(),
                      color=colors[i], edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    "{:.2f}".format(val), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    
    ax.set_xlabel("Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_title("Per-Class Metrics (Subject-Level CV)", fontsize=16,
                 fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    path = save_dir / "per_class_metrics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Per-class metrics saved: {}".format(path))


def plot_per_subject_results(results, save_dir):
    """Plot per-subject prediction results (aggregated across folds)."""
    per_fold = results["per_fold"]
    
    # Aggregate subject results from all folds
    all_subject_results = []
    for fold_result in per_fold:
        for sr in fold_result.get("subject_results", []):
            all_subject_results.append(sr)
    
    if not all_subject_results:
        print("  ⚠ No subject results found, skipping per-subject plot.")
        return
    
    # Sort by true label, then subject
    all_subject_results.sort(key=lambda r: (r["true_label"], r["subject"]))
    
    subjects = [r["subject"] for r in all_subject_results]
    true_labels = [r["true_label"] for r in all_subject_results]
    correct = [r["correct"] for r in all_subject_results]
    win_accs = [r["window_acc"] for r in all_subject_results]
    
    fig, ax = plt.subplots(figsize=(18, 6))
    
    colors = ["#F44336" if not c else "#4CAF50" for c in correct]
    bars = ax.bar(range(len(subjects)), win_accs, color=colors, edgecolor="none", width=0.8)
    
    # Class boundaries
    ad_end = sum(1 for t in true_labels if t == 0)
    ftd_end = ad_end + sum(1 for t in true_labels if t == 1)
    
    if ad_end > 0:
        ax.axvline(x=ad_end - 0.5, color="black", linestyle="--", alpha=0.5)
    if ftd_end > ad_end:
        ax.axvline(x=ftd_end - 0.5, color="black", linestyle="--", alpha=0.5)
    
    # Class labels
    if ad_end > 0:
        ax.text(ad_end / 2, 1.05, "AD ({})".format(ad_end), ha="center",
                fontsize=13, fontweight="bold", color="#1565C0")
    if ftd_end > ad_end:
        ax.text(ad_end + (ftd_end - ad_end) / 2, 1.05,
                "FTD ({})".format(ftd_end - ad_end), ha="center",
                fontsize=13, fontweight="bold", color="#E65100")
    if len(subjects) > ftd_end:
        ax.text(ftd_end + (len(subjects) - ftd_end) / 2, 1.05,
                "CN ({})".format(len(subjects) - ftd_end), ha="center",
                fontsize=13, fontweight="bold", color="#2E7D32")
    
    ax.set_xlabel("Subject", fontsize=14, fontweight="bold")
    ax.set_ylabel("Window Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Per-Subject Window Accuracy (K-Fold CV)\nGreen=Correct, Red=Misclassified",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([s.replace("sub-", "") for s in subjects],
                       rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color="#4CAF50", label="Correct")
    red_patch = mpatches.Patch(color="#F44336", label="Misclassified")
    ax.legend(handles=[green_patch, red_patch], fontsize=11, loc="upper right")
    
    plt.tight_layout()
    path = save_dir / "per_subject_results.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Per-subject results saved: {}".format(path))


def plot_training_summary(results, save_dir):
    """Plot summary of training epochs and val loss across folds."""
    per_fold = results["per_fold"]
    
    epochs = [r["epochs_trained"] for r in per_fold]
    val_losses = [r["best_val_loss"] for r in per_fold]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Epochs distribution
    ax1.hist(epochs, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
    ax1.axvline(np.mean(epochs), color="#F44336", linestyle="--", linewidth=2,
                label="Mean: {:.0f}".format(np.mean(epochs)))
    ax1.set_xlabel("Epochs Trained", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax1.set_title("Training Epochs Distribution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # Val loss distribution
    ax2.hist(val_losses, bins=20, color="#FF9800", edgecolor="white", alpha=0.8)
    ax2.axvline(np.mean(val_losses), color="#F44336", linestyle="--", linewidth=2,
                label="Mean: {:.4f}".format(np.mean(val_losses)))
    ax2.set_xlabel("Best Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax2.set_title("Best Validation Loss Distribution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    plt.tight_layout()
    path = save_dir / "training_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Training summary saved: {}".format(path))


# ─── Main ────────────────────────────────────────────────────────────────────

def generate_all_plots(results_path=None):
    """Generate all thesis plots from CV results."""
    if results_path is None:
        results_path = RESULTS_DIR / "cv_results.json"
    
    print()
    print("=" * 55)
    print("  Generating CV Result Plots")
    print("=" * 55)
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    acc_key = "overall_subject_accuracy" if "overall_subject_accuracy" in results else "subject_accuracy"
    print("  Subject accuracy: {:.1%}".format(results[acc_key]))
    print()
    
    plot_confusion_matrix(results, FIGURES_DIR)
    plot_per_class_metrics(results, FIGURES_DIR)
    plot_per_subject_results(results, FIGURES_DIR)
    plot_training_summary(results, FIGURES_DIR)
    
    print()
    print("  All plots saved to: {}".format(FIGURES_DIR))
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate thesis plots from LOOCV results"
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to loocv_results.json"
    )
    args = parser.parse_args()
    generate_all_plots(results_path=args.results)
