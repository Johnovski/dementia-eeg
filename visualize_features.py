"""
EEG Feature Visualization for Thesis
======================================
Generates thesis-quality plots for MPC matrices, MSC matrices,
and the combined 3D image representation.

Usage:
    python visualize_features.py                 # Default: sub-001, window 0
    python visualize_features.py --subject 5     # Specific subject number
    python visualize_features.py --window 10     # Specific window index
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

from feature_engineering import (
    BANDS, CHANNEL_NAMES, N_CHANNELS, LABEL_MAP, LABEL_NAMES,
    PREPROCESSED_DIR, SFREQ, WINDOW_SIZE, STRIDE,
    load_eeg_data, load_participants, extract_windows,
    bandpass_filter, compute_mpc, compute_msc, construct_3d_image,
)

# ─── Plot Style Configuration ────────────────────────────────────────────────

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Academic-friendly styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

BAND_DISPLAY = {"alpha": "α (8–13 Hz)", "beta": "β (13–30 Hz)", "gamma": "γ (30–45 Hz)"}
BAND_COLORS = {"alpha": "YlOrRd", "beta": "YlGnBu", "gamma": "PuRd"}


# ─── Visualization Functions ─────────────────────────────────────────────────

def plot_mpc_heatmaps(mpc_matrices, sub_id, win_idx, group_label):
    """Plot MPC matrices for all three frequency bands.
    
    Generates a figure with 3 subplots (alpha, beta, gamma) showing
    the Mean Phase Coherence between all 19 EEG channel pairs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Mean Phase Coherence (MPC) — {sub_id} ({group_label}), Window {win_idx}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    
    for idx, (band_name, band_label) in enumerate(BAND_DISPLAY.items()):
        ax = axes[idx]
        mpc = mpc_matrices[band_name]
        
        # Mask diagonal for cleaner visualization
        mask = np.eye(N_CHANNELS, dtype=bool)
        
        sns.heatmap(
            mpc,
            ax=ax,
            mask=mask,
            xticklabels=CHANNEL_NAMES,
            yticklabels=CHANNEL_NAMES,
            cmap=BAND_COLORS[band_name],
            vmin=0, vmax=1,
            square=True,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "MPC", "shrink": 0.8},
            annot=False,
        )
        ax.set_title(band_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    
    plt.tight_layout()
    
    out_path = FIGURES_DIR / f"mpc_{sub_id}_win{win_idx}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ MPC heatmap saved: {out_path}")
    return out_path


def plot_msc_heatmaps(msc_matrices, sub_id, win_idx, group_label):
    """Plot MSC matrices for all three frequency bands.
    
    Generates a figure with 3 subplots (alpha, beta, gamma) showing
    the Magnitude-Squared Coherence between all 19 EEG channel pairs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Magnitude-Squared Coherence (MSC) — {sub_id} ({group_label}), Window {win_idx}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    
    for idx, (band_name, band_label) in enumerate(BAND_DISPLAY.items()):
        ax = axes[idx]
        msc = msc_matrices[band_name]
        
        mask = np.eye(N_CHANNELS, dtype=bool)
        
        sns.heatmap(
            msc,
            ax=ax,
            mask=mask,
            xticklabels=CHANNEL_NAMES,
            yticklabels=CHANNEL_NAMES,
            cmap=BAND_COLORS[band_name],
            vmin=0, vmax=1,
            square=True,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "MSC", "shrink": 0.8},
            annot=False,
        )
        ax.set_title(band_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    
    plt.tight_layout()
    
    out_path = FIGURES_DIR / f"msc_{sub_id}_win{win_idx}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ MSC heatmap saved: {out_path}")
    return out_path


def plot_combined_3d_image(image_3d, mpc_matrices, msc_matrices, sub_id, win_idx, group_label):
    """Plot the combined 3D image representation.
    
    Shows:
    - Top row: Individual band channels (alpha, beta, gamma) of the combined image
    - Bottom: RGB-like composite view
    
    The combined matrix has upper triangle = MPC, lower triangle = MSC.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1.2], hspace=0.35, wspace=0.35)
    
    fig.suptitle(
        f"3D Feature Image (19×19×3) — {sub_id} ({group_label}), Window {win_idx}\n"
        f"Upper triangle: MPC  |  Lower triangle: MSC  |  Diagonal: 0",
        fontsize=14, fontweight="bold", y=0.98,
    )
    
    # Top row: Individual band channels
    for idx, (band_name, band_label) in enumerate(BAND_DISPLAY.items()):
        ax = fig.add_subplot(gs[0, idx])
        channel_data = image_3d[:, :, idx]
        
        sns.heatmap(
            channel_data,
            ax=ax,
            xticklabels=CHANNEL_NAMES,
            yticklabels=CHANNEL_NAMES,
            cmap="viridis",
            vmin=0, vmax=1,
            square=True,
            linewidths=0.2,
            linecolor="gray",
            cbar_kws={"label": "Value", "shrink": 0.8},
        )
        ax.set_title(f"Channel {idx}: {band_label}", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    
    # Top right: Structure annotation
    ax_struct = fig.add_subplot(gs[0, 3])
    struct_matrix = np.zeros((N_CHANNELS, N_CHANNELS))
    struct_matrix[np.triu_indices(N_CHANNELS, k=1)] = 0.8   # MPC region
    struct_matrix[np.tril_indices(N_CHANNELS, k=-1)] = 0.3  # MSC region
    
    from matplotlib.colors import ListedColormap
    cmap_struct = ListedColormap(["#2c3e50", "#3498db", "#e74c3c"])
    bounds = [0, 0.15, 0.55, 1.0]
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bounds, cmap_struct.N)
    
    sns.heatmap(
        struct_matrix,
        ax=ax_struct,
        xticklabels=CHANNEL_NAMES,
        yticklabels=CHANNEL_NAMES,
        cmap=cmap_struct,
        vmin=0, vmax=1,
        square=True,
        linewidths=0.2,
        linecolor="gray",
        cbar=False,
    )
    ax_struct.set_title("Matrix Structure", fontsize=11, fontweight="bold")
    ax_struct.tick_params(axis="x", rotation=45)
    ax_struct.tick_params(axis="y", rotation=0)
    
    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="MPC (upper triangle)"),
        Patch(facecolor="#3498db", label="MSC (lower triangle)"),
        Patch(facecolor="#2c3e50", label="Diagonal (zero)"),
    ]
    ax_struct.legend(
        handles=legend_elements, loc="upper left",
        bbox_to_anchor=(0, -0.05), fontsize=8, framealpha=0.9,
    )
    
    # Bottom: RGB composite
    ax_rgb = fig.add_subplot(gs[1, :3])
    
    # Normalize each channel to [0, 1] for RGB display
    rgb_image = np.zeros((N_CHANNELS, N_CHANNELS, 3))
    for c in range(3):
        ch = image_3d[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            rgb_image[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            rgb_image[:, :, c] = 0
    
    ax_rgb.imshow(rgb_image, interpolation="nearest", aspect="equal")
    ax_rgb.set_xticks(range(N_CHANNELS))
    ax_rgb.set_yticks(range(N_CHANNELS))
    ax_rgb.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax_rgb.set_yticklabels(CHANNEL_NAMES, fontsize=8)
    ax_rgb.set_title(
        "RGB Composite: R=α, G=β, B=γ",
        fontsize=12, fontweight="bold",
    )
    
    # Add grid lines
    for i in range(N_CHANNELS + 1):
        ax_rgb.axhline(i - 0.5, color="white", linewidth=0.3, alpha=0.5)
        ax_rgb.axvline(i - 0.5, color="white", linewidth=0.3, alpha=0.5)
    
    # Bottom right: value statistics
    ax_stats = fig.add_subplot(gs[1, 3])
    ax_stats.axis("off")
    
    stats_text = "Feature Statistics\n" + "─" * 24 + "\n\n"
    for idx, (band_name, band_label) in enumerate(BAND_DISPLAY.items()):
        ch = image_3d[:, :, idx]
        upper = ch[np.triu_indices(N_CHANNELS, k=1)]
        lower = ch[np.tril_indices(N_CHANNELS, k=-1)]
        stats_text += f"{band_label}\n"
        stats_text += f"  MPC: μ={upper.mean():.3f}, σ={upper.std():.3f}\n"
        stats_text += f"  MSC: μ={lower.mean():.3f}, σ={lower.std():.3f}\n\n"
    
    stats_text += f"Image shape: {image_3d.shape}\n"
    stats_text += f"Value range: [{image_3d.min():.4f}, {image_3d.max():.4f}]"
    
    ax_stats.text(
        0.05, 0.95, stats_text,
        transform=ax_stats.transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"),
    )
    
    out_path = FIGURES_DIR / f"3d_image_{sub_id}_win{win_idx}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 3D image plot saved: {out_path}")
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(subject_num=1, window_idx=0):
    """Generate visualization for a specific subject and window."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    sub_id = f"sub-{subject_num:03d}"
    participants = load_participants()
    
    if sub_id not in participants:
        print(f"Error: {sub_id} not found in participants.tsv")
        return
    
    group = participants[sub_id]["group"]
    group_label = {v: k for k, v in LABEL_MAP.items()}
    group_full = {
        "A": "Alzheimer's Disease",
        "F": "Frontotemporal Dementia",
        "C": "Healthy Control",
    }[group]
    
    print(f"\n{'═' * 55}")
    print(f"  Generating Thesis Visualizations")
    print(f"  Subject : {sub_id} ({group_full})")
    print(f"  MMSE    : {participants[sub_id]['mmse']}")
    print(f"  Age     : {participants[sub_id]['age']}")
    print(f"{'═' * 55}\n")
    
    # Load EEG data
    set_file = PREPROCESSED_DIR / sub_id / "eeg" / f"{sub_id}_task-eyesclosed_eeg.set"
    if not set_file.exists():
        print(f"Error: {set_file} not found")
        return
    
    print("  Loading EEG data...")
    raw_data, ch_names = load_eeg_data(set_file)
    print(f"  Shape: {raw_data.shape} ({raw_data.shape[1] / SFREQ:.1f} seconds)")
    
    # Extract windows
    windows = extract_windows(raw_data)
    print(f"  Windows extracted: {len(windows)}")
    
    if window_idx >= len(windows):
        print(f"  Warning: Window {window_idx} out of range, using window 0")
        window_idx = 0
    
    window = windows[window_idx]
    
    # Compute features for each band
    print(f"\n  Computing features for window {window_idx}...")
    mpc_matrices = {}
    msc_matrices = {}
    
    for band_name, (low, high) in BANDS.items():
        filtered = bandpass_filter(window, low, high)
        mpc_matrices[band_name] = compute_mpc(filtered)
        msc_matrices[band_name] = compute_msc(filtered, band=(low, high))
        print(f"    ✓ {band_name} band processed")
    
    # Construct 3D image
    image_3d = construct_3d_image(mpc_matrices, msc_matrices)
    print(f"    ✓ 3D image constructed: {image_3d.shape}")
    
    # Generate visualizations
    print(f"\n  Generating plots...")
    plot_mpc_heatmaps(mpc_matrices, sub_id, window_idx, group_full)
    plot_msc_heatmaps(msc_matrices, sub_id, window_idx, group_full)
    plot_combined_3d_image(image_3d, mpc_matrices, msc_matrices, sub_id, window_idx, group_full)
    
    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Feature Visualization")
    parser.add_argument("--subject", type=int, default=1, help="Subject number (default: 1)")
    parser.add_argument("--window", type=int, default=0, help="Window index (default: 0)")
    args = parser.parse_args()
    main(subject_num=args.subject, window_idx=args.window)
