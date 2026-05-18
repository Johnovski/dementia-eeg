"""
EEG Feature Engineering Pipeline for Dementia Classification
=============================================================
Extracts Mean Phase Coherence (MPC) and Magnitude-Squared Coherence (MSC)
from preprocessed EEG .set files. Creates 3D image-like representations
(19x19x3) stacking alpha, beta and gamma frequency bands.

Usage:
    python feature_engineering.py                    # Process all subjects
    python feature_engineering.py --subjects 3       # Process first 3 subjects
    python feature_engineering.py --output out.h5    # Custom output path
    python feature_engineering.py --resume           # Resume from last checkpoint
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import h5py
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, coherence, hilbert
from tqdm import tqdm

# Suppress MNE verbose output
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
RAW_DIR = BASE_DIR / "ds004504"
PARTICIPANTS_TSV = RAW_DIR / "participants.tsv"
FEATURES_DIR = BASE_DIR / "features"
CHECKPOINT_DIR = FEATURES_DIR / "checkpoints"

SFREQ = 500  # Sampling frequency (Hz)
WINDOW_SIZE = 2 * SFREQ  # 2 seconds = 1000 samples
STRIDE = 1 * SFREQ  # 1 second = 500 samples (50% overlap)

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]
N_CHANNELS = len(CHANNEL_NAMES)  # 19

# Frequency bands
BANDS = {
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Label encoding
LABEL_MAP = {"A": 0, "F": 1, "C": 2}
LABEL_NAMES = {0: "Alzheimer's Disease", 1: "Frontotemporal Dementia", 2: "Healthy Control"}


# ─── Helper Functions ────────────────────────────────────────────────────────

def load_participants():
    """Load participant metadata from participants.tsv."""
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


def bandpass_filter(data, low, high, fs=SFREQ, order=4):
    """Apply a Butterworth band-pass filter.
    
    Args:
        data: EEG data array (n_channels, n_samples)
        low: Lower cutoff frequency (Hz)
        high: Upper cutoff frequency (Hz)
        fs: Sampling frequency
        order: Filter order
    
    Returns:
        Filtered data array
    """
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def compute_mpc(data):
    """Compute Mean Phase Coherence between all channel pairs.
    
    Uses the Hilbert transform to extract instantaneous phase,
    then computes the mean phase locking value between all pairs.
    
    Args:
        data: Band-filtered EEG (n_channels, n_samples)
    
    Returns:
        MPC matrix (n_channels, n_channels), values in [0, 1]
    """
    n_ch, n_samples = data.shape
    
    # Extract instantaneous phase via Hilbert transform
    analytic_signal = hilbert(data, axis=-1)
    phase = np.angle(analytic_signal)
    
    # Compute MPC for all pairs
    mpc_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            phase_diff = phase[i] - phase[j]
            # Mean Phase Coherence = |mean(exp(j * delta_phase))|
            mpc = np.abs(np.mean(np.exp(1j * phase_diff)))
            mpc_matrix[i, j] = mpc
            mpc_matrix[j, i] = mpc
    
    return mpc_matrix


def compute_msc(data, fs=SFREQ, band=(8, 13)):
    """Compute Magnitude-Squared Coherence between all channel pairs.
    
    Uses Welch's method with Hamming window and 8 segments.
    
    Args:
        data: Band-filtered EEG (n_channels, n_samples)
        fs: Sampling frequency
        band: (low, high) frequency band to average over
    
    Returns:
        MSC matrix (n_channels, n_channels), values in [0, 1]
    """
    n_ch, n_samples = data.shape
    nperseg = n_samples // 8  # 8 segments
    if nperseg < 4:
        nperseg = n_samples  # fallback for very short windows
    
    msc_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs, cxy = coherence(
                data[i], data[j],
                fs=fs,
                window="hamming",
                nperseg=nperseg,
                noverlap=nperseg // 2,
            )
            # Average MSC over the frequency band of interest
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            if np.any(band_mask):
                msc_val = np.mean(cxy[band_mask])
            else:
                msc_val = 0.0
            msc_matrix[i, j] = msc_val
            msc_matrix[j, i] = msc_val
    
    return msc_matrix


def construct_3d_image(mpc_matrices, msc_matrices):
    """Construct the 3D image-like representation.
    
    For each band: upper triangle = MPC, lower triangle = MSC, diagonal = 0.
    Stacks alpha, beta, gamma as 3 channels.
    
    Args:
        mpc_matrices: dict of band_name -> (19, 19) MPC matrix
        msc_matrices: dict of band_name -> (19, 19) MSC matrix
    
    Returns:
        3D image array of shape (19, 19, 3)
    """
    image = np.zeros((N_CHANNELS, N_CHANNELS, 3))
    
    for ch_idx, band_name in enumerate(["alpha", "beta", "gamma"]):
        mpc = mpc_matrices[band_name]
        msc = msc_matrices[band_name]
        
        combined = np.zeros((N_CHANNELS, N_CHANNELS))
        # Upper triangle: MPC
        upper_idx = np.triu_indices(N_CHANNELS, k=1)
        combined[upper_idx] = mpc[upper_idx]
        # Lower triangle: MSC
        lower_idx = np.tril_indices(N_CHANNELS, k=-1)
        combined[lower_idx] = msc[lower_idx]
        # Diagonal = 0 (already zero)
        
        image[:, :, ch_idx] = combined
    
    return image


def extract_windows(raw_data, window_size=WINDOW_SIZE, stride=STRIDE):
    """Extract overlapping windows from continuous EEG data.
    
    Args:
        raw_data: EEG data (n_channels, n_samples)
        window_size: Window length in samples
        stride: Step size in samples
    
    Returns:
        List of windowed data arrays, each (n_channels, window_size)
    """
    n_samples = raw_data.shape[1]
    windows = []
    start = 0
    while start + window_size <= n_samples:
        windows.append(raw_data[:, start:start + window_size])
        start += stride
    return windows


def process_window(window_data):
    """Process one window to produce a 19x19x3 feature image.
    
    Args:
        window_data: EEG window (n_channels, window_size)
    
    Returns:
        3D image (19, 19, 3) and intermediate MPC/MSC dicts for visualization
    """
    mpc_matrices = {}
    msc_matrices = {}
    
    for band_name, (low, high) in BANDS.items():
        # Band-pass filter the window
        filtered = bandpass_filter(window_data, low, high)
        
        # Compute connectivity features
        mpc_matrices[band_name] = compute_mpc(filtered)
        msc_matrices[band_name] = compute_msc(filtered, band=(low, high))
    
    # Construct the 3D image
    image_3d = construct_3d_image(mpc_matrices, msc_matrices)
    
    return image_3d, mpc_matrices, msc_matrices


def load_eeg_data(set_file_path):
    """Load EEG data from an EEGLAB .set file.
    
    Args:
        set_file_path: Path to the .set file
    
    Returns:
        raw_data: numpy array (n_channels, n_samples)
        channel_names: list of channel names
    """
    raw = mne.io.read_raw_eeglab(str(set_file_path), preload=True, verbose=False)
    
    # Ensure we have the expected channels in the expected order
    available_channels = raw.ch_names
    channels_to_pick = [ch for ch in CHANNEL_NAMES if ch in available_channels]
    
    if len(channels_to_pick) < N_CHANNELS:
        missing = set(CHANNEL_NAMES) - set(channels_to_pick)
        print(f"  ⚠ Missing channels: {missing}")
    
    raw.pick(channels_to_pick)
    raw.reorder_channels(channels_to_pick)
    
    data = raw.get_data()  # (n_channels, n_samples)
    return data, raw.ch_names


# ─── Checkpoint Helpers ──────────────────────────────────────────────────────

def get_checkpoint_path(sub_id):
    """Get the checkpoint file path for a subject."""
    return CHECKPOINT_DIR / f"{sub_id}.npz"


def is_subject_done(sub_id):
    """Check if a subject has already been processed (checkpoint exists)."""
    cp = get_checkpoint_path(sub_id)
    return cp.exists()


def save_checkpoint(sub_id, images, label, window_indices):
    """Save a single subject's extracted features to a checkpoint file.
    
    Args:
        sub_id: Subject identifier (e.g. 'sub-001')
        images: numpy array of shape (n_windows, 19, 19, 3)
        label: integer label for this subject
        window_indices: array of window indices
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_path = get_checkpoint_path(sub_id)
    np.savez_compressed(
        cp_path,
        images=images,
        label=label,
        window_indices=window_indices,
        sub_id=sub_id,
    )


def load_checkpoint(sub_id):
    """Load a subject's checkpoint.
    
    Returns:
        dict with keys: images, label, window_indices, sub_id
    """
    cp_path = get_checkpoint_path(sub_id)
    data = np.load(cp_path, allow_pickle=True)
    return {
        "images": data["images"],
        "label": int(data["label"]),
        "window_indices": data["window_indices"],
        "sub_id": str(data["sub_id"]),
    }


def merge_checkpoints(subject_ids, output_path):
    """Merge all subject checkpoints into a single HDF5 file.
    
    Args:
        subject_ids: list of subject IDs to merge
        output_path: Path for the final HDF5 file
    """
    all_images = []
    all_labels = []
    all_subject_ids = []
    all_window_indices = []
    
    for sub_id in sorted(subject_ids):
        if not is_subject_done(sub_id):
            continue
        cp = load_checkpoint(sub_id)
        n_windows = len(cp["images"])
        all_images.append(cp["images"])
        all_labels.extend([cp["label"]] * n_windows)
        all_subject_ids.extend([cp["sub_id"]] * n_windows)
        all_window_indices.extend(cp["window_indices"].tolist())
    
    X = np.concatenate(all_images, axis=0).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        hf.create_dataset("y", data=y)
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("subject_ids", data=np.array(all_subject_ids, dtype=object), dtype=dt)
        hf.create_dataset("window_indices", data=np.array(all_window_indices, dtype=np.int32))
        hf.attrs["channel_names"] = CHANNEL_NAMES
        hf.attrs["band_names"] = ["alpha", "beta", "gamma"]
        hf.attrs["label_map"] = str(LABEL_MAP)
        hf.attrs["window_size_samples"] = WINDOW_SIZE
        hf.attrs["stride_samples"] = STRIDE
        hf.attrs["sfreq"] = SFREQ
    
    return X, y


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(max_subjects=None, output_path=None, resume=False):
    """Run the full feature extraction pipeline with per-subject checkpointing.
    
    Args:
        max_subjects: If set, only process this many subjects
        output_path: Path for the output HDF5 file
        resume: If True, skip subjects that already have checkpoints
    """
    if output_path is None:
        output_path = FEATURES_DIR / "features.h5"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load participant metadata
    participants = load_participants()
    
    # Collect all subject directories
    subject_dirs = sorted([
        d for d in PREPROCESSED_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])
    
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    
    # Check for existing checkpoints
    all_sub_ids = [d.name for d in subject_dirs]
    if resume:
        already_done = [s for s in all_sub_ids if is_subject_done(s)]
        remaining = [d for d in subject_dirs if not is_subject_done(d.name)]
        print(f"  ⏩ Resuming: {len(already_done)} subjects already checkpointed, {len(remaining)} remaining.")
    else:
        remaining = subject_dirs
    
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║   EEG Feature Engineering Pipeline              ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Subjects total      : {len(subject_dirs):>4}                      ║")
    print(f"║  Subjects to process : {len(remaining):>4}                      ║")
    overlap_pct = int((1 - STRIDE / WINDOW_SIZE) * 100)
    print(f"║  Window size         : {WINDOW_SIZE} samples ({WINDOW_SIZE/SFREQ:.1f} s)       ║")
    print(f"║  Stride              : {STRIDE} samples ({STRIDE/SFREQ:.2f} s)      ║")
    print(f"║  Overlap             : {overlap_pct}%                       ║")
    print(f"║  Bands               : alpha, beta, gamma        ║")
    print(f"║  Output shape        : 19 × 19 × 3              ║")
    print(f"║  Checkpoint dir      : checkpoints/              ║")
    print(f"║  Output file         : {output_path.name:<25} ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()
    
    for sub_dir in tqdm(remaining, desc="Processing subjects", unit="sub"):
        sub_id = sub_dir.name  # e.g. "sub-001"
        
        # Find the .set file
        eeg_dir = sub_dir / "eeg"
        set_files = list(eeg_dir.glob("*.set"))
        if not set_files:
            print(f"  ⚠ No .set file found for {sub_id}, skipping.")
            continue
        
        set_file = set_files[0]
        
        # Get label
        if sub_id not in participants:
            print(f"  ⚠ No metadata for {sub_id}, skipping.")
            continue
        
        group = participants[sub_id]["group"]
        label = LABEL_MAP[group]
        
        # Load EEG
        try:
            raw_data, ch_names = load_eeg_data(set_file)
        except Exception as e:
            print(f"  ⚠ Error loading {sub_id}: {e}")
            continue
        
        # Extract windows
        windows = extract_windows(raw_data)
        
        if len(windows) == 0:
            print(f"  ⚠ No windows extracted for {sub_id} (recording too short).")
            continue
        
        # Process each window for this subject
        sub_images = []
        sub_win_indices = []
        for win_idx, window in enumerate(windows):
            image_3d, _, _ = process_window(window)
            sub_images.append(image_3d)
            sub_win_indices.append(win_idx)
        
        # Save checkpoint for this subject
        sub_images = np.array(sub_images, dtype=np.float32)
        sub_win_indices = np.array(sub_win_indices, dtype=np.int32)
        save_checkpoint(sub_id, sub_images, label, sub_win_indices)
    
    # Merge all checkpoints into final HDF5
    print(f"\n  Merging checkpoints into {output_path} ...")
    X, y = merge_checkpoints(all_sub_ids, output_path)
    
    print(f"\n{'─' * 50}")
    print(f"  Total samples (windows): {len(X)}")
    print(f"  Feature shape:           {X.shape[1:]}")
    print(f"  Label distribution:")
    for label_code, label_name in LABEL_NAMES.items():
        count = np.sum(y == label_code)
        print(f"    {label_name}: {count}")
    print(f"{'─' * 50}")
    print(f"  ✓ Features saved successfully!\n")
    
    return X, y


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Feature Engineering for Dementia Classification"
    )
    parser.add_argument(
        "--subjects", type=int, default=None,
        help="Number of subjects to process (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HDF5 file path (default: features/features.h5)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints, skipping already-processed subjects"
    )
    parser.add_argument(
        "--window", type=float, default=None,
        help="Window size in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--stride", type=float, default=None,
        help="Stride in seconds (default: 1.0)"
    )
    args = parser.parse_args()
    
    # Override window/stride if provided
    if args.window is not None:
        WINDOW_SIZE = int(args.window * SFREQ)
    if args.stride is not None:
        STRIDE = int(args.stride * SFREQ)
    
    run_pipeline(max_subjects=args.subjects, output_path=args.output, resume=args.resume)
