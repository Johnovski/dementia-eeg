"""
Quick Demo: Cross-Dataset EEG Dementia Classification
=======================================================
Run the trained DS-CNN + MHSA model on unseen EEG data,
including data from different recording conditions.

Designed for thesis defense demonstration:
  - Full pipeline: raw .set → features → prediction
  - Formatted output with confidence scores
  - Supports single file or batch directory mode

Usage:
    # Single subject
    python quick_demo.py --eeg photic_data/sub-001/eeg/sub-001_task-photomark_eeg.set --label AD

    # Batch mode (all subjects in a directory)
    python quick_demo.py --dir photic_data/ --participants photic_data/participants.tsv

    # Original dataset comparison
    python quick_demo.py --eeg preprocessed/sub-001/eeg/sub-001_task-eyesclosed_eeg.set --label AD

    # Specify fold model (default: best fold)
    python quick_demo.py --dir photic_data/ --participants photic_data/participants.tsv --fold 0
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FEATURES_FILE = BASE_DIR / "features" / "features.h5"

LABEL_NAMES = {0: "AD", 1: "FTD", 2: "CN"}
LABEL_FULL = {
    0: "Alzheimer's Disease",
    1: "Frontotemporal Dementia",
    2: "Healthy Control",
}
GROUP_TO_LABEL = {"A": 0, "F": 1, "C": 2}
LABEL_STR_TO_INT = {"AD": 0, "FTD": 1, "CN": 2}

N_CLASSES = 3
INPUT_SHAPE = (19, 19, 3)
SFREQ = 500
WINDOW_SIZE = 2 * SFREQ  # 1000 samples
STRIDE = 1 * SFREQ       # 500 samples (50% overlap)

CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

BANDS = {
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


# ─── Feature Engineering (self-contained) ───────────────────────────────────

def bandpass_filter(data, low, high, fs=SFREQ, order=4):
    """Apply Butterworth band-pass filter."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def compute_mpc(data):
    """Compute Mean Phase Coherence between all channel pairs."""
    from scipy.signal import hilbert
    n_ch = data.shape[0]
    analytic = hilbert(data, axis=-1)
    phase = np.angle(analytic)
    mpc = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            diff = phase[i] - phase[j]
            val = np.abs(np.mean(np.exp(1j * diff)))
            mpc[i, j] = val
            mpc[j, i] = val
    return mpc


def compute_msc(data, fs=SFREQ, band=(8, 13)):
    """Compute Magnitude-Squared Coherence between all channel pairs."""
    from scipy.signal import coherence
    n_ch, n_samples = data.shape
    nperseg = max(n_samples // 8, 4)
    msc = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs, cxy = coherence(
                data[i], data[j], fs=fs, window="hamming",
                nperseg=nperseg, noverlap=nperseg // 2,
            )
            mask = (freqs >= band[0]) & (freqs <= band[1])
            val = np.mean(cxy[mask]) if np.any(mask) else 0.0
            msc[i, j] = val
            msc[j, i] = val
    return msc


def construct_3d_image(mpc_matrices, msc_matrices):
    """Build 19×19×3 image: upper=MPC, lower=MSC per band."""
    n_ch = len(CHANNEL_NAMES)
    image = np.zeros((n_ch, n_ch, 3))
    for ch_idx, band_name in enumerate(["alpha", "beta", "gamma"]):
        combined = np.zeros((n_ch, n_ch))
        upper = np.triu_indices(n_ch, k=1)
        lower = np.tril_indices(n_ch, k=-1)
        combined[upper] = mpc_matrices[band_name][upper]
        combined[lower] = msc_matrices[band_name][lower]
        image[:, :, ch_idx] = combined
    return image


def load_eeg(set_file_path):
    """Load EEG from .set file, pick and reorder 19 channels."""
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_eeglab(str(set_file_path), preload=True, verbose=False)
    available = raw.ch_names
    pick = [ch for ch in CHANNEL_NAMES if ch in available]
    if len(pick) < 19:
        missing = set(CHANNEL_NAMES) - set(pick)
        print("  ⚠ Missing channels: {}".format(missing))
        return None, None
    raw.pick(pick)
    raw.reorder_channels(pick)
    return raw.get_data(), raw.ch_names


def extract_features_from_eeg(raw_data):
    """Full pipeline: raw EEG → list of 19×19×3 feature images."""
    n_samples = raw_data.shape[1]
    images = []
    start = 0
    while start + WINDOW_SIZE <= n_samples:
        window = raw_data[:, start:start + WINDOW_SIZE]
        mpc_matrices = {}
        msc_matrices = {}
        for band_name, (low, high) in BANDS.items():
            filtered = bandpass_filter(window, low, high)
            mpc_matrices[band_name] = compute_mpc(filtered)
            msc_matrices[band_name] = compute_msc(filtered, band=(low, high))
        images.append(construct_3d_image(mpc_matrices, msc_matrices))
        start += STRIDE
    return np.array(images, dtype=np.float32) if images else None


# ─── Model Loading ──────────────────────────────────────────────────────────

def find_best_fold():
    """Find fold with highest subject accuracy.

    Checks checkpoint files first, falls back to cv_results.json.
    """
    best_fold, best_acc = 0, 0.0

    # Try checkpoint files first
    cp_files = sorted(CHECKPOINT_DIR.glob("fold_*.json"))
    if cp_files:
        for cp in cp_files:
            with open(cp) as f:
                r = json.load(f)
            if r["subject_accuracy"] > best_acc:
                best_acc = r["subject_accuracy"]
                best_fold = r["fold"]
        return best_fold, best_acc

    # Fallback: check cv_results.json
    cv_results_path = BASE_DIR / "results" / "cv_results.json"
    if cv_results_path.exists():
        with open(cv_results_path) as f:
            cv = json.load(f)
        for fold_result in cv.get("per_fold", []):
            acc = fold_result.get("subject_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_fold = fold_result["fold"]
        return best_fold, best_acc

    # Default: fold 0
    return best_fold, best_acc


def get_training_stats(fold_idx):
    """Get training set mean/std for standardization from features.h5."""
    import h5py
    from sklearn.model_selection import StratifiedKFold

    with h5py.File(FEATURES_FILE, "r") as hf:
        X = hf["X"][:]
        y = hf["y"][:]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_idx, _) in enumerate(skf.split(np.zeros(len(y)), y)):
        if i == fold_idx:
            X_train = X[train_idx]
            mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
            std = np.maximum(X_train.std(axis=(0, 1, 2), keepdims=True), 1e-8)
            return mean, std
    raise ValueError("Fold {} not found".format(fold_idx))


def load_model_and_stats(fold_idx=None):
    """Load the model and training normalization statistics."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    if fold_idx is None:
        fold_idx, best_acc = find_best_fold()
    else:
        best_acc = None

    model_path = MODELS_DIR / "fold_{:03d}".format(fold_idx) / "model.keras"
    if not model_path.exists():
        print("  ✗ Model not found: {}".format(model_path))
        sys.exit(1)

    model = tf.keras.models.load_model(str(model_path))
    mean, std = get_training_stats(fold_idx)
    return model, mean, std, fold_idx, best_acc


# ─── Participants ────────────────────────────────────────────────────────────

def load_participants_tsv(tsv_path):
    """Parse participants.tsv for ground truth labels."""
    participants = {}
    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().replace("\r", "").split("\t")
        for line in f:
            fields = line.strip().replace("\r", "").split("\t")
            if len(fields) < 5:
                continue
            row = dict(zip(header, fields))
            pid = row.get("participant_id", "").strip()
            if not pid:
                continue
            participants[pid] = {
                "group": row.get("Group", "").strip(),
                "age": row.get("Age", "").strip(),
                "gender": row.get("Gender", "").strip(),
                "mmse": row.get("MMSE", "").strip(),
            }
    return participants


# ─── Inference ───────────────────────────────────────────────────────────────

def predict_subject(model, mean, std, eeg_path, label=None):
    """Run full inference on a single .set file.

    Returns:
        dict with prediction results, or None on failure
    """
    eeg_path = Path(eeg_path)
    sub_id = eeg_path.parent.parent.name  # photic_data/sub-001/eeg/file.set → sub-001

    # Load EEG
    raw_data, ch_names = load_eeg(eeg_path)
    if raw_data is None:
        return None

    duration = raw_data.shape[1] / SFREQ

    # Extract features
    X = extract_features_from_eeg(raw_data)
    if X is None or len(X) == 0:
        print("  ⚠ No windows extracted (recording too short)")
        return None

    # Standardize with training set statistics
    X_norm = ((X - mean) / std).astype(np.float32)

    # Predict
    proba = model.predict(X_norm, verbose=0)
    preds = np.argmax(proba, axis=1)

    # Subject-level: majority vote
    vote_counts = Counter(preds.tolist())
    pred_label = vote_counts.most_common(1)[0][0]

    # Average probabilities
    avg_proba = proba.mean(axis=0)

    # Ground truth
    true_label = None
    if label is not None:
        if label.upper() in LABEL_STR_TO_INT:
            true_label = LABEL_STR_TO_INT[label.upper()]

    return {
        "subject": sub_id,
        "file": str(eeg_path.name),
        "duration_s": duration,
        "n_windows": len(X),
        "pred_label": pred_label,
        "pred_name": LABEL_NAMES[pred_label],
        "pred_full": LABEL_FULL[pred_label],
        "avg_proba": avg_proba.tolist(),
        "vote_counts": {LABEL_NAMES[k]: int(v) for k, v in sorted(vote_counts.items())},
        "true_label": true_label,
        "true_name": LABEL_NAMES[true_label] if true_label is not None else None,
        "correct": (pred_label == true_label) if true_label is not None else None,
    }


# ─── Display ─────────────────────────────────────────────────────────────────

def print_header(test_source="unknown"):
    """Print demo header."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   EEG Dementia Classification — Cross-Dataset Inference     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Model     : Hybrid DS-CNN + MHSA (98,359 params)          ║")
    print("║  Trained on: ds004504 (eyes-closed resting state)           ║")
    print("║  Testing on: {:<47}║".format(test_source))
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def print_subject_result(result):
    """Print formatted result for one subject."""
    r = result
    # Subject header
    print("  Subject: {}".format(r["subject"]))
    print("  File   : {}".format(r["file"]))
    print("  Duration: {:.1f}s ({:.1f} min) → {} windows".format(
        r["duration_s"], r["duration_s"] / 60, r["n_windows"]))
    print()

    # Ground truth
    if r["true_name"] is not None:
        print("    Ground Truth : {} ({})".format(
            LABEL_FULL[r["true_label"]], r["true_name"]))

    # Prediction
    marker = ""
    if r["correct"] is True:
        marker = " ✓"
    elif r["correct"] is False:
        marker = " ✗"

    print("    Prediction   : {} ({}){}".format(
        r["pred_full"], r["pred_name"], marker))

    # Confidence
    proba = r["avg_proba"]
    print("    Confidence   : AD={:.1%}  FTD={:.1%}  CN={:.1%}".format(
        proba[0], proba[1], proba[2]))

    # Vote distribution
    votes = r["vote_counts"]
    vote_str = ", ".join("{}: {}".format(k, v) for k, v in votes.items())
    print("    Window votes : {} total → {}".format(r["n_windows"], vote_str))
    print()


def print_summary(results):
    """Print batch summary table."""
    n_total = len(results)
    has_labels = all(r["correct"] is not None for r in results)

    if has_labels:
        n_correct = sum(1 for r in results if r["correct"])
        acc = n_correct / n_total if n_total > 0 else 0
    else:
        n_correct = None

    print("━" * 62)
    print()
    if has_labels:
        print("  Cross-Dataset Summary: {}/{} correct ({:.1%})".format(
            n_correct, n_total, acc))
    else:
        print("  Cross-Dataset Summary: {} subjects".format(n_total))

    print()
    print("  ┌──────────┬───────┬──────┬────────────────────────┬─────────┐")
    print("  │ Subject  │ True  │ Pred │ Confidence             │ Correct │")
    print("  ├──────────┼───────┼──────┼────────────────────────┼─────────┤")

    for r in results:
        true = r["true_name"] if r["true_name"] else "?"
        pred = r["pred_name"]
        p = r["avg_proba"]
        conf = "AD={:.0%} FTD={:.0%} CN={:.0%}".format(p[0], p[1], p[2])
        correct_str = "  ✓  " if r["correct"] is True else \
                      "  ✗  " if r["correct"] is False else "  ?  "
        print("  │ {:<8} │ {:<5} │ {:<4} │ {:<22} │ {}   │".format(
            r["subject"], true, pred, conf, correct_str))

    print("  └──────────┴───────┴──────┴────────────────────────┴─────────┘")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset EEG Dementia Classification Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_demo.py --eeg photic_data/sub-001/eeg/sub-001_task-photomark_eeg.set --label AD
  python quick_demo.py --dir photic_data/ --participants photic_data/participants.tsv
  python quick_demo.py --eeg preprocessed/sub-001/eeg/sub-001_task-eyesclosed_eeg.set --label AD
        """,
    )
    parser.add_argument("--eeg", type=str, default=None,
                        help="Path to a single .set EEG file")
    parser.add_argument("--label", type=str, default=None,
                        help="Ground truth label: AD, FTD, or CN")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing sub-XXX/eeg/*.set files")
    parser.add_argument("--participants", type=str, default=None,
                        help="Path to participants.tsv for ground truth labels")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold model to use (default: best fold)")
    args = parser.parse_args()

    if args.eeg is None and args.dir is None:
        parser.print_help()
        sys.exit(1)

    # Load model
    print("\n  Loading model...")
    model, mean, std, fold_idx, best_acc = load_model_and_stats(args.fold)
    print("  ✓ Model loaded successfully")

    # Single file mode
    if args.eeg is not None:
        eeg_path = Path(args.eeg)
        test_source = eeg_path.name
        if "photic" in str(eeg_path).lower() or "photomark" in str(eeg_path).lower():
            test_source = "ds006036 (eyes-open photic stimulation)"
        elif "eyesclosed" in str(eeg_path).lower():
            test_source = "ds004504 (eyes-closed resting state)"

        print_header(test_source)
        print("  Processing {}...".format(eeg_path))
        result = predict_subject(model, mean, std, eeg_path, args.label)
        if result:
            print()
            print_subject_result(result)
        else:
            print("  ✗ Failed to process {}".format(eeg_path))
        return

    # Batch directory mode
    data_dir = Path(args.dir)
    participants = {}
    if args.participants:
        participants = load_participants_tsv(args.participants)

    # Auto-detect test source
    test_source = str(data_dir)
    set_files = list(data_dir.glob("sub-*/eeg/*.set"))
    if set_files:
        sample = str(set_files[0])
        if "photic" in sample.lower() or "photomark" in sample.lower():
            test_source = "ds006036 (eyes-open photic stimulation)"
        elif "eyesclosed" in sample.lower():
            test_source = "ds004504 (eyes-closed resting state)"

    print_header(test_source)

    # Find all subjects
    subject_dirs = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    if not subject_dirs:
        print("  ✗ No subject directories found in {}".format(data_dir))
        sys.exit(1)

    print("  Found {} subjects to classify".format(len(subject_dirs)))
    print()

    results = []
    for sub_dir in subject_dirs:
        sub_id = sub_dir.name
        eeg_dir = sub_dir / "eeg"
        set_files = sorted(eeg_dir.glob("*.set"))
        if not set_files:
            print("  ⚠ No .set file for {}, skipping".format(sub_id))
            continue

        eeg_file = set_files[0]

        # Get ground truth
        label = None
        if sub_id in participants:
            group = participants[sub_id]["group"]
            if group in GROUP_TO_LABEL:
                label = LABEL_NAMES[GROUP_TO_LABEL[group]]

        print("  Processing {} ({})...".format(
            sub_id, "GT: " + label if label else "no label"))

        result = predict_subject(model, mean, std, eeg_file, label)
        if result:
            # Add extra metadata from participants
            if sub_id in participants:
                result["age"] = participants[sub_id]["age"]
                result["gender"] = participants[sub_id]["gender"]
                result["mmse"] = participants[sub_id]["mmse"]
            results.append(result)
            marker = ""
            if result["correct"] is True:
                marker = "✓"
            elif result["correct"] is False:
                marker = "✗"
            print("    → {} ({:.0%}) {}".format(
                result["pred_name"],
                result["avg_proba"][result["pred_label"]],
                marker))

    if not results:
        print("  ✗ No subjects processed successfully")
        sys.exit(1)

    # Print detailed results
    print()
    print("━" * 62)
    print("  Detailed Results")
    print("━" * 62)
    for r in results:
        print_subject_result(r)

    # Print summary table
    print_summary(results)


if __name__ == "__main__":
    main()
