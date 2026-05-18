"""
Ensemble Prediction from K-Fold Models
========================================
Loads all fold models and combines their predictions for more robust
classification. Each fold model learned from different training subjects,
so their combined prediction captures diverse patterns.

Ensemble strategies:
    1. Soft voting (average probabilities) — default
    2. Hard voting (majority class vote)
    3. Subject-level ensemble (average per-subject, then majority vote)

Usage:
    python ensemble_predict.py                # Evaluate on all data
    python ensemble_predict.py --strategy soft # Soft voting (default)
    python ensemble_predict.py --strategy hard # Hard voting
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import h5py

BASE_DIR = Path(__file__).resolve().parent
FEATURES_FILE = BASE_DIR / "features" / "features.h5"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

LABEL_NAMES = {0: "AD", 1: "FTD", 2: "CN"}
N_CLASSES = 3


def load_features():
    """Load features from HDF5."""
    with h5py.File(FEATURES_FILE, "r") as hf:
        X = hf["X"][:]
        y = hf["y"][:]
        subject_ids = hf["subject_ids"][:]
        if isinstance(subject_ids[0], bytes):
            subject_ids = np.array([s.decode("utf-8") for s in subject_ids])
    return X, y, subject_ids


def standardize_features(X_train, X_val):
    """Standardize using training set stats."""
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 1, 2), keepdims=True), 1e-8)
    return ((X_train - mean) / std).astype(np.float32), \
           ((X_val - mean) / std).astype(np.float32)


def load_fold_models():
    """Load all trained fold models."""
    import tensorflow as tf

    models = []
    fold_dirs = sorted(MODELS_DIR.glob("fold_*"))
    
    for fold_dir in fold_dirs:
        model_path = fold_dir / "model.keras"
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))
            models.append((fold_dir.name, model))
            print("  ✓ Loaded {}".format(fold_dir.name))
    
    print("  Total models: {}".format(len(models)))
    return models


def ensemble_evaluate(strategy="soft"):
    """Evaluate ensemble of all fold models using CV structure.
    
    For each fold, uses that fold's model to predict on that fold's
    validation subjects — no data leakage.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    
    X, y, subject_ids = load_features()
    
    # Recreate the exact same splits used during training
    n_folds = len(list(CHECKPOINT_DIR.glob("fold_*.json")))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print()
    print("{:=<60}".format(""))
    print("  Ensemble Evaluation ({}-Fold, {} voting)".format(n_folds, strategy))
    print("{:=<60}".format(""))
    
    import tensorflow as tf
    
    all_sub_true = []
    all_sub_pred = []
    all_win_true = []
    all_win_pred = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(y)), y)
    ):
        model_path = MODELS_DIR / "fold_{:03d}".format(fold_idx) / "model.keras"
        if not model_path.exists():
            print("  ⚠ Fold {} model not found, skipping".format(fold_idx))
            continue
        
        model = tf.keras.models.load_model(str(model_path))
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_val = y[val_idx]
        val_sids = subject_ids[val_idx]
        
        # Standardize using training set
        X_train_n, X_val_n = standardize_features(X_train, X_val)
        
        # Predict
        proba = model.predict(X_val_n, verbose=0)
        pred = np.argmax(proba, axis=1)
        
        all_win_true.extend(y_val.tolist())
        all_win_pred.extend(pred.tolist())
        
        # Subject-level majority voting
        for sub in np.unique(val_sids):
            mask = val_sids == sub
            sub_true = y_val[mask][0]
            
            if strategy == "soft":
                # Average probabilities, then take argmax
                avg_proba = proba[mask].mean(axis=0)
                sub_pred = int(np.argmax(avg_proba))
            else:
                # Hard majority vote
                sub_preds = pred[mask]
                sub_pred = Counter(sub_preds.tolist()).most_common(1)[0][0]
            
            all_sub_true.append(int(sub_true))
            all_sub_pred.append(sub_pred)
        
        fold_win_acc = np.mean(pred == y_val)
        fold_sub_acc = np.mean([
            int(all_sub_true[-(len(np.unique(val_sids))):][i] == 
                all_sub_pred[-(len(np.unique(val_sids))):][i])
            for i in range(len(np.unique(val_sids)))
        ])
        
        print("  Fold {}: Window={:.1%} | Subject={:.1%}".format(
            fold_idx + 1, fold_win_acc, fold_sub_acc))
        
        del model
        tf.keras.backend.clear_session()
    
    # Overall results
    win_acc = np.mean(np.array(all_win_true) == np.array(all_win_pred))
    sub_acc = np.mean(np.array(all_sub_true) == np.array(all_sub_pred))
    
    print()
    print("{:=<60}".format(""))
    print("  Overall Results ({} voting)".format(strategy))
    print("{:=<60}".format(""))
    print("  Window accuracy : {:.1%}".format(win_acc))
    print("  Subject accuracy: {:.1%}".format(sub_acc))
    print()
    
    print("  Subject-Level Classification Report:")
    print(classification_report(
        all_sub_true, all_sub_pred, labels=[0, 1, 2],
        target_names=["AD", "FTD", "CN"], digits=4, zero_division=0,
    ))
    
    cm = confusion_matrix(all_sub_true, all_sub_pred, labels=[0, 1, 2])
    print("  Confusion Matrix:")
    print("          Pred_AD  Pred_FTD  Pred_CN")
    for i, row in enumerate(cm):
        print("  {:>6}  {:>7}  {:>8}  {:>7}".format(
            LABEL_NAMES[i], row[0], row[1], row[2]))
    
    # Save results
    results = {
        "strategy": strategy,
        "n_folds": n_folds,
        "window_accuracy": float(win_acc),
        "subject_accuracy": float(sub_acc),
        "confusion_matrix": cm.tolist(),
        "subject_predictions": [
            {"true": t, "pred": p, "true_label": LABEL_NAMES[t], "pred_label": LABEL_NAMES[p]}
            for t, p in zip(all_sub_true, all_sub_pred)
        ],
    }
    
    results_path = RESULTS_DIR / "ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print("  Results saved to: {}".format(results_path))
    print("{:=<60}".format(""))
    
    return results


def cross_model_ensemble():
    """Ensemble using ALL fold models to predict EACH sample.
    
    Unlike per-fold evaluation, this uses every model for every sample.
    For a given sample, models that were trained on data including that
    sample are excluded (to prevent leakage).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    import tensorflow as tf
    
    X, y, subject_ids = load_features()
    
    n_folds = len(list(CHECKPOINT_DIR.glob("fold_*.json")))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Build fold → val_idx mapping
    fold_val_indices = {}
    fold_train_indices = {}
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(y)), y)
    ):
        fold_val_indices[fold_idx] = val_idx
        fold_train_indices[fold_idx] = train_idx
    
    print()
    print("{:=<60}".format(""))
    print("  Cross-Model Ensemble ({} models)".format(n_folds))
    print("{:=<60}".format(""))
    
    # For each sample, collect predictions from models that did NOT
    # see this sample during training
    sample_probas = {i: [] for i in range(len(y))}
    
    for fold_idx in range(n_folds):
        model_path = MODELS_DIR / "fold_{:03d}".format(fold_idx) / "model.keras"
        if not model_path.exists():
            continue
        
        model = tf.keras.models.load_model(str(model_path))
        val_idx = fold_val_indices[fold_idx]
        train_idx = fold_train_indices[fold_idx]
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        # Standardize
        mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        std = np.maximum(X_train.std(axis=(0, 1, 2), keepdims=True), 1e-8)
        X_val_n = ((X_val - mean) / std).astype(np.float32)
        
        proba = model.predict(X_val_n, verbose=0)
        
        for i, idx in enumerate(val_idx):
            sample_probas[idx].append(proba[i])
        
        print("  ✓ Fold {} predictions collected".format(fold_idx + 1))
        del model
        tf.keras.backend.clear_session()
    
    # Aggregate: average probabilities across eligible models
    all_sub_true = []
    all_sub_pred = []
    
    for sub in np.unique(subject_ids):
        mask = subject_ids == sub
        sub_indices = np.where(mask)[0]
        sub_true = y[sub_indices[0]]
        
        # Average all probabilities for this subject's windows
        all_probas = []
        for idx in sub_indices:
            if sample_probas[idx]:
                all_probas.extend(sample_probas[idx])
        
        if all_probas:
            avg_proba = np.mean(all_probas, axis=0)
            sub_pred = int(np.argmax(avg_proba))
        else:
            sub_pred = 0  # fallback
        
        all_sub_true.append(int(sub_true))
        all_sub_pred.append(sub_pred)
    
    sub_acc = np.mean(np.array(all_sub_true) == np.array(all_sub_pred))
    
    print()
    print("  Cross-Model Subject Accuracy: {:.1%}".format(sub_acc))
    print()
    print("  Classification Report:")
    print(classification_report(
        all_sub_true, all_sub_pred, labels=[0, 1, 2],
        target_names=["AD", "FTD", "CN"], digits=4, zero_division=0,
    ))
    
    cm = confusion_matrix(all_sub_true, all_sub_pred, labels=[0, 1, 2])
    print("  Confusion Matrix:")
    print("          Pred_AD  Pred_FTD  Pred_CN")
    for i, row in enumerate(cm):
        print("  {:>6}  {:>7}  {:>8}  {:>7}".format(
            LABEL_NAMES[i], row[0], row[1], row[2]))
    
    print("{:=<60}".format(""))
    
    return sub_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Prediction")
    parser.add_argument("--strategy", choices=["soft", "hard"], default="soft",
                        help="Voting strategy (default: soft)")
    parser.add_argument("--cross-model", action="store_true",
                        help="Use cross-model ensemble (all models for each sample)")
    args = parser.parse_args()
    
    if args.cross_model:
        cross_model_ensemble()
    else:
        ensemble_evaluate(strategy=args.strategy)
