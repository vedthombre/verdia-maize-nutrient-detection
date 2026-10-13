# -*- coding: utf-8 -*-
"""
Maize Leaf Nutrient-Deficiency -- Classical ML Classifiers
===========================================================
Trains three classical machine-learning classifiers on CNN feature vectors
extracted from a FROZEN, ImageNet-pretrained EfficientNet-B0 backbone.

Feature extraction pipeline
----------------------------
The EfficientNet-B0 backbone (IMAGENET1K_V1 weights) is used as a fixed
feature extractor.  The final classification head is removed so that the
network outputs a 1280-dimensional feature vector (the output of the global
average-pooling layer) for each image.

IMPORTANT: The feature extractor is NOT trained on this dataset in any way.
Its weights are the original ImageNet-pretrained weights.  Only the three
downstream classifiers (SVM, Random Forest, KNN) are fitted on the extracted
training features.

Preprocessing applied before feature extraction
-------------------------------------------------
Identical to the project's deterministic val/test transform defined in
transforms.py:
    Resize(256, BILINEAR) -> CenterCrop(224) -> ToTensor()
    -> Normalize(ImageNet mean, ImageNet std)

No augmentation is applied.  The transform is deterministic so results are
reproducible.

Classifiers trained
--------------------
A. SVM (Support Vector Machine)
   - Features scaled with StandardScaler (fitted on training features only).
   - Kernel: RBF.  C=10, gamma='scale'.  Selected using validation accuracy.
   - Probability calibration enabled (probability=True) for optional
     confidence output.

B. Random Forest
   - Features used directly (tree-based models do not require scaling).
   - n_estimators=500, max_depth=None, min_samples_split=2.
   - random_state=42 for reproducibility.

C. K-Nearest Neighbours
   - Features scaled with the same StandardScaler fitted on training.
   - k=7, metric='cosine'.  Selected using validation accuracy.

Scaler note
-----------
StandardScaler is fitted ONLY on the training split and then applied (without
refitting) to validation and test features.  No test-set statistics are used
for scaling.

Hyperparameter selection
------------------------
A small manual grid was evaluated on the VALIDATION set only.  The
configurations documented above achieved the best validation accuracy.
Test labels were NEVER consulted during this process.

Outputs
-------
artifacts/
    comparison_features_train.npz  -- (X_train, y_train)
    comparison_features_val.npz    -- (X_val, y_val)
    comparison_features_test.npz   -- (X_test, y_test)
    comparison_feature_scaler.pkl  -- fitted StandardScaler
    comparison_svm.pkl             -- fitted SVM classifier
    comparison_random_forest.pkl   -- fitted Random Forest classifier
    comparison_knn.pkl             -- fitted KNN classifier
    comparison_val_grid_search.json -- validation results for all evaluated
                                       hyperparameter configurations

Usage
-----
    python train_classical_models.py
"""

import json
import pickle
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from dataset import MaizeLeafDataset, CLASS_NAMES
from transforms import val_transform   # deterministic, no augmentation

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
CSV_PATH      = PROJECT_ROOT / "dataset_split.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"   # feature cache & model files
RESULTS_DIR   = PROJECT_ROOT / "results"
SEED          = 42
BATCH_SIZE    = 32
NUM_WORKERS   = 0

# ──────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ──────────────────────────────────────────────────────────────────────
def build_feature_extractor(device):
    """
    Return a FROZEN EfficientNet-B0 with the classification head removed.

    The network outputs a 1280-d feature vector per image (post global-
    average-pool, pre dropout/linear).  No gradients are computed and no
    weights are modified — this is purely a fixed, pre-trained feature map.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # Remove the classifier (Dropout + Linear(1280, 1000)) so the network
    # outputs the 1280-d pooled feature vector.
    model.classifier = torch.nn.Identity()

    # Freeze all parameters — we are NOT training the feature extractor.
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)
    print("  Feature extractor: EfficientNet-B0 (IMAGENET1K_V1, FROZEN)")
    print("  Output feature dimension: 1280")
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract feature vectors and ground-truth labels for all samples in
    a DataLoader.

    Returns
    -------
    features : np.ndarray, shape (N, 1280)
    labels   : np.ndarray, shape (N,)
    """
    all_features = []
    all_labels   = []
    for images, labels in loader:
        images = images.to(device)
        feats  = model(images)          # (batch, 1280)
        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels,   axis=0)
    return features, labels


# ──────────────────────────────────────────────────────────────────────
# HYPERPARAMETER VALIDATION HELPER
# ──────────────────────────────────────────────────────────────────────
def evaluate_on_val(clf, X_val, y_val):
    preds = clf.predict(X_val)
    return accuracy_score(y_val, preds)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  Classical ML Classifier Training")
    print(f"  Feature extractor: EfficientNet-B0 (IMAGENET1K_V1, FROZEN)")
    print(f"  Device: {device}")
    print(f"{'='*65}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    # val_transform is deterministic: Resize(256)->CenterCrop(224)->
    # ToTensor()->Normalize(ImageNet).  No augmentation.
    train_ds = MaizeLeafDataset(CSV_PATH, "train", transform=val_transform)
    val_ds   = MaizeLeafDataset(CSV_PATH, "val",   transform=val_transform)
    test_ds  = MaizeLeafDataset(CSV_PATH, "test",  transform=val_transform)

    print("Dataset split sizes:")
    print(f"  Train : {len(train_ds)}")
    print(f"  Val   : {len(val_ds)}")
    print(f"  Test  : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # ── Feature extraction (with caching) ─────────────────────────────
    train_feat_path = ARTIFACTS_DIR / "comparison_features_train.npz"
    val_feat_path   = ARTIFACTS_DIR / "comparison_features_val.npz"
    test_feat_path  = ARTIFACTS_DIR / "comparison_features_test.npz"

    if (train_feat_path.exists() and val_feat_path.exists()
            and test_feat_path.exists()):
        print("\nLoading cached feature vectors ...")
        train_data = np.load(train_feat_path)
        X_train, y_train = train_data["features"], train_data["labels"]
        val_data = np.load(val_feat_path)
        X_val,   y_val   = val_data["features"],   val_data["labels"]
        test_data = np.load(test_feat_path)
        X_test,  y_test  = test_data["features"],  test_data["labels"]
        print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    else:
        print("\nExtracting features from frozen EfficientNet-B0 backbone ...")
        extractor = build_feature_extractor(device)

        t0 = time.time()
        print("  Extracting TRAIN features ...")
        X_train, y_train = extract_features(extractor, train_loader, device)
        print(f"  Extracting VAL features ...")
        X_val,   y_val   = extract_features(extractor, val_loader,   device)
        print(f"  Extracting TEST features ...")
        X_test,  y_test  = extract_features(extractor, test_loader,  device)
        elapsed = time.time() - t0
        print(f"  Feature extraction done in {elapsed:.1f}s")
        print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

        np.savez(train_feat_path, features=X_train, labels=y_train)
        np.savez(val_feat_path,   features=X_val,   labels=y_val)
        np.savez(test_feat_path,  features=X_test,  labels=y_test)
        print(f"  Feature vectors cached to {ARTIFACTS_DIR}/")

        # Free GPU memory
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Label distribution check
    print("\nLabel distribution (train):")
    for idx, name in enumerate(CLASS_NAMES):
        n = int((y_train == idx).sum())
        print(f"  {name:12s}: {n}")

    # ── Feature scaling ───────────────────────────────────────────────
    # StandardScaler fitted on TRAINING features only.
    # The same scaler is applied to val and test WITHOUT refitting.
    scaler_path = ARTIFACTS_DIR / "comparison_feature_scaler.pkl"
    if scaler_path.exists():
        print("\nLoading cached StandardScaler ...")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        print("\nFitting StandardScaler on TRAINING features only ...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler saved: {scaler_path}")

    X_train_sc = scaler.transform(X_train)   # train scaled
    X_val_sc   = scaler.transform(X_val)     # val scaled (no refit)
    X_test_sc  = scaler.transform(X_test)    # test scaled (no refit)

    # ── Hyperparameter search on VALIDATION set ────────────────────────
    # A small manual grid evaluated on the validation set.  Test labels
    # are NEVER used during this stage.
    print("\n" + "="*65)
    print("  Hyperparameter selection (validation set only)")
    print("="*65)

    val_grid_results = {}

    # ── A. SVM ────────────────────────────────────────────────────────
    print("\n[A] SVM — RBF kernel, manual grid on C:")
    svm_grid = [
        {"C": 1,  "gamma": "scale"},
        {"C": 5,  "gamma": "scale"},
        {"C": 10, "gamma": "scale"},
        {"C": 50, "gamma": "scale"},
    ]
    best_svm_acc  = -1
    best_svm_cfg  = None
    svm_grid_log  = []
    for cfg in svm_grid:
        clf = SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"],
                  probability=True, random_state=SEED)
        clf.fit(X_train_sc, y_train)
        val_acc = evaluate_on_val(clf, X_val_sc, y_val)
        svm_grid_log.append({"C": cfg["C"], "gamma": cfg["gamma"],
                              "val_accuracy": round(val_acc, 4)})
        print(f"   C={cfg['C']:>4},  gamma={cfg['gamma']:>6}  -> val_acc={val_acc:.4f}")
        if val_acc > best_svm_acc:
            best_svm_acc = val_acc
            best_svm_cfg = cfg
    val_grid_results["svm"] = svm_grid_log
    print(f"   Best SVM config: C={best_svm_cfg['C']}, gamma={best_svm_cfg['gamma']} "
          f"(val_acc={best_svm_acc:.4f})")

    # Train final SVM on training features using best config
    print("\n  Training final SVM ...")
    t0 = time.time()
    svm_clf = SVC(kernel="rbf", C=best_svm_cfg["C"],
                  gamma=best_svm_cfg["gamma"],
                  probability=True, random_state=SEED)
    svm_clf.fit(X_train_sc, y_train)
    svm_train_time = time.time() - t0
    print(f"  SVM trained in {svm_train_time:.2f}s")

    svm_path = ARTIFACTS_DIR / "comparison_svm.pkl"
    with open(svm_path, "wb") as f:
        pickle.dump(svm_clf, f)
    print(f"  SVM saved: {svm_path}")

    # ── B. Random Forest ──────────────────────────────────────────────
    print("\n[B] Random Forest — manual grid on n_estimators:")
    rf_grid = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 300, "max_depth": None},
        {"n_estimators": 500, "max_depth": None},
        {"n_estimators": 500, "max_depth": 20},
    ]
    best_rf_acc  = -1
    best_rf_cfg  = None
    rf_grid_log  = []
    for cfg in rf_grid:
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=SEED, n_jobs=-1
        )
        clf.fit(X_train, y_train)   # RF does not require scaling
        val_acc = evaluate_on_val(clf, X_val, y_val)
        rf_grid_log.append({
            "n_estimators": cfg["n_estimators"],
            "max_depth": str(cfg["max_depth"]),
            "val_accuracy": round(val_acc, 4)
        })
        print(f"   n_estimators={cfg['n_estimators']:>4}, max_depth={str(cfg['max_depth']):>5}"
              f"  -> val_acc={val_acc:.4f}")
        if val_acc > best_rf_acc:
            best_rf_acc = val_acc
            best_rf_cfg = cfg
    val_grid_results["random_forest"] = rf_grid_log
    print(f"   Best RF config: n_estimators={best_rf_cfg['n_estimators']}, "
          f"max_depth={best_rf_cfg['max_depth']} (val_acc={best_rf_acc:.4f})")

    print("\n  Training final Random Forest ...")
    t0 = time.time()
    rf_clf = RandomForestClassifier(
        n_estimators=best_rf_cfg["n_estimators"],
        max_depth=best_rf_cfg["max_depth"],
        random_state=SEED, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    print(f"  Random Forest trained in {rf_train_time:.2f}s")

    rf_path = ARTIFACTS_DIR / "comparison_random_forest.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf_clf, f)
    print(f"  Random Forest saved: {rf_path}")

    # ── C. KNN ────────────────────────────────────────────────────────
    print("\n[C] KNN — manual grid on k and metric:")
    knn_grid = [
        {"n_neighbors": 3,  "metric": "cosine"},
        {"n_neighbors": 5,  "metric": "cosine"},
        {"n_neighbors": 7,  "metric": "cosine"},
        {"n_neighbors": 7,  "metric": "euclidean"},
        {"n_neighbors": 11, "metric": "cosine"},
    ]
    best_knn_acc  = -1
    best_knn_cfg  = None
    knn_grid_log  = []
    for cfg in knn_grid:
        clf = KNeighborsClassifier(n_neighbors=cfg["n_neighbors"],
                                   metric=cfg["metric"], n_jobs=-1)
        clf.fit(X_train_sc, y_train)
        val_acc = evaluate_on_val(clf, X_val_sc, y_val)
        knn_grid_log.append({"n_neighbors": cfg["n_neighbors"],
                              "metric": cfg["metric"],
                              "val_accuracy": round(val_acc, 4)})
        print(f"   k={cfg['n_neighbors']:>3},  metric={cfg['metric']:>10}"
              f"  -> val_acc={val_acc:.4f}")
        if val_acc > best_knn_acc:
            best_knn_acc = val_acc
            best_knn_cfg = cfg
    val_grid_results["knn"] = knn_grid_log
    print(f"   Best KNN config: k={best_knn_cfg['n_neighbors']}, "
          f"metric={best_knn_cfg['metric']} (val_acc={best_knn_acc:.4f})")

    print("\n  Training final KNN ...")
    t0 = time.time()
    knn_clf = KNeighborsClassifier(
        n_neighbors=best_knn_cfg["n_neighbors"],
        metric=best_knn_cfg["metric"],
        n_jobs=-1
    )
    knn_clf.fit(X_train_sc, y_train)
    knn_train_time = time.time() - t0
    print(f"  KNN fitted in {knn_train_time:.2f}s")

    knn_path = ARTIFACTS_DIR / "comparison_knn.pkl"
    with open(knn_path, "wb") as f:
        pickle.dump(knn_clf, f)
    print(f"  KNN saved: {knn_path}")

    # ── Save validation grid-search log ───────────────────────────────
    grid_log_path = ARTIFACTS_DIR / "comparison_val_grid_search.json"
    with open(grid_log_path, "w") as f:
        json.dump(val_grid_results, f, indent=2)
    print(f"\nValidation grid-search log saved: {grid_log_path}")

    # ── Quick validation-set sanity check ─────────────────────────────
    print("\n" + "="*65)
    print("  Final model validation accuracy (sanity check)")
    print("="*65)
    svm_val_acc = accuracy_score(y_val, svm_clf.predict(X_val_sc))
    rf_val_acc  = accuracy_score(y_val, rf_clf.predict(X_val))
    knn_val_acc = accuracy_score(y_val, knn_clf.predict(X_val_sc))
    print(f"  SVM           : {svm_val_acc:.4f}  ({svm_val_acc:.1%})")
    print(f"  Random Forest : {rf_val_acc:.4f}  ({rf_val_acc:.1%})")
    print(f"  KNN           : {knn_val_acc:.4f}  ({knn_val_acc:.1%})")

    print("\n" + "="*65)
    print("  Training complete — all artifacts saved.")
    print("  Run evaluate_model_comparison.py to generate full metrics.")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
