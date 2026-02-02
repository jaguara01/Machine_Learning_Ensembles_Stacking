"""
03_ensemble_stacking.py
=======================

Description:
    This script executes the comparative analysis of the project. It evaluates whether
    replacing the CNN's linear classifier with a Stacking Ensemble improves performance.
    It compares three approaches using 5-Fold Cross-Validation:
    1.  **Base CNN**: The standard End-to-End Deep Learning model.
    2.  **Stacking (KPCA)**: CNN Embeddings -> Kernel PCA (50d) -> Stacking Ensemble.
    3.  **Stacking (Raw)**: CNN Embeddings (128d) -> Stacking Ensemble.

Key Operations:
    1.  **Feature Extraction**: Loads the trained CNN (`cnn_model_output/best_cnn.pth`)
        and extracts the 128-dimensional latent vectors (fc1 layer) for all images.
    2.  **Cross-Validation**: Splits the features into 5 stratified folds.
    3.  **Ensemble Training**: Trains the Hybrid Stacking Ensemble (RF + GradBoost + MLP)
        on the extracted features.
    4.  **Reporting**:
        - Calculates Accuracy, F1-Score, and AUC.
        - performs statistical significance testing (T-Test).
        - Generates a text report (`results/experiment_report.txt`).
        - Plots a triple confusion matrix (`results/comparison_matrix.png`).

Usage:
    python 03_ensemble_stacking.py
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from src.data_loader import get_dataloaders
from src.cnn_architecture import FontCNN
from src.kernel_methods import apply_kpca
from src.stacking_ensemble import HybridStackingEnsemble

# --- CONFIGURATION ---
CHECKPOINT_PATH = "cnn_model_output/best_cnn.pth"
DATA_PATH = "data/processed_fonts.pt"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# KPCA Settings
N_COMPONENTS = 50
MAX_FIT_SAMPLES = 5000
N_FOLDS = 5


def extract_features_and_logits(model, loader, device):
    """
    Returns features (fc1) for Ensemble and logits (fc2) for Base CNN.
    """
    model.eval()
    all_features = []
    all_logits = []
    all_labels = []
    hook_data = {}

    def get_fc1_output(m, input, output):
        hook_data["features"] = output.detach()

    if not hasattr(model, "fc1"):
        raise AttributeError("Model does not have 'fc1' layer.")

    handle = model.fc1.register_forward_hook(get_fc1_output)
    print(f"[Data Extraction] Processing {len(loader)} batches...")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            if "features" in hook_data:
                features = hook_data["features"]
            else:
                raise RuntimeError("Hook failed.")

            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_labels.append(labels)

    handle.remove()
    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
    )


def plot_triple_confusion_matrix(y_true, pred_base, pred_kpca, pred_raw, save_path):
    """
    Plots 3 Confusion Matrices with NUMBERS (annot=True).
    """
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))

    def plot_cm(ax, y, p, title, cmap):
        cm = confusion_matrix(y, p)
        # annot=True: Show numbers
        # fmt="d": Format as integers (no scientific notation)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 10},
        )
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

    plot_cm(axes[0], y_true, pred_base, "Baseline: Standard CNN", "Blues")
    plot_cm(
        axes[1], y_true, pred_kpca, f"Stacking + KPCA ({N_COMPONENTS} dim)", "Oranges"
    )
    plot_cm(axes[2], y_true, pred_raw, "Stacking + Raw Features", "Greens")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Comparison Matrix saved to {save_path}")


def main():
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. LOAD DATA
    print("\n--- Phase 1: Data Loading ---")
    train_loader, val_loader, _, num_classes, _ = get_dataloaders(
        DATA_PATH, include_other=False
    )

    # 3. LOAD MODEL
    print(f"\n--- Phase 2: Loading CNN from {CHECKPOINT_PATH} ---")
    model = FontCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Extract data without checks
    model.load_state_dict(checkpoint["model_state_dict"])
    PHASE_1_CNN_TIME = checkpoint["training_time"]

    # 4. EXTRACT DATA
    print("Extracting features...")
    X_feat_train, logits_train, y_train = extract_features_and_logits(
        model, train_loader, device
    )
    X_feat_val, logits_val, y_val = extract_features_and_logits(
        model, val_loader, device
    )

    X_full = torch.cat([X_feat_train, X_feat_val], dim=0).numpy()
    L_full = torch.cat([logits_train, logits_val], dim=0).numpy()
    y_full = torch.cat([y_train, y_val], dim=0).numpy()

    # 5. CROSS-VALIDATION LOOP
    print(f"\n--- Starting {N_FOLDS}-Fold Comparison ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    results = {
        "base": {"acc": [], "f1": [], "auc": [], "time": []},
        "kpca": {"acc": [], "f1": [], "auc": [], "time": []},
        "raw": {"acc": [], "f1": [], "auc": [], "time": []},
    }

    all_y_true, all_pred_base, all_pred_kpca, all_pred_raw = [], [], [], []

    fold = 1
    for train_idx, test_idx in skf.split(X_full, y_full):
        print(f"\n[Fold {fold}/{N_FOLDS}] Processing...")

        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train_fold, y_test_fold = y_full[train_idx], y_full[test_idx]
        L_test = L_full[test_idx]

        # --- A. BASELINE (CNN) ---
        base_probs = F.softmax(torch.tensor(L_test), dim=1).numpy()
        base_preds = np.argmax(base_probs, axis=1)

        results["base"]["acc"].append(accuracy_score(y_test_fold, base_preds))
        results["base"]["f1"].append(f1_score(y_test_fold, base_preds, average="macro"))
        results["base"]["time"].append(PHASE_1_CNN_TIME)
        try:
            results["base"]["auc"].append(
                roc_auc_score(y_test_fold, base_probs, multi_class="ovr")
            )
        except:
            results["base"]["auc"].append(0.5)

        # --- B. ENSEMBLE + KPCA ---
        X_train_kpca, X_test_kpca, _ = apply_kpca(
            train_features=torch.tensor(X_train),
            train_labels=y_train_fold,
            val_features=torch.tensor(X_test),
            val_labels=y_test_fold,
            n_components=N_COMPONENTS,
            kernel="cosine",  # "rbf"
            max_fit_samples=MAX_FIT_SAMPLES,
        )

        start_t = time.time()
        ens_kpca = HybridStackingEnsemble(n_jobs=-1)
        ens_kpca.fit(X_train_kpca.numpy(), y_train_fold)
        results["kpca"]["time"].append(time.time() - start_t)

        kpca_preds = ens_kpca.predict(X_test_kpca.numpy())
        kpca_probs = ens_kpca.predict_proba(X_test_kpca.numpy())

        results["kpca"]["acc"].append(accuracy_score(y_test_fold, kpca_preds))
        results["kpca"]["f1"].append(f1_score(y_test_fold, kpca_preds, average="macro"))
        try:
            results["kpca"]["auc"].append(
                roc_auc_score(y_test_fold, kpca_probs, multi_class="ovr")
            )
        except:
            results["kpca"]["auc"].append(0.5)

        # --- C. ENSEMBLE + RAW ---
        start_t = time.time()
        ens_raw = HybridStackingEnsemble(n_jobs=-1)
        ens_raw.fit(X_train, y_train_fold)
        results["raw"]["time"].append(time.time() - start_t)

        raw_preds = ens_raw.predict(X_test)
        raw_probs = ens_raw.predict_proba(X_test)

        results["raw"]["acc"].append(accuracy_score(y_test_fold, raw_preds))
        results["raw"]["f1"].append(f1_score(y_test_fold, raw_preds, average="macro"))
        try:
            results["raw"]["auc"].append(
                roc_auc_score(y_test_fold, raw_probs, multi_class="ovr")
            )
        except:
            results["raw"]["auc"].append(0.5)

        # Log
        print(
            f"   Base: {results['base']['acc'][-1]:.4f} | KPCA: {results['kpca']['acc'][-1]:.4f} | Raw: {results['raw']['acc'][-1]:.4f}"
        )

        # Store for Viz
        all_y_true.extend(y_test_fold)
        all_pred_base.extend(base_preds)
        all_pred_kpca.extend(kpca_preds)
        all_pred_raw.extend(raw_preds)
        fold += 1

    # 6. REPORT GENERATION
    def get_stats(key):
        return (
            f"{np.mean(results[key]['acc']):.4f} ± {np.std(results[key]['acc']):.4f}",
            f"{np.mean(results[key]['f1']):.4f}",
            f"{np.mean(results[key]['auc']):.4f}",
            f"{np.mean(results[key]['time']):.1f}s",
        )

    report_path = os.path.join(RESULTS_DIR, "experiment_report.txt")
    with open(report_path, "w") as f:
        header = f"""
================================================================================
EXPERIMENT REPORT: CNN vs KPCA-Stacking vs Raw-Stacking
================================================================================
N_FOLDS: {N_FOLDS} | KPCA Components: {N_COMPONENTS}
================================================================================
"""
        f.write(header)
        print(header)

        row_fmt = "{:<20} {:<20} {:<12} {:<12} {:<15}\n"
        headers = ("Method", "Accuracy", "F1-Score", "AUC", "Time (s)")

        f.write(row_fmt.format(*headers))
        print(row_fmt.format(*headers))
        f.write("-" * 80 + "\n")

        f.write(row_fmt.format("Base CNN", *get_stats("base")))
        f.write(row_fmt.format("Stacking (KPCA)", *get_stats("kpca")))
        f.write(row_fmt.format("Stacking (Raw)", *get_stats("raw")))

        print(row_fmt.format("Base CNN", *get_stats("base")))
        print(row_fmt.format("Stacking (KPCA)", *get_stats("kpca")))
        print(row_fmt.format("Stacking (Raw)", *get_stats("raw")))

        # Significance
        f.write("\nStatistical Significance (Paired T-Test vs Baseline):\n")
        _, p_raw = stats.ttest_rel(results["raw"]["acc"], results["base"]["acc"])
        f.write(
            f"Raw vs Base: p={p_raw:.5f} {'(Significant)' if p_raw < 0.05 else '(Not Sig)'}\n"
        )

    print(f"\n Report saved to: {report_path}")

    # 7. PLOT
    plot_triple_confusion_matrix(
        all_y_true,
        all_pred_base,
        all_pred_kpca,
        all_pred_raw,
        os.path.join(RESULTS_DIR, "comparison_matrix.png"),
    )


if __name__ == "__main__":
    main()
