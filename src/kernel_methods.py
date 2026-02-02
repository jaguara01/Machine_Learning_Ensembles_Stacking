"""
src/kernel_methods.py
=====================

Description:
    This module implements Kernel Principal Component Analysis (KPCA) for
    dimensionality reduction of CNN embeddings. It is used in the "KPCA-Stacking"
    branch of the comparative experiment.

Key Functions:
    - **apply_kpca**:
        1. **Standardization**: Scales input features to zero mean and unit variance.
        2. **Subsampling**: Uses a random subset of data to fit the Kernel Matrix
           (to avoid memory OOM on large datasets).
        3. **Transformation**: Projects the high-dimensional embeddings (128d) into
           a lower-dimensional manifold (e.g., 50d) using an RBF or Cosine kernel.
        4. **Validation**: Quickly evaluates the quality of the projection using
           a fast LinearSVC probe on a subset of the data.
"""
import torch
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC  # Faster than SVC
from sklearn.metrics import accuracy_score


def apply_kpca(
    train_features,
    train_labels,
    val_features=None,
    val_labels=None,
    n_components=50,
    kernel="rbf",
    gamma=None,
    max_fit_samples=10000,
    random_state=42,
):
    """
    Applies Kernel PCA and calculates a validation score using a FAST linear probe.
    """
    print(
        f"\n[KPCA] Initializing (components={n_components}, kernel='{kernel}', gamma={gamma})..."
    )

    # 1. Convert to NumPy
    X_train = (
        train_features.cpu().numpy()
        if torch.is_tensor(train_features)
        else train_features
    )
    y_train = (
        train_labels.cpu().numpy() if torch.is_tensor(train_labels) else train_labels
    )

    if val_features is not None:
        X_val = (
            val_features.cpu().numpy()
            if torch.is_tensor(val_features)
            else val_features
        )
        y_val = val_labels.cpu().numpy() if torch.is_tensor(val_labels) else val_labels
    else:
        X_val, y_val = None, None

    # 2. Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)

    # 3. Subsampling for Fitting
    n_samples = X_train_scaled.shape[0]
    if n_samples > max_fit_samples:
        print(
            f"[KPCA] Subsampling {max_fit_samples}/{n_samples} for kernel matrix construction..."
        )
        indices = np.random.choice(n_samples, max_fit_samples, replace=False)
        X_fit = X_train_scaled[indices]
    else:
        X_fit = X_train_scaled

    # 4. Fit KPCA
    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=False,
        n_jobs=-1,
        random_state=random_state,
    )

    kpca.fit(X_fit)

    # 5. Transform
    print(f"[KPCA] Transforming datasets...")
    X_train_kpca = kpca.transform(X_train_scaled)
    X_val_kpca = kpca.transform(X_val_scaled) if X_val is not None else None

    # 6. VALIDATION SCORE (OPTIMIZED)
    # Only use a small subset (max 5000) for this check to avoid hanging
    if X_val_kpca is not None and y_val is not None:
        print("[KPCA] Evaluating Projection Quality (Fast Linear Probe)...")

        # --- SPEED FIX: Use subset + LinearSVC ---
        limit = 5000
        X_train_sub = X_train_kpca[:limit]
        y_train_sub = y_train[:limit]
        X_val_sub = X_val_kpca[:limit]
        y_val_sub = y_val[:limit]

        # LinearSVC is much faster than SVC(kernel='linear')
        clf = LinearSVC(random_state=random_state)
        clf.fit(X_train_sub, y_train_sub)
        acc = clf.score(X_val_sub, y_val_sub)
        print(f"       >> kPCA Projection Accuracy (Approx): {acc*100:.2f}%")

    return (
        torch.tensor(X_train_kpca, dtype=torch.float32),
        (
            torch.tensor(X_val_kpca, dtype=torch.float32)
            if X_val_kpca is not None
            else None
        ),
        kpca,
    )
