"""
src/stacking_ensemble.py
========================

Description:
    This module implements the Stacking Ensemble logic used to classify the CNN embeddings.
    It follows a "Level-0" -> "Level-1" hierarchy to combine the strengths of multiple
    diverse algorithms.

Components:
    1.  **HybridStackingEnsemble**: A custom scikit-learn compatible estimator.
    2.  **Base Learners (Level-0)**:
        - **RandomForest**: Handling high variance and noise.
        - **HistGradientBoosting**: Capturing non-linear feature interactions efficiently.
        - **MLPClassifier**: Providing a neural-network based decision boundary.
    3.  **Meta Learner (Level-1)**:
        - **LogisticRegression**: Aggregates the probability outputs of the base learners
          to produce the final prediction.

Usage:
    ensemble = HybridStackingEnsemble()
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier # Optional: If you prefer XGB over HistGradientBoosting

class HybridStackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        rf_n_estimators=200,
        xgb_learning_rate=0.1, 
        mlp_hidden_sizes=(64, 32),
        nb_var_smoothing=1e-9,
        meta_C=1.0,
        model_subset=None,  # NEW: List of models to use, e.g., ['rf', 'mlp', 'xgb']
        n_jobs=-1, 
        random_state=42
    ):
        """
        Args:
            rf_n_estimators (int): Trees for Random Forest.
            xgb_learning_rate (float): Learning rate for Gradient Boosting.
            mlp_hidden_sizes (tuple): Hidden layers for MLP.
            nb_var_smoothing (float): Variance smoothing for Naive Bayes.
            meta_C (float): Regularization strength for Meta-Learner.
            model_subset (list): List of model names to use. If None, uses all 4.
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Hyperparameters (saved for cloning/tuning)
        self.rf_n_estimators = rf_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.nb_var_smoothing = nb_var_smoothing
        self.meta_C = meta_C
        self.model_subset = model_subset if model_subset is not None else ['rf', 'xgb', 'mlp', 'nb']

        # Level-0 Base Learners (all possible models)
        all_base_models = {
            "rf": RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=None,
                n_jobs=n_jobs,
                random_state=random_state,
                class_weight="balanced"
            ),
            # Alternative: If you specifically want XGBoost package
            "xgb": XGBClassifier(learning_rate=xgb_learning_rate, 
                                 n_estimators=50, n_jobs=n_jobs,
                                 early_stopping_rounds=20),
            
            "mlp": MLPClassifier(
                hidden_layer_sizes=mlp_hidden_sizes,
                max_iter=500,
                random_state=random_state,
                early_stopping=True
            ),
            "nb": GaussianNB(var_smoothing=nb_var_smoothing)
        }
        
        # Select only the models specified in model_subset
        self.base_models = {name: all_base_models[name] for name in self.model_subset}

        # Level-1 Meta Learner
        self.meta_model = LogisticRegression(
            C=meta_C,
            class_weight="balanced", 
            random_state=random_state, 
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial'
        )

    def fit(self, X, y):
        """
        Trains base models and then the meta-learner.
        """
        # 1. Train Base Models
        meta_features = []
        for name, model in self.base_models.items():
            model.fit(X, y)
            
            # Predict probabilities on training data 
            # (Note: For rigorous Stacking, use cross_val_predict here, but this is faster for prototyping)
            probas = model.predict_proba(X)
            meta_features.append(probas)

        # 2. Stack Predictions (Column-wise)
        X_meta = np.hstack(meta_features)

        # 3. Train Meta-Learner
        self.meta_model.fit(X_meta, y)
        return self

    def predict(self, X):
        """Returns Class Labels"""
        meta_features = []
        for name, model in self.base_models.items():
            probas = model.predict_proba(X)
            meta_features.append(probas)

        X_meta = np.hstack(meta_features)
        return self.meta_model.predict(X_meta)

    def predict_proba(self, X):
        """Returns Class Probabilities"""
        meta_features = []
        for name, model in self.base_models.items():
            probas = model.predict_proba(X)
            meta_features.append(probas)

        X_meta = np.hstack(meta_features)
        return self.meta_model.predict_proba(X_meta)