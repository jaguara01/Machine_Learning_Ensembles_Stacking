"""
Staged Hyperparameter Tuning for Font Classification Pipeline

This module implements a three-stage tuning strategy:
1. CNN Architecture & Training
2. Kernel PCA Configuration  
3. Ensemble Classifier Parameters

Each stage optimizes independently, passing best configs forward.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import product
import json
import time
from typing import Dict, List, Tuple, Any
import copy


class CNNHyperparameterTuner:
    """
    Stage 1: Tune CNN architecture and training hyperparameters
    """
    
    def __init__(self, train_loader, val_loader, device, num_classes=10):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
    def create_search_space(self) -> Dict[str, List]:
        """Define CNN hyperparameter search space"""
        return {
            'base_filters': [32, 64, 128],           # First conv layer filters
            'filter_multiplier': [1, 2],              # How filters grow (2x per layer)
            'latent_dim': [64, 128, 256],            # FC1 output dimension
            'dropout': [0.3, 0.5, 0.7],              # Dropout rate
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'weight_decay': [1e-5, 1e-4, 1e-3],      # L2 regularization
            'batch_norm': [True, False],              # Use batch normalization
            'activation': ['relu', 'leaky_relu'],    # Activation function
            'num_conv_blocks': [3, 4],               # Number of conv blocks
        }
    
    def build_cnn(self, config: Dict) -> nn.Module:
        """Build CNN with given configuration"""
        
        class ConfigurableCNN(nn.Module):
            def __init__(self, num_classes, cfg):
                super().__init__()
                self.config = cfg
                
                # Choose activation
                if cfg['activation'] == 'relu':
                    self.act = nn.ReLU()
                else:
                    self.act = nn.LeakyReLU(0.2)
                
                # Build convolutional blocks
                layers = []
                in_channels = 1
                filters = cfg['base_filters']
                
                for i in range(cfg['num_conv_blocks']):
                    # Conv block
                    layers.append(nn.Conv2d(in_channels, filters, kernel_size=3, padding=1))
                    if cfg['batch_norm']:
                        layers.append(nn.BatchNorm2d(filters))
                    layers.append(self.act)
                    layers.append(nn.MaxPool2d(2))
                    layers.append(nn.Dropout2d(cfg['dropout'] * 0.5))  # Lighter dropout in conv
                    
                    in_channels = filters
                    filters *= cfg['filter_multiplier']
                
                self.features = nn.Sequential(*layers)
                
                # Calculate flattened size (depends on num_conv_blocks)
                # Start: 28x28, after each maxpool: /2
                final_size = 28 // (2 ** cfg['num_conv_blocks'])
                final_channels = cfg['base_filters'] * (cfg['filter_multiplier'] ** (cfg['num_conv_blocks'] - 1))
                flatten_size = final_channels * final_size * final_size
                
                # Fully connected layers
                self.fc1 = nn.Linear(flatten_size, cfg['latent_dim'])
                self.dropout = nn.Dropout(cfg['dropout'])
                self.fc2 = nn.Linear(cfg['latent_dim'], num_classes)
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.act(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
            
            def get_features(self, x):
                """Extract latent features from fc1"""
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.act(self.fc1(x))
                return x
        
        return ConfigurableCNN(self.num_classes, config).to(self.device)
    
    def train_and_evaluate(self, config: Dict, num_epochs: int = 10) -> Tuple[float, nn.Module]:
        """Train CNN with config and return validation accuracy"""
        model = self.build_cnn(config)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_model = None
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
        
        return best_val_acc, best_model
    
    def grid_search(self, num_epochs: int = 10, top_k: int = 3) -> List[Dict]:
        """
        Perform grid search over CNN hyperparameters
        
        Args:
            num_epochs: Training epochs per configuration
            top_k: Return top K configurations
        """
        search_space = self.create_search_space()
        
        # Generate all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        configs = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"🔍 CNN Stage 1: Testing {len(configs)} configurations...")
        print(f"   Estimated time: {len(configs) * num_epochs * 0.5:.1f} minutes\n")
        
        results = []
        
        for idx, config in enumerate(configs, 1):
            print(f"[{idx}/{len(configs)}] Testing: ", end="")
            print(f"filters={config['base_filters']}, latent={config['latent_dim']}, "
                  f"lr={config['learning_rate']:.4f}, dropout={config['dropout']}")
            
            try:
                start = time.time()
                val_acc, model = self.train_and_evaluate(config, num_epochs)
                elapsed = time.time() - start
                
                results.append({
                    'config': config,
                    'val_accuracy': val_acc,
                    'model': model,
                    'time': elapsed
                })
                
                print(f"   ✓ Validation Accuracy: {val_acc:.4f} ({elapsed:.1f}s)\n")
                
            except Exception as e:
                print(f"   ✗ Failed: {e}\n")
                continue
        
        # Sort by accuracy and return top K
        results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_k} CNN CONFIGURATIONS:")
        print(f"{'='*60}")
        for i, res in enumerate(results[:top_k], 1):
            print(f"\nRank {i}: Accuracy = {res['val_accuracy']:.4f}")
            print(f"Config: {json.dumps(res['config'], indent=2)}")
        
        return results[:top_k]


class KPCAHyperparameterTuner:
    """
    Stage 2: Tune Kernel PCA hyperparameters
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def create_search_space(self) -> Dict[str, List]:
        """Define KPCA hyperparameter search space"""
        return {
            'n_components': [8, 16, 32, 64, 100],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': [None, 0.001, 0.01, 0.1, 1.0],  # None = 1/n_features
            'degree': [2, 3, 4],  # Only for poly kernel
            'coef0': [0.0, 1.0],  # Only for poly/sigmoid
        }
    
    def apply_kpca(self, config: Dict) -> Tuple[np.ndarray, np.ndarray, KernelPCA]:
        """Apply KPCA with given configuration"""
        
        # Build params based on kernel type
        params = {
            'n_components': config['n_components'],
            'kernel': config['kernel'],
            'fit_inverse_transform': False,
            'random_state': 42
        }
        
        if config['kernel'] == 'rbf':
            params['gamma'] = config['gamma']
        elif config['kernel'] == 'poly':
            params['gamma'] = config['gamma']
            params['degree'] = config['degree']
            params['coef0'] = config['coef0']
        elif config['kernel'] == 'sigmoid':
            params['gamma'] = config['gamma']
            params['coef0'] = config['coef0']
        
        kpca = KernelPCA(**params)
        
        # Fit on train, transform both
        X_train_kpca = kpca.fit_transform(self.X_train)
        X_val_kpca = kpca.transform(self.X_val)
        
        return X_train_kpca, X_val_kpca, kpca
    
    def evaluate_with_simple_classifier(self, X_train_kpca, X_val_kpca) -> float:
        """Use simple SVM to evaluate KPCA quality"""
        clf = SVC(kernel='linear', C=1.0, random_state=42)
        clf.fit(X_train_kpca, self.y_train)
        val_acc = clf.score(X_val_kpca, self.y_val)
        return val_acc
    
    def grid_search(self, top_k: int = 3) -> List[Dict]:
        """
        Perform grid search over KPCA hyperparameters
        """
        search_space = self.create_search_space()
        
        # Generate valid combinations (kernel-specific params)
        configs = []
        for n_comp in search_space['n_components']:
            for kernel in search_space['kernel']:
                if kernel == 'rbf':
                    for gamma in search_space['gamma']:
                        configs.append({
                            'n_components': n_comp,
                            'kernel': kernel,
                            'gamma': gamma,
                            'degree': None,
                            'coef0': None
                        })
                elif kernel == 'poly':
                    for gamma in search_space['gamma']:
                        for degree in search_space['degree']:
                            for coef0 in search_space['coef0']:
                                configs.append({
                                    'n_components': n_comp,
                                    'kernel': kernel,
                                    'gamma': gamma,
                                    'degree': degree,
                                    'coef0': coef0
                                })
                elif kernel == 'sigmoid':
                    for gamma in search_space['gamma']:
                        for coef0 in search_space['coef0']:
                            configs.append({
                                'n_components': n_comp,
                                'kernel': kernel,
                                'gamma': gamma,
                                'degree': None,
                                'coef0': coef0
                            })
        
        print(f"\n🔍 KPCA Stage 2: Testing {len(configs)} configurations...")
        
        results = []
        
        for idx, config in enumerate(configs, 1):
            print(f"[{idx}/{len(configs)}] Testing: ", end="")
            print(f"kernel={config['kernel']}, n_comp={config['n_components']}, "
                  f"gamma={config['gamma']}")
            
            try:
                start = time.time()
                X_train_kpca, X_val_kpca, kpca = self.apply_kpca(config)
                val_acc = self.evaluate_with_simple_classifier(X_train_kpca, X_val_kpca)
                elapsed = time.time() - start
                
                results.append({
                    'config': config,
                    'val_accuracy': val_acc,
                    'X_train_kpca': X_train_kpca,
                    'X_val_kpca': X_val_kpca,
                    'kpca_model': kpca,
                    'time': elapsed
                })
                
                print(f"   ✓ Validation Accuracy: {val_acc:.4f} ({elapsed:.1f}s)\n")
                
            except Exception as e:
                print(f"   ✗ Failed: {e}\n")
                continue
        
        # Sort by accuracy
        results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_k} KPCA CONFIGURATIONS:")
        print(f"{'='*60}")
        for i, res in enumerate(results[:top_k], 1):
            print(f"\nRank {i}: Accuracy = {res['val_accuracy']:.4f}")
            print(f"Config: {json.dumps({k: v for k, v in res['config'].items() if v is not None}, indent=2)}")
        
        return results[:top_k]


class EnsembleHyperparameterTuner:
    """
    Stage 3: Tune ensemble classifier hyperparameters
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
    
    def create_search_space(self) -> Dict[str, List]:
        """Define ensemble hyperparameter search space"""
        return {
            # SVM parameters
            'svm_C': [0.1, 1.0, 10.0, 100.0],
            'svm_gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'svm_kernel': ['rbf', 'poly'],
            
            # XGBoost parameters
            'xgb_n_estimators': [50, 100, 200],
            'xgb_max_depth': [3, 5, 7, 10],
            'xgb_learning_rate': [0.01, 0.05, 0.1, 0.3],
            'xgb_subsample': [0.7, 0.8, 1.0],
            
            # Random Forest parameters
            'rf_n_estimators': [50, 100, 200, 300],
            'rf_max_depth': [10, 20, 30, None],
            'rf_min_samples_split': [2, 5, 10],
            
            # Meta-learner parameters
            'meta_C': [0.1, 1.0, 10.0],
            'meta_solver': ['lbfgs', 'saga'],
        }
    
    def build_ensemble(self, config: Dict) -> StackingClassifier:
        """Build stacking ensemble with given configuration"""
        
        # Base estimators
        estimators = [
            ('svm', SVC(
                C=config['svm_C'],
                gamma=config['svm_gamma'],
                kernel=config['svm_kernel'],
                probability=True,
                random_state=42
            )),
            ('xgb', XGBClassifier(
                n_estimators=config['xgb_n_estimators'],
                max_depth=config['xgb_max_depth'],
                learning_rate=config['xgb_learning_rate'],
                subsample=config['xgb_subsample'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=config['rf_n_estimators'],
                max_depth=config['rf_max_depth'],
                min_samples_split=config['rf_min_samples_split'],
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        # Meta-learner
        meta_clf = LogisticRegression(
            C=config['meta_C'],
            solver=config['meta_solver'],
            max_iter=1000,
            random_state=42
        )
        
        # Stacking ensemble
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_clf,
            cv=3,
            n_jobs=-1
        )
        
        return ensemble
    
    def train_and_evaluate(self, config: Dict) -> Tuple[float, StackingClassifier]:
        """Train ensemble and return validation accuracy"""
        ensemble = self.build_ensemble(config)
        ensemble.fit(self.X_train, self.y_train)
        val_acc = ensemble.score(self.X_val, self.y_val)
        return val_acc, ensemble
    
    def random_search(self, n_iterations: int = 20, top_k: int = 3) -> List[Dict]:
        """
        Perform random search over ensemble hyperparameters
        (Grid search would be too expensive here)
        """
        search_space = self.create_search_space()
        
        print(f"\n🔍 Ensemble Stage 3: Testing {n_iterations} random configurations...")
        
        results = []
        
        for idx in range(n_iterations):
            # Sample random configuration
            config = {key: np.random.choice(values) for key, values in search_space.items()}
            
            print(f"\n[{idx+1}/{n_iterations}] Testing ensemble config:")
            print(f"   SVM: C={config['svm_C']}, gamma={config['svm_gamma']}")
            print(f"   XGB: n_est={config['xgb_n_estimators']}, depth={config['xgb_max_depth']}, lr={config['xgb_learning_rate']}")
            print(f"   RF: n_est={config['rf_n_estimators']}, depth={config['rf_max_depth']}")
            
            try:
                start = time.time()
                val_acc, ensemble = self.train_and_evaluate(config)
                elapsed = time.time() - start
                
                results.append({
                    'config': config,
                    'val_accuracy': val_acc,
                    'ensemble': ensemble,
                    'time': elapsed
                })
                
                print(f"   ✓ Validation Accuracy: {val_acc:.4f} ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                continue
        
        # Sort by accuracy
        results.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_k} ENSEMBLE CONFIGURATIONS:")
        print(f"{'='*60}")
        for i, res in enumerate(results[:top_k], 1):
            print(f"\nRank {i}: Accuracy = {res['val_accuracy']:.4f}")
            print(f"Config:")
            for key, val in res['config'].items():
                print(f"  {key}: {val}")
        
        return results[:top_k]


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class StagedHyperparameterOptimizer:
    """
    Orchestrates the three-stage hyperparameter optimization process
    """
    
    def __init__(self, train_loader, val_loader, device, num_classes=10):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        self.results = {
            'stage1_cnn': None,
            'stage2_kpca': None,
            'stage3_ensemble': None
        }
    
    def run_stage1_cnn(self, num_epochs=10, top_k=3):
        """Stage 1: Optimize CNN"""
        print("\n" + "="*60)
        print("STAGE 1: CNN HYPERPARAMETER TUNING")
        print("="*60)
        
        tuner = CNNHyperparameterTuner(
            self.train_loader, self.val_loader, self.device, self.num_classes
        )
        self.results['stage1_cnn'] = tuner.grid_search(num_epochs, top_k)
        
        return self.results['stage1_cnn'][0]  # Return best config
    
    def run_stage2_kpca(self, best_cnn_model, top_k=3):
        """Stage 2: Optimize KPCA using best CNN"""
        print("\n" + "="*60)
        print("STAGE 2: KPCA HYPERPARAMETER TUNING")
        print("="*60)
        
        # Extract features from best CNN
        print("Extracting features from best CNN model...")
        X_train, y_train = self._extract_features(best_cnn_model, self.train_loader)
        X_val, y_val = self._extract_features(best_cnn_model, self.val_loader)
        
        tuner = KPCAHyperparameterTuner(X_train, y_train, X_val, y_val)
        self.results['stage2_kpca'] = tuner.grid_search(top_k)
        
        return self.results['stage2_kpca'][0]  # Return best config
    
    def run_stage3_ensemble(self, best_kpca_data, n_iterations=20, top_k=3):
        """Stage 3: Optimize Ensemble using best KPCA"""
        print("\n" + "="*60)
        print("STAGE 3: ENSEMBLE HYPERPARAMETER TUNING")
        print("="*60)
        
        X_train = best_kpca_data['X_train_kpca']
        y_train = self.results['stage2_kpca'][0]['config']  # Need to store y_train
        X_val = best_kpca_data['X_val_kpca']
        y_val = best_kpca_data['val_accuracy']  # Need to store y_val
        
        # We need actual labels, let's extract them
        _, y_train = self._extract_features(
            self.results['stage1_cnn'][0]['model'], 
            self.train_loader
        )
        _, y_val = self._extract_features(
            self.results['stage1_cnn'][0]['model'],
            self.val_loader
        )
        
        tuner = EnsembleHyperparameterTuner(X_train, y_train, X_val, y_val)
        self.results['stage3_ensemble'] = tuner.random_search(n_iterations, top_k)
        
        return self.results['stage3_ensemble'][0]  # Return best config
    
    def _extract_features(self, model, loader):
        """Extract features from CNN"""
        model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for images, lbls in loader:
                images = images.to(self.device)
                feats = model.get_features(images)
                features.append(feats.cpu().numpy())
                labels.append(lbls.numpy())
        
        return np.concatenate(features), np.concatenate(labels)
    
    def run_full_pipeline(self, cnn_epochs=10, kpca_top_k=3, ensemble_iters=20):
        """Run all three stages sequentially"""
        
        print("\n" + "🚀"*30)
        print("STARTING FULL STAGED HYPERPARAMETER OPTIMIZATION")
        print("🚀"*30)
        
        # Stage 1: CNN
        best_cnn = self.run_stage1_cnn(num_epochs=cnn_epochs, top_k=3)
        print(f"\n✅ Stage 1 Complete. Best CNN Val Acc: {best_cnn['val_accuracy']:.4f}")
        
        # Stage 2: KPCA
        best_kpca = self.run_stage2_kpca(best_cnn['model'], top_k=kpca_top_k)
        print(f"\n✅ Stage 2 Complete. Best KPCA Val Acc: {best_kpca['val_accuracy']:.4f}")
        
        # Stage 3: Ensemble
        best_ensemble = self.run_stage3_ensemble(best_kpca, n_iterations=ensemble_iters, top_k=3)
        print(f"\n✅ Stage 3 Complete. Best Ensemble Val Acc: {best_ensemble['val_accuracy']:.4f}")
        
        # Final Summary
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION COMPLETE - FINAL RESULTS")
        print("="*60)
        print(f"\nBest CNN Config:")
        print(json.dumps(best_cnn['config'], indent=2))
        print(f"Validation Accuracy: {best_cnn['val_accuracy']:.4f}")
        
        print(f"\nBest KPCA Config:")
        print(json.dumps({k: v for k, v in best_kpca['config'].items() if v is not None}, indent=2))
        print(f"Validation Accuracy: {best_kpca['val_accuracy']:.4f}")
        
        print(f"\nBest Ensemble Config:")
        print("SVM:", {k: v for k, v in best_ensemble['config'].items() if k.startswith('svm')})
        print("XGB:", {k: v for k, v in best_ensemble['config'].items() if k.startswith('xgb')})
        print("RF:", {k: v for k, v in best_ensemble['config'].items() if k.startswith('rf')})
        print(f"Validation Accuracy: {best_ensemble['val_accuracy']:.4f}")
        
        return {
            'best_cnn': best_cnn,
            'best_kpca': best_kpca,
            'best_ensemble': best_ensemble
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# In your notebook:

from hyperparameter_tuning import StagedHyperparameterOptimizer

# Initialize optimizer
optimizer = StagedHyperparameterOptimizer(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_classes=NUM_CLASSES
)

# Run full pipeline
results = optimizer.run_full_pipeline(
    cnn_epochs=10,      # Epochs per CNN config
    kpca_top_k=3,       # Top K KPCA configs to keep
    ensemble_iters=20   # Random search iterations for ensemble
)

# Or run stages individually:

# Stage 1 only
best_cnn = optimizer.run_stage1_cnn(num_epochs=15, top_k=5)

# Stage 2 (requires Stage 1)
best_kpca = optimizer.run_stage2_kpca(best_cnn['model'], top_k=3)

# Stage 3 (requires Stage 2)
best_ensemble = optimizer.run_stage3_ensemble(best_kpca, n_iterations=30, top_k=3)
"""