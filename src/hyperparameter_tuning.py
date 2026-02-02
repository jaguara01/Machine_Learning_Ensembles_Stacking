"""
Hyperparameter Tuning for Robust Font Classification
Performs grid search over CNN, kPCA, and ensemble parameters
"""

import torch
import numpy as np
from itertools import product
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.models import FontCNN
from src.kernel_methods import apply_kpca
from src.data_loader import get_dataloaders


class HyperparameterTuner:
    """
    Systematic tuning of all pipeline components:
    1. CNN architecture parameters
    2. Kernel PCA parameters  
    3. Base model parameters (for ensemble)
    """
    
    def __init__(self, data_path, device, results_dir="./tuning_results"):
        self.data_path = data_path
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load data once
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader, \
            self.num_classes, self.img_dim = get_dataloaders(
                data_path=data_path,
                batch_size=64,
                include_other=False
            )
        
        # Storage for results
        self.results = []
        
    def define_search_space(self):
        """
        Define hyperparameter search spaces for each component
        """
        search_space = {
            # CNN Architecture Hyperparameters
            'cnn_base_filters': [32, 64],  # Width of conv layers
            'cnn_latent_dim': [128, 256],  # Size of embedding space
            'cnn_dropout': [0.3, 0.5],     # Regularization strength
            'cnn_learning_rate': [0.001, 0.0005],
            
            # Kernel PCA Hyperparameters
            'kpca_n_components': [50, 100, 150],  # Dimensionality reduction target
            'kpca_gamma': [None, 0.1, 0.01, 0.001],  # RBF kernel bandwidth
            
            # Training Hyperparameters
            'batch_size': [64],  # Fixed for consistency
            'num_epochs': [15]   # Fixed for time constraints
        }
        
        return search_space
    
    def extract_features(self, model, loader):
        """
        Extract features from fc1 layer of trained CNN
        """
        model.eval()
        all_features = []
        all_labels = []
        hook_data = {}
        
        def get_fc1_output(m, input, output):
            hook_data["features"] = output.detach()
        
        handle = model.fc1.register_forward_hook(get_fc1_output)
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                _ = model(images)
                all_features.append(hook_data["features"].cpu())
                all_labels.append(labels)
        
        handle.remove()
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    
    def train_cnn_with_config(self, config):
        """
        Train CNN with specific hyperparameter configuration
        Returns trained model and validation accuracy
        """
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        import torch.nn as nn
        
        # Reload data with specified batch size
        train_loader, val_loader, _, _, _ = get_dataloaders(
            data_path=self.data_path,
            batch_size=config['batch_size'],
            include_other=False
        )
        
        # Initialize model with config
        model = FontCNN(
            num_classes=self.num_classes,
            input_dim=self.img_dim,
            base_filters=config['cnn_base_filters'],
            latent_dim=config['cnn_latent_dim'],
            dropout=config['cnn_dropout']
        ).to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['cnn_learning_rate'])
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(config['num_epochs']):
            # Train
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits, _ = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits, _ = model(images)
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
        
        return model, best_val_acc
    
    def evaluate_kpca_config(self, X_train, y_train, X_val, y_val, config):
        """
        Apply kPCA with config and return projection quality score
        """
        X_train_kpca, X_val_kpca, kpca_model = apply_kpca(
            X_train, y_train,
            X_val, y_val,
            n_components=config['kpca_n_components'],
            kernel='rbf',
            gamma=config['kpca_gamma'],
            max_fit_samples=5000
        )
        
        # Quality is measured by linear probe accuracy (printed in apply_kpca)
        # For scoring, we'll use a simple SVM
        from sklearn.svm import SVC
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train_kpca.numpy(), y_train.numpy())
        kpca_score = clf.score(X_val_kpca.numpy(), y_val.numpy())
        
        return X_train_kpca, X_val_kpca, kpca_score
    
    def run_grid_search(self, sample_configs=None):
        """
        Perform grid search over hyperparameter space
        
        Args:
            sample_configs: If int, randomly sample N configurations
                          If None, search entire space (can be large!)
        """
        search_space = self.define_search_space()
        
        # Generate all combinations
        keys = search_space.keys()
        values = search_space.values()
        all_configs = [dict(zip(keys, v)) for v in product(*values)]
        
        print(f"Total configurations to evaluate: {len(all_configs)}")
        
        # Sample if requested
        if sample_configs and sample_configs < len(all_configs):
            import random
            random.seed(42)
            all_configs = random.sample(all_configs, sample_configs)
            print(f"Sampling {sample_configs} configurations...")
        
        # Evaluate each configuration
        for idx, config in enumerate(tqdm(all_configs, desc="Grid Search")):
            print(f"\n{'='*60}")
            print(f"Configuration {idx+1}/{len(all_configs)}")
            print(f"{'='*60}")
            print(json.dumps(config, indent=2))
            
            try:
                result = self.evaluate_configuration(config)
                result['config_id'] = idx
                result['config'] = config
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                print(f"ERROR in config {idx}: {e}")
                continue
        
        return self.get_best_config()
    
    def evaluate_configuration(self, config):
        """
        Full pipeline evaluation for a single configuration
        """
        # Phase 1: Train CNN
        print("\n[Phase 1] Training CNN...")
        model, cnn_val_acc = self.train_cnn_with_config(config)
        
        # Phase 2: Extract features
        print("\n[Phase 2] Extracting features...")
        X_train, y_train = self.extract_features(model, self.train_loader)
        X_val, y_val = self.extract_features(model, self.val_loader)
        
        # Phase 3: Apply kPCA
        print("\n[Phase 3] Applying Kernel PCA...")
        X_train_kpca, X_val_kpca, kpca_score = self.evaluate_kpca_config(
            X_train, y_train, X_val, y_val, config
        )
        
        # Phase 4: Quick ensemble score (simplified for speed)
        print("\n[Phase 4] Quick ensemble evaluation...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Use a fast tree model for scoring
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train_kpca.numpy(), y_train.numpy())
        ensemble_score = rf.score(X_val_kpca.numpy(), y_val.numpy())
        
        result = {
            'cnn_val_acc': cnn_val_acc,
            'kpca_projection_quality': kpca_score,
            'ensemble_val_acc': ensemble_score,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n📊 Results:")
        print(f"  CNN Val Acc: {cnn_val_acc:.4f}")
        print(f"  kPCA Quality: {kpca_score:.4f}")
        print(f"  Ensemble Acc: {ensemble_score:.4f}")
        
        return result
    
    def save_results(self):
        """Save results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_path = self.results_dir / f"tuning_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        df = pd.DataFrame([
            {
                **r['config'],
                'cnn_val_acc': r['cnn_val_acc'],
                'kpca_quality': r['kpca_projection_quality'],
                'ensemble_acc': r['ensemble_val_acc']
            }
            for r in self.results
        ])
        csv_path = self.results_dir / f"tuning_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n💾 Results saved to: {self.results_dir}")
    
    def get_best_config(self):
        """Return configuration with best ensemble accuracy"""
        if not self.results:
            return None
        
        best = max(self.results, key=lambda x: x['ensemble_val_acc'])
        
        print("\n" + "="*60)
        print("🏆 BEST CONFIGURATION")
        print("="*60)
        print(json.dumps(best['config'], indent=2))
        print(f"\nEnsemble Validation Accuracy: {best['ensemble_val_acc']:.4f}")
        print(f"CNN Validation Accuracy: {best['cnn_val_acc']:.4f}")
        print(f"kPCA Projection Quality: {best['kpca_projection_quality']:.4f}")
        
        return best


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                       default='./data/processed_fonts.pt')
    parser.add_argument('--sample_configs', type=int, default=10,
                       help='Number of random configs to sample (None for full grid)')
    parser.add_argument('--results_dir', type=str, default='./tuning_results')
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        data_path=args.data_path,
        device=device,
        results_dir=args.results_dir
    )
    
    # Run grid search
    best_config = tuner.run_grid_search(sample_configs=args.sample_configs)
    
    print("\n✅ Hyperparameter tuning complete!")
    print(f"📁 Results saved in: {args.results_dir}")