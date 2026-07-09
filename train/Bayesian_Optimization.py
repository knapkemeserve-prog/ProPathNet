"""
==============================================================================
File: Bayesian_Optimization.py
Description: 
    Hyperparameter tuning script for ProPathNet using Bayesian Optimization.
    This script utilizes 'skopt' (scikit-optimize) to intelligently search the
    hyperparameter space (hidden dimensions, layers, learning rate, etc.) 
    to maximize the Validation C-index. It evaluates multiple configurations,
    employs early stopping, and automatically tracks the Top-3 best models.
==============================================================================
"""

import os
import sys
import time
import json
import pickle
import random
import argparse
import traceback
import gc

import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index

# Import libraries for Bayesian optimization
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import DeadlineStopper
from skopt import dump

# --- Add project root directory to the system path to locate custom modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import our custom refactored modules ---
from preprocess.data_loader import preprocess_data
from models.propathnet import ProPathNet
from train.loss import NegativeLogLikelihood


# ==============================================================================
# 1. General Utility Functions
# ==============================================================================
def set_seed(seed=42):
    """
    Fix all global random seeds to ensure 100% reproducibility across experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_memory():
    """
    Clean up GPU and CPU memory. Crucial to prevent Out-Of-Memory (OOM) 
    errors during intensive iterative hyperparameter tuning.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def print_gpu_memory_usage(device):
    """Print current GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"[*] GPU Memory: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")

def c_index(risk_pred, y, e):
    """
    Utility function to calculate the Concordance index (C-index).
    Handles tensor-to-numpy conversion and NaN imputation safely.
    """
    if not isinstance(y, np.ndarray): y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray): risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray): e = e.detach().cpu().numpy()
    
    if np.isnan(risk_pred).any(): risk_pred = np.nan_to_num(risk_pred, nan=np.nanmedian(risk_pred))
    if np.isnan(y).any(): y = np.nan_to_num(y, nan=np.nanmedian(y))
    if np.isnan(e).any(): e = np.nan_to_num(e, nan=0)
    return concordance_index(y, risk_pred, e)

def convert_numpy_types(obj):
    """
    Recursively convert NumPy data types to standard Python types.
    Required because the 'json' module cannot natively serialize NumPy types.
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
    elif isinstance(obj, (np.ndarray,)): return obj.tolist()
    elif isinstance(obj, dict): return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, tuple): return tuple(convert_numpy_types(element) for element in obj)
    else: return obj


# ==============================================================================
# 2. Single Configuration Training Engine (Tuning Iteration)
# ==============================================================================
def train_single_config(train_data, val_data, test_data, params, gene_names, clinical_dim, model_path, biovnn_dict, device):
    """
    Executes a complete training pipeline for a single set of hyperparameters.
    Implements early stopping and evaluates on the test set only using the best validation checkpoint.
    """
    model, vnn_model, optimizer, criterion = None, None, None, None
    try:
        batch_size = params['batch_size']
        
        # Initialize DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
        train_eval_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

        num_genes = len(gene_names)
        num_nodes = train_data[0].x.shape[0]

        # Note: Instantiate the fully integrated ProPathNet architecture
        model = ProPathNet(
            num_nodes=num_nodes, num_genes=num_genes, hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'], biovnn_dict=biovnn_dict,
            config=params, clinical_dim=clinical_dim
        ).to(device)

        # Initialize optimizer and Cox Loss (with dynamically tuned L2 regularization)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = NegativeLogLikelihood({'l2_reg': params['l2_reg']})

        best_val_cindex, best_test_cindex, best_train_cindex = 0.0, 0.0, 0.0
        best_epoch, no_improvement_epochs = 0, 0

        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0.0
            
            # Training loop
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                
                batch_size_current = torch.max(batch.batch).item() + 1
                gene_expression = batch.x.view(batch_size_current, num_genes).to(device, non_blocking=True)
                clinical_features = batch.clinical.to(device, non_blocking=True)
                
                # Forward pass
                risk_pred = model(
                    x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                    gene_expression=gene_expression, clinical=clinical_features
                )
                
                # Compute loss, backpropagate, and update weights
                loss = criterion(risk_pred, batch.y, batch.e, model)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs

            # Validation Evaluation (done efficiently at the end of each epoch)
            model.eval()
            val_risk_scores, val_times, val_events = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    batch_size_current = torch.max(batch.batch).item() + 1
                    gene_expression = batch.x.view(batch_size_current, num_genes).to(device, non_blocking=True)
                    
                    risk_pred = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, gene_expression=gene_expression, clinical=batch.clinical.to(device))
                    
                    val_risk_scores.append(risk_pred.cpu())
                    val_times.append(batch.y.cpu())
                    val_events.append(batch.e.cpu())
                    
            val_risk_scores = torch.cat(val_risk_scores).squeeze()
            # Negate risk scores for Concordance Index computation
            val_cindex = c_index(risk_pred=-val_risk_scores, y=torch.cat(val_times), e=torch.cat(val_events))

            # Record best validation model and reset early stopping counter
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_epoch = epoch + 1
                no_improvement_epochs = 0
                # Temporarily save the best weights for this specific tuning iteration
                torch.save(model.state_dict(), f'{model_path}.pth')
            else:
                no_improvement_epochs += 1
                
            # Trigger Early Stopping
            if no_improvement_epochs >= params['patience']:
                break

        # Post-training: Load the best epoch's model to calculate the final Test C-index
        model.load_state_dict(torch.load(f'{model_path}.pth'))
        model.eval()
        test_risk_scores, test_times, test_events = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device, non_blocking=True)
                batch_size_current = torch.max(batch.batch).item() + 1
                
                risk_pred = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, gene_expression=batch.x.view(batch_size_current, num_genes).to(device), clinical=batch.clinical.to(device))
                
                test_risk_scores.append(risk_pred.cpu())
                test_times.append(batch.y.cpu())
                test_events.append(batch.e.cpu())
        
        best_test_cindex = c_index(risk_pred=-torch.cat(test_risk_scores).squeeze(), y=torch.cat(test_times), e=torch.cat(test_events))

        return {'best_val_cindex': best_val_cindex, 'best_test_cindex': best_test_cindex, 'best_epoch': best_epoch}

    except Exception as e:
        print(f"[x] Training exception occurred: {e}")
        return {'best_val_cindex': 0.0, 'best_test_cindex': 0.0, 'best_epoch': 0}
    finally:
        # Guarantee memory release regardless of errors
        del model, optimizer, criterion
        cleanup_memory()


# ==============================================================================
# 3. Bayesian Optimization Objective Wrapper
# ==============================================================================
def objective_function(params, train_data, val_data, test_data, gene_names, clinical_dim, model_path_base, results, top_results, iteration_count, biovnn_dict, device):
    """
    Objective function called by `skopt`. Wraps the training engine, enforces types, 
    tracks the Top-3 best configurations dynamically, and logs results to JSON.
    """
    # Enforce integer types for categorical/discrete hyperparameters
    int_params = ['hidden_dim', 'batch_size', 'neuron_min', 'fusion_heads', 'gat_heads', 'num_layers']
    for param in int_params:
        if param in params: params[param] = int(params[param])

    # Fixed parameters for the rapid search phase
    params['fusion_type'] = 'bottleneck'
    params['batch_size'] = 512
    params['epochs'] = 5        # Keep epochs low for fast exploratory search
    params['patience'] = 2      # Strict early stopping
    
    # Propagate the unified 'dropout' parameter to specific sub-modules
    params['vnn_dropout_p'] = params.get('dropout', 0.5) 
    params['fusion_dropout_prob'] = params.get('dropout', 0.5)
    params['gat_dropout'] = params.get('dropout', 0.5)
    
    print(f"\n{'='*40}")
    print(f"[*] Running Tuning Iteration #{iteration_count}")
    print(f"{'='*40}")
    
    iteration_model_path = f"{model_path_base}_bayes_{iteration_count}"
    result = train_single_config(train_data, val_data, test_data, params, gene_names, clinical_dim, iteration_model_path, biovnn_dict, device)
    
    current_result = {
        'params': params,
        'val_cindex': result['best_val_cindex'],
        'test_cindex': result['best_test_cindex'],
        'epochs': result['best_epoch'],
        'model_path': f"{iteration_model_path}.pth",
        'iteration': iteration_count
    }
    results.append(current_result)
    
    # Dynamically update the Top-3 performing configurations list
    if len(top_results) < 3:
        top_results.append(current_result)
        top_results.sort(key=lambda x: x['val_cindex'], reverse=True)
    else:
        min_val_cindex = min([r['val_cindex'] for r in top_results])
        if result['best_val_cindex'] > min_val_cindex:
            for i, r in enumerate(top_results):
                if r['val_cindex'] == min_val_cindex:
                    top_results[i] = current_result
                    break
            top_results.sort(key=lambda x: x['val_cindex'], reverse=True)

    print(f"[+] Result -> Val C-index: {result['best_val_cindex']:.4f} | Test C-index: {result['best_test_cindex']:.4f}")
    
    # Persist intermediate results to disk safely
    with open(f'{model_path_base}_bayesian_results.json', 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=4)
        
    # Return negative Validation C-index (skopt inherently minimizes the objective)
    return -result['best_val_cindex']


# ==============================================================================
# 4. Main Execution Pipeline
# ==============================================================================
def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='ProPathNet Bayesian Hyperparameter Optimization')
    parser.add_argument('--adata_path', type=str, required=True, help="Path to input .h5ad dataset")
    parser.add_argument('--ppi_path', type=str, required=True, help="Path to input PPI network")
    parser.add_argument('--clinical_path', type=str, required=True, help="Path to clinical features CSV")
    parser.add_argument('--biovnn_path', type=str, required=True, help="Path to BioVNN dictionary .pkl")
    parser.add_argument('--model_save_path', type=str, required=True, help="Base path for saving model checkpoints")
    parser.add_argument('--n_calls', type=int, default=30, help="Total number of Bayesian optimization iterations")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'[*] Initializing training environment on device: {device}')

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    model_path_base = os.path.splitext(args.model_save_path)[0]

    print("\n>>> Loading and preprocessing datasets...")
    # NOTE: preprocess_data parameters must match the signature defined in your data_loader module
    train_data, val_data, test_data, _, gene_names, clinical_dim = preprocess_data(
        args.adata_path, args.ppi_path, args.clinical_path
    )
    
    with open(args.biovnn_path, "rb") as f:
        biovnn_dict = pickle.load(f)

    # Define the Bayesian hyperparameter search space boundary
    param_space = [
        Categorical([32, 64, 128], name='hidden_dim'),
        Categorical([4, 6, 8], name='num_layers'),
        Categorical([4, 8], name='gat_heads'),
        Real(0.4, 0.6, name='dropout'),
        Categorical([16, 32, 64], name='neuron_min'),
        Real(0.2, 0.6, name='neuron_ratio'),
        Categorical([4, 6, 8], name='fusion_heads'),
        Real(1e-5, 1e-3, prior='log-uniform', name='lr'),              # Logarithmic scale for Learning Rate
        Real(1e-5, 1e-4, prior='log-uniform', name='weight_decay'),    # Logarithmic scale for Optimizer L2
        Real(1e-4, 1e-2, prior='log-uniform', name='l2_reg'),          # Logarithmic scale for Cox Loss L2
    ]

    results = []
    top_results = []
    iteration_count = [0]

    # Wrapper to unpack Named Args generated by skopt into our dictionary
    @use_named_args(param_space)
    def objective_wrapper(**params):
        iteration_count[0] += 1
        return objective_function(params, train_data, val_data, test_data, gene_names, clinical_dim, model_path_base, results, top_results, iteration_count[0], biovnn_dict, device)

    print(f"\n>>> Commencing Bayesian Optimization (Total predefined iterations: {args.n_calls})")
    
    # Execute Gaussian Process minimization
    res = gp_minimize(
        objective_wrapper, param_space, n_calls=args.n_calls, n_initial_points=15, 
        random_state=42, verbose=True, callback=[DeadlineStopper(60*60*48)] # Hard stop at 48 hours
    )

    print("\n" + "="*50)
    print("🎉 Optimization Complete! Best Discovered Hyperparameter Combination:")
    print("="*50)
    best_params = {}
    for i, space in enumerate(param_space):
        best_params[space.name] = res.x[i]
        
    for k, v in best_params.items():
        print(f"  - {k}: {v}")
    print("="*50)

if __name__ == "__main__":
    main()

"""
==============================================================================
Example Terminal Execution Command (Run in background):
==============================================================================
nohup python -u olink_disease_predict/ProPathNet-github/train/Bayesian_Optimization.py \
    --adata_path data/sample_data.h5ad \
    --clinical_path data/sample_lifestyle_data.csv \
    --ppi_path data/human_ppi_intact.csv \
    --biovnn_path data/biovnn_dict.pkl \
    --model_save_path checkpoints/best_test_model.pth \
    --n_calls 30 \
    > log/tune_bayes.log 2>&1 &
"""