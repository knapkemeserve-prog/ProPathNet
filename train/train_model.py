"""
==============================================================================
File: train_model.py
Description: 
    Base training script for the ProPathNet model.
    This script is responsible for:
    1. Loading and preprocessing multi-omics and clinical data.
    2. Initializing the ProPathNet architecture (GAT + BioVNN + Clinical FNN).
    3. Training the model using Cox Negative Log Partial Likelihood Loss.
    4. Implementing Early Stopping based on validation Concordance Index (C-index).
    5. Evaluating the best model on a hold-out test set and saving the checkpoint.
==============================================================================
"""

import os
import sys
import pickle
import argparse
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index

# Append the project root directory to the environment path to locate 'models' and 'preprocess' modules.
# Since this file is located in the train/ directory, we need to move one level up.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.propathnet import ProPathNet
from preprocess.data_loader import preprocess_data

# Configure computing device (Use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def c_index(risk_pred, y, e):
    """
    Utility function to calculate the Concordance index (C-index).
    
    Args:
        risk_pred: Predicted risk/hazard scores.
        y: Actual survival times.
        e: Event indicators (1 if event occurred, 0 if censored).
        
    Returns:
        float: Computed Concordance index.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()

    # Handle potential NaN values to prevent lifelines module from crashing
    if np.isnan(risk_pred).any():
        risk_pred = np.nan_to_num(risk_pred, nan=np.nanmedian(risk_pred))
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=np.nanmedian(y))
    if np.isnan(e).any():
        e = np.nan_to_num(e, nan=0)

    # Note: risk_pred in Cox models represents hazard (risk score).
    # Higher score = higher risk = shorter survival time.
    # The lifelines package expects the risk score to be negated for proper calculation.
    return concordance_index(y, -risk_pred, e)

def cox_loss(surv_time, censor, hazard_pred):
    """
    Cox Negative Log Partial Likelihood Loss Function.
    
    Args:
        surv_time (torch.Tensor): Survival times for the batch.
        censor (torch.Tensor): Event indicators (1 for event, 0 for censored).
        hazard_pred (torch.Tensor): Predicted hazard/risk scores from the model.
        
    Returns:
        torch.Tensor: Computed scalar loss value.
    """
    # Flatten tensors to ensure 1D shape
    hazard_pred = hazard_pred.view(-1)
    surv_time = surv_time.view(-1)
    censor = censor.view(-1)
    
    # Sort patients by survival time in descending order
    _, indices = torch.sort(surv_time, descending=True)
    censor = censor[indices]
    hazard_pred = hazard_pred[indices]

    # Check if there are any actual events in the current batch
    events = censor == 1
    if events.sum() == 0:
        return torch.tensor(0.0, requires_grad=True).to(hazard_pred.device)

    # Numerical stability trick (Log-Sum-Exp)
    # Subtract the maximum hazard prediction to prevent exponential overflow
    hazard_pred = hazard_pred - hazard_pred.max()
    exp_pred = torch.exp(hazard_pred)
    cumsum_exp_pred = torch.cumsum(exp_pred, dim=0)

    # Calculate log risks and uncensored likelihood
    log_risk = torch.log(cumsum_exp_pred + 1e-8)
    uncensored_likelihood = hazard_pred - log_risk
    
    # Final Cox loss: Average over all occurred events
    loss = -torch.sum(uncensored_likelihood[events]) / events.sum()
    
    return loss

def evaluate(model, dataloader, gene_names):
    """
    Evaluation function for Validation and Test sets.
    
    Args:
        model: The trained ProPathNet model.
        dataloader: PyTorch Geometric DataLoader containing validation/test data.
        gene_names: List of gene/protein names to format expressions properly.
        
    Returns:
        float: Computed C-index for the dataset.
    """
    model.eval()
    all_risk_scores = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            batch_size = torch.max(batch.batch).item() + 1
            
            # Reshape gene expression for the VNN
            gene_expression = batch.x.view(batch_size, len(gene_names)).to(device, non_blocking=True)
            
            # Forward pass
            risk_pred = model(
                x=batch.x, 
                edge_index=batch.edge_index, 
                batch=batch.batch,
                gene_expression=gene_expression, 
                clinical=batch.clinical
            )
            
            all_risk_scores.append(risk_pred.cpu().numpy().squeeze())
            all_times.append(batch.y.cpu().numpy())
            all_events.append(batch.e.cpu().numpy())
            
    all_risk_scores = np.concatenate(all_risk_scores)
    all_times = np.concatenate(all_times)
    all_events = np.concatenate(all_events)
    
    return c_index(all_risk_scores, all_times, all_events)

def main():
    parser = argparse.ArgumentParser(description='ProPathNet Base Training Script')
    
    # Required arguments provided by the user
    parser.add_argument('--adata_path', type=str, required=True, help='Path to the input .h5ad dataset')
    parser.add_argument('--clinical_path', type=str, required=True, help='Path to the clinical features CSV')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the best model weights (e.g., .pth)')
    
    # Optional supplementary arguments (with defaults to ensure standalone usability)
    parser.add_argument('--disease', type=str, default='General', help='Disease name (used for single-sex feature filtering)')
    parser.add_argument('--ppi_path', type=str, default='data/human_ppi_intact.csv', help='Path to the PPI network file')
    parser.add_argument('--biovnn_path', type=str, default='data/biovnn_dict.pkl', help='Path to the BioVNN dictionary')
    
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"🚀 Starting ProPathNet Model Training Pipeline")
    print(f"{'='*50}")

    # ==========================================
    # 🌟 Preset Hyperparameter Configuration
    # ==========================================
    config = {
        'epochs': 100,           # Fixed total number of training epochs
        'batch_size': 128,       # Fixed Batch Size
        'hidden_dim': 64,        # Selected medium dimension from [32, 64, 128]
        'num_layers': 4,         # Selected 4 layers from [4, 6, 8]
        'gat_heads': 4,          # Selected 4 attention heads from [4, 8]
        'dropout': 0.5,          # Midpoint dropout between 0.4 - 0.6
        'neuron_min': 32,        # Minimum neurons in VNN layer
        'neuron_ratio': 0.4,     # Scaling ratio for VNN hierarchical nodes
        'fusion_heads': 4,       # Heads for the multimodal fusion attention mechanism
        'lr': 1e-4,              # Moderate learning rate (log uniform 1e-5 to 1e-3)
        'weight_decay': 5e-5,    # L2 penalty for Adam optimizer
        'l2_reg': 1e-3,          # Explicit L2 regularization for Cox loss
        'fusion_type': 'bottleneck', # Multi-omics fusion architecture type
        'patience': 20           # 🌟 Early stopping patience threshold
    }

    print(f"[*] Loaded internal default Config hyperparameters:")
    for k, v in config.items():
        print(f"    - {k}: {v}")

    # ==========================================
    # 1. Data Loading and Preprocessing
    # ==========================================
    print(f"\n[*] Initializing Data Splitting and Preprocessing (Train/Val/Test)...")
    train_data_list, val_data_list, test_data_list, _, gene_names, clinical_dim = preprocess_data(
        adata_path=args.adata_path, 
        ppi_path=args.ppi_path, 
        clinical_path=args.clinical_path,
        disease_name=args.disease
    )
    
    print(f"    - Training samples: {len(train_data_list)}")
    print(f"    - Validation samples: {len(val_data_list)}")
    print(f"    - Test samples: {len(test_data_list)}")

    # Initialize PyTorch Geometric DataLoaders
    train_loader = DataLoader(train_data_list, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data_list, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data_list, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    print(f"\n[*] Initializing ProPathNet Architecture...")
    with open(args.biovnn_path, "rb") as f:
        biovnn_dict = pickle.load(f)

    # Instantiate the main network structure
    model = ProPathNet(
        num_nodes=train_data_list[0].x.shape[0], 
        num_genes=len(gene_names), 
        hidden_dim=config['hidden_dim'], 
        num_layers=config['num_layers'], 
        biovnn_dict=biovnn_dict, 
        config=config, 
        clinical_dim=clinical_dim
    ).to(device)

    # Configure the Adam optimizer using defined learning rate and weight decay
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # ==========================================
    # 3. Training Loop with Early Stopping
    # ==========================================
    print(f"\n[*] Entering Training Loop (Epochs: {config['epochs']}, LR: {config['lr']})...")
    best_val_cindex = -1.0
    epochs_no_improve = 0  # 🌟 Counter for Early Stopping
    
    # Ensure the target directory for saving model weights exists
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            batch_size = torch.max(batch.batch).item() + 1
            
            # Reshape input expression matrix
            gene_expression = batch.x.view(batch_size, len(gene_names)).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            risk_pred = model(
                x=batch.x, 
                edge_index=batch.edge_index, 
                batch=batch.batch,
                gene_expression=gene_expression, 
                clinical=batch.clinical
            )
            
            # Calculate Cox Loss and Backpropagate
            loss = cox_loss(batch.y, batch.e, risk_pred)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            
        avg_loss = total_loss / len(train_data_list)
        
        # Evaluate on the validation set at the end of every epoch
        val_cindex = evaluate(model, val_loader, gene_names)
        
        # Print epoch metrics
        print(f"Epoch [{epoch:03d}/{config['epochs']:03d}] | Loss: {avg_loss:.4f} | Val C-index: {val_cindex:.4f}")
        
        # 🌟 Early Stopping & Model Saving Logic
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            epochs_no_improve = 0  # 🌟 Reset counter if validation metric improves
            
            # Save the complete checkpoint state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_cindex': best_val_cindex
            }
            torch.save(checkpoint, args.model_save_path)
            print(f"  👉 [Checkpoint Saved] Validation C-index improved to {best_val_cindex:.4f}")
        else:
            epochs_no_improve += 1  # 🌟 Increment counter if no improvement
            print(f"  ⚠️ Validation C-index did not improve ({epochs_no_improve}/{config['patience']})")
            
        # 🌟 Trigger Early Stopping
        if epochs_no_improve >= config['patience']:
            print(f"\n⏹️ Early stopping triggered! Validation C-index has not improved for {config['patience']} consecutive epochs.")
            break

    # ==========================================
    # 4. Final Evaluation on Hold-out Test Set
    # ==========================================
    print(f"\n{'='*50}")
    print(f"[*] Training finished. Evaluating the best model checkpoint on the Test set...")
    
    # Reload the best weights saved during the training loop
    checkpoint = torch.load(args.model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate test performance
    test_cindex = evaluate(model, test_loader, gene_names)
    
    print(f"\n🎉 [Final Results] Best Val C-index: {best_val_cindex:.4f} | Final Test C-index: {test_cindex:.4f}")
    print(f"💾 Optimal model checkpoint safely stored at: {args.model_save_path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()