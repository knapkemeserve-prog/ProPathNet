"""
==============================================================================
File: disease_prediction.py
Description: 
    Automated Inference and Prediction Script for ProPathNet.
    This script is designed for production/inference use. It handles:
    1. Automated routing to the correct pre-trained model via DiseaseModelManager.
    2. Strict inference data preprocessing (no shuffling, exact patient ID mapping).
    3. Legacy checkpoint compatibility (mapping old model weights to the new architecture).
    4. Forward pass risk score computation and C-index evaluation.
    5. Exporting individual patient risk predictions to a CSV file.
==============================================================================
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

# Append the project root directory to the system path to locate the models module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.propathnet import ProPathNet
from model_manager import DiseaseModelManager

# Configure device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def c_index(risk_pred, y, e):
    """
    Utility function to calculate the Concordance index (C-index).
    
    Args:
        risk_pred: Predicted risk scores.
        y: Actual survival times.
        e: Event indicators (1 for event, 0 for censored).
        
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

    # Handle potential NaN values to ensure lifelines.concordance_index doesn't crash
    if np.isnan(risk_pred).any():
        risk_pred = np.nan_to_num(risk_pred, nan=np.nanmedian(risk_pred))
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=np.nanmedian(y))
    if np.isnan(e).any():
        e = np.nan_to_num(e, nan=0)

    return concordance_index(y, risk_pred, e)


def preprocess_inference_data(adata_path, ppi_path, clinical_path, disease_name):
    """
    Custom data preprocessing function specifically for testing/inference.
    [CRITICAL]: Disables data shuffling to preserve the original patient order,
    ensuring that the output risk scores strictly correspond to the correct Patient IDs.
    """
    import scanpy as sc
    print(f"[*] Loading expression data (adata) from: {adata_path}")
    
    adata = sc.read_h5ad(adata_path)
    ppi_net = pd.read_csv(ppi_path)
    clinical_data = pd.read_csv(clinical_path)
    
    # --- 1. Filter PPI Edges and Proteins ---
    # Retain only high-confidence interactions
    filtered_ppi = ppi_net[ppi_net['Conn'] >= 0.5]  
    ppi_proteins = set(filtered_ppi['Source'].tolist() + filtered_ppi['Target'].tolist())
    
    # Extract intersection of proteins
    common_proteins = [gene for gene in adata.var_names if gene in ppi_proteins]
    adata = adata[:, common_proteins].copy()
    print(f"[*] Final number of retained protein nodes: {len(adata.var_names)}")
    
    protein_names = adata.var_names.tolist()
    protein_to_idx = {protein: idx for idx, protein in enumerate(protein_names)}
    
    # --- 2. Construct PyG edge_index ---
    unique_undirected_edges = set()
    for _, row in filtered_ppi.iterrows():
        p1, p2 = row['Source'], row['Target']
        if p1 != p2 and p1 in protein_to_idx and p2 in protein_to_idx:
            edge = tuple(sorted((p1, p2)))
            unique_undirected_edges.add(edge)
    
    edge_index = []
    for p1, p2 in unique_undirected_edges:
        i1 = protein_to_idx[p1]
        i2 = protein_to_idx[p2]
        edge_index.append([i1, i2])
        edge_index.append([i2, i1]) # Add reverse direction for undirected graph
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # --- 3. Extract Survival Labels and Clinical Features ---
    survival_time = adata.obs['time'].values
    event_status = adata.obs['event'].values
    patient_ids = adata.obs.index.tolist()
    
    # Align clinical data with patient order in expression data
    clinical_data_filtered = clinical_data[clinical_data['participant.eid'].astype(str).isin(patient_ids)].copy()
    clinical_data_filtered['patient_id'] = clinical_data_filtered['participant.eid'].astype(str)
    clinical_data_filtered = clinical_data_filtered.set_index('patient_id')
    clinical_data_filtered = clinical_data_filtered.reindex(patient_ids) 
    clinical_data_filtered = clinical_data_filtered.drop('participant.eid', axis=1)
    
    # ==========================================
    # Single-Sex Disease Feature Filtering
    # ==========================================
    single_sex_bases = [
        "Primary_Malignancy_Prostate",
        "Benign_neoplasm_and_polyp_of_uterus",
        "Leiomyoma_of_uterus",
        "Female_genital_prolapse",
        "Hyperplasia_of_prostate",
        "Menorrhagia_and_polymenorrhoea",
        "Postmenopausal_bleeding"
    ]
    
    # Remove suffix to check the base disease name
    base_name = disease_name.replace("_filtered", "")
    
    if base_name in single_sex_bases:
        # Check for various sex/gender column naming conventions
        cols_to_drop = [col for col in ['31', 'sex', 'Sex', 'gender', 'Gender'] if col in clinical_data_filtered.columns]
        if cols_to_drop:
            print(f"[!] Warning: Single-sex disease ({base_name}) detected. Automatically dropping sex features: {cols_to_drop}")
            clinical_data_filtered = clinical_data_filtered.drop(columns=cols_to_drop)
            
    # Separate Continuous and Categorical Variables
    continuous_vars = ['874','884','904','864','1090','1080','1070','1160','1269','1279',
                       '1289','1299','1309','1319','1438','1458','1488','1498','1528','1050',
                       '1060','2277','2139','2149','age','bmi']
    
    available_continuous_vars = [var for var in continuous_vars if var in clinical_data_filtered.columns]
    categorical_vars = [var for var in clinical_data_filtered.columns if var not in available_continuous_vars]
    
    continuous_features = clinical_data_filtered[available_continuous_vars].values if available_continuous_vars else np.zeros((len(clinical_data_filtered), 0))
    categorical_features = clinical_data_filtered[categorical_vars].values if categorical_vars else np.zeros((len(clinical_data_filtered), 0))
    
    # In pure inference mode, apply standalone standard scaling to new continuous features
    if continuous_features.shape[1] > 0:
        continuous_scaler = StandardScaler()
        continuous_features_scaled = continuous_scaler.fit_transform(continuous_features)
    else:
        continuous_features_scaled = continuous_features
        
    clinical_features_combined = np.hstack([continuous_features_scaled, categorical_features])
    clinical_dim = clinical_features_combined.shape[1]

    # --- 4. Construct PyG Data List ---
    data_list = []
    for i in range(len(patient_ids)):
        x = torch.tensor(adata.X[i].reshape(-1, 1), dtype=torch.float)
        clinical_feat = torch.tensor(clinical_features_combined[i], dtype=torch.float).view(1, -1)
        data = Data(
            x=x, 
            edge_index=edge_index,
            y=torch.tensor(survival_time[i], dtype=torch.float),
            e=torch.tensor(event_status[i], dtype=torch.float),
            clinical=clinical_feat
        )
        data_list.append(data)
        
    return data_list, patient_ids, protein_names, clinical_dim


def load_legacy_checkpoint(model, model_path, device):
    """
    Legacy Checkpoint Compatibility Loader:
    1. Parses old nested CRESCENT-style checkpoint structures.
    2. Automatically maps old scattered layer names to the newly refactored module namespaces 
       (e.g., mapping `raw_bn` to `gat_net.raw_bn`).
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    old_model_state = checkpoint.get('model_state_dict', {})
    old_vnn_state = checkpoint.get('vnn_model_state_dict', {})
    
    new_state_dict = {}
    
    # Iterate through the legacy state dict and route weights to new submodule namespaces
    for k, v in old_model_state.items():
        new_k = k
        
        # --- 1. GAT specific scattered layers -> gat_net ---
        if k.startswith(('raw_feature_fc.', 'raw_bn.', 'gat_layers.', 'bn_layers.')):
            new_k = f"gat_net.{k}"
            
        # --- 2. Clinical FNN specific layers -> fnn_net.fc/bn ---
        elif k.startswith('clinical_fc'):
            new_k = k.replace('clinical_fc', 'fnn_net.fc', 1)
        elif k.startswith('clinical_bn'):
            new_k = k.replace('clinical_bn', 'fnn_net.bn', 1)
            
        # --- 3. Fusion network -> fusion_net.fusion ---
        elif k.startswith('fusion.'):
            new_k = k.replace('fusion.', 'fusion_net.fusion.', 1)
            
        # --- 4. Survival prediction layer -> survival_net ---
        elif k.startswith('survival_fc.'):
            new_k = f"survival_net.{k}"
            
        # --- 5. VNN Model (Fixing double prefixes: vnn_net.vnn_model) ---
        elif k.startswith('vnn_model.'):
            new_k = f"vnn_net.{k}"  
            
        new_state_dict[new_k] = v
        
    # Fallback Mechanism: If VNN weights were saved in a separate dictionary (very early versions)
    for k, v in old_vnn_state.items():
        new_k = k
        if k.startswith('vnn_model.'):
            new_k = f"vnn_net.{k}"
        elif not k.startswith('vnn_'):
            new_k = f"vnn_net.vnn_model.{k}"
            
        if new_k not in new_state_dict:
            new_state_dict[new_k] = v
            
    # Inject the remapped state dictionary into the current model structure
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"\n[+] Legacy weights loaded successfully!")
    
    # Log any unmatched keys for diagnostic purposes
    if missing:
        print(f"\n[!] Missing keys (Expected by new model, but absent in remapped checkpoint):")
        for m_key in missing:
            print(f"  - {m_key}")
            
    if unexpected:
        print(f"\n[?] Unexpected keys (Present in checkpoint, but unused by new model):")
        for u_key in unexpected:
            print(f"  - {u_key}")
            
    return model


def main():
    parser = argparse.ArgumentParser(description='ProPathNet Automated Inference & Production Script')
    
    parser.add_argument('--disease', type=str, required=True, help='Target disease name for routing')
    parser.add_argument('--adata_path', type=str, required=True, help='Path to input .h5ad dataset')
    parser.add_argument('--clinical_path', type=str, required=True, help='Path to input clinical features CSV')
    
    parser.add_argument('--ppi_path', type=str, default='data/human_ppi_intact.csv', help='PPI network file path')
    parser.add_argument('--biovnn_path', type=str, default='data/biovnn_dict.pkl', help='BioVNN dictionary file path')
    parser.add_argument('--output_dir', type=str, default='results_prediction', help='Output directory for prediction results')
    
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"🚀 Initializing ProPathNet Automated Inference Pipeline")
    print(f"{'='*50}")

    # ==========================================
    # STEP 1: Automated Model Routing & Configuration
    # ==========================================
    manager = DiseaseModelManager()
    disease_info = manager.get_disease_info(args.disease)
    
    actual_disease_name = disease_info['disease_name'] 
    model_path = disease_info['model_path']
    config = disease_info['config']
    
    print(f"\n[+] Model routing matched successfully!")
    print(f"    - Disease Category: {disease_info['category']}")
    print(f"    - Applied Model Name: {actual_disease_name}")
    print(f"    - Model Weights Path: {model_path}")

    # ==========================================
    # STEP 2: Data Preprocessing
    # ==========================================
    print(f"\n>>> Preprocessing patient inference data...")
    data_list, patient_ids, gene_names, clinical_dim = preprocess_inference_data(
        adata_path=args.adata_path, 
        ppi_path=args.ppi_path, 
        clinical_path=args.clinical_path,
        disease_name=args.disease  # Pass name to enable single-sex filtering
    )
    print(f"[*] Preprocessing complete. Total samples to predict: {len(data_list)}")
    
    # ==========================================
    # STEP 3: Model Initialization & Legacy Checkpoint Loading
    # ==========================================
    print(f"\n>>> Initializing ProPathNet architecture...")
    
    test_loader = DataLoader(data_list, batch_size=int(config.get('batch_size', 256)), shuffle=False, pin_memory=True)
    
    with open(args.biovnn_path, "rb") as f:
        biovnn_dict = pickle.load(f)

    # Initialize empty model using retrieved hyperparameters
    model = ProPathNet(
        num_nodes=data_list[0].x.shape[0], 
        num_genes=len(gene_names), 
        hidden_dim=int(config['hidden_dim']), 
        num_layers=int(config['num_layers']), 
        biovnn_dict=biovnn_dict, 
        config=config, 
        clinical_dim=clinical_dim
    ).to(device)

    # Use the compatibility loader to merge and remap legacy weights
    model = load_legacy_checkpoint(model, model_path, device)
    model.eval()
    
    # ==========================================
    # STEP 4: Forward Inference & Risk Calculation
    # ==========================================
    print(f"\n[*] Executing forward pass for risk score computation...")
    all_risk_scores = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            batch_size = torch.max(batch.batch).item() + 1
            gene_expression = batch.x.view(batch_size, len(gene_names)).to(device, non_blocking=True)
            
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
    
    # Compute C-index for current prediction set (requires negating risk scores for lifelines API)
    test_cindex = c_index(risk_pred=-all_risk_scores, y=all_times, e=all_events)
    print(f"\n[+] Evaluation C-index on current prediction cohort: {test_cindex:.4f}")
    
    # ==========================================
    # STEP 5: Export Predictions to CSV
    # ==========================================
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"{actual_disease_name}_new_samples_predictions.csv")
    
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'risk_score': all_risk_scores,
        'time': all_times,
        'event': all_events
    })
    
    results_df.to_csv(out_csv, index=False)
    
    print(f"\n{'-'*50}")
    print(f"[+] Success! Full cohort risk predictions for {actual_disease_name} have been exported.")
    print(f"[+] Output File Path: {out_csv}")
    print(f"{'-'*50}\n")

if __name__ == "__main__":
    main()