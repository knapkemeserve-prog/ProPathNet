"""
==============================================================================
File: data_loader.py
Description: 
    Data preprocessing and dataset construction module for ProPathNet.
    This script is responsible for:
    1. Loading and filtering multi-omics data (Proteomics H5AD, Clinical CSV).
    2. Constructing the Protein-Protein Interaction (PPI) graph structure (PyG format).
    3. Aligning and cleaning clinical/lifestyle covariates (handling sex-specific diseases).
    4. Splitting and scaling the data for model training, validation, and inference.
==============================================================================
"""

import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

def preprocess_data(adata_path, ppi_path, clinical_path, disease_name, test_size=0.2, val_size=0.125, random_state=42):
    """
    Data preprocessing and PyTorch Geometric (PyG) graph dataset construction module.
    
    Args:
        adata_path (str): Path to the single-cell/bulk expression data (.h5ad format).
        ppi_path (str): Path to the Protein-Protein Interaction network data (.csv format).
        clinical_path (str): Path to the clinical and lifestyle features (.csv format).
        disease_name (str): Name of the target disease (used to filter sex-specific features).
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Seed used by the random number generator.
        
    Returns:
        tuple: Contains train/val/test PyG Data lists, protein-to-index mapping, 
               list of gene names, and the dimension of clinical features.
    """
    print(f"[*] Loading expression data (adata) from: {adata_path}")
    print("[*] Starting clinic-protein data tuning and alignment...")
    
    # 1. Load raw data
    adata = sc.read_h5ad(adata_path)
    ppi_net = pd.read_csv(ppi_path)
    clinical_data = pd.read_csv(clinical_path)
    
    # 2. Filter PPI network
    # Retain only high-confidence protein-protein interactions (weight >= 0.5)
    filtered_ppi = ppi_net[ppi_net['Conn'] >= 0.5]  

    # Extract all unique proteins present in the filtered PPI network
    ppi_proteins = set(filtered_ppi['Source'].tolist() + filtered_ppi['Target'].tolist())
    
    # Find the intersection between proteins in the PPI network and proteins in the expression data
    common_proteins = [gene for gene in adata.var_names if gene in ppi_proteins]
    
    # Subset the expression data to keep only the intersecting proteins
    adata = adata[:, common_proteins].copy()
    print(f"[*] Final number of retained protein nodes: {len(adata.var_names)}")
    
    # Create mappings from protein names to numerical indices for graph construction
    protein_names = adata.var_names.tolist()
    protein_to_idx = {protein: idx for idx, protein in enumerate(protein_names)}
    
    # 3. Construct Graph Edges (edge_index)
    # Use a set to avoid duplicate undirected edges
    unique_undirected_edges = set()
    for _, row in filtered_ppi.iterrows():
        p1, p2 = row['Source'], row['Target']
        # Only add edge if both proteins exist in our filtered dataset
        if p1 != p2 and p1 in protein_to_idx and p2 in protein_to_idx:
            # Sort the tuple to treat (A, B) and (B, A) as the same undirected edge
            edge = tuple(sorted((p1, p2)))
            unique_undirected_edges.add(edge)
    
    # Convert unique edges into PyG edge_index format (bidirectional/undirected)
    edge_index = []
    for p1, p2 in unique_undirected_edges:
        i1 = protein_to_idx[p1]
        i2 = protein_to_idx[p2]
        edge_index.append([i1, i2])
        edge_index.append([i2, i1]) # Add reverse direction for undirected graph
    
    # Transpose to match PyG's expected shape [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"[*] Graph Edge Index shape: {edge_index.shape}")
    
    # Sanity checks to ensure edge indices are within valid node bounds
    num_nodes = len(protein_names)
    assert edge_index.max() < num_nodes, f"Edge index out of bounds: {edge_index.max()} >= {num_nodes}"
    assert edge_index.min() >= 0, f"Edge index contains negative values: {edge_index.min()}"
    
    # 4. Extract survival labels and patient IDs
    survival_time = adata.obs['time'].values
    event_status = adata.obs['event'].values
    patient_ids = adata.obs.index.tolist()
    
    # 5. Align Clinical Data
    # Filter clinical data to include only the patients present in the expression data
    clinical_data_filtered = clinical_data[clinical_data['participant.eid'].astype(str).isin(patient_ids)].copy()
    clinical_data_filtered['patient_id'] = clinical_data_filtered['participant.eid'].astype(str)
    clinical_data_filtered = clinical_data_filtered.set_index('patient_id')
    
    # Reindex to ensure clinical data exactly matches the patient order in `adata`
    clinical_data_filtered = clinical_data_filtered.reindex(patient_ids)
    clinical_data_filtered = clinical_data_filtered.drop('participant.eid', axis=1)
    
    # ==========================================
    # 6. Sex-Specific Clinical Feature Filtering
    # ==========================================
    # List of diseases that exclusively affect one biological sex
    single_sex_bases = [
        "Primary_Malignancy_Prostate",
        "Benign_neoplasm_and_polyp_of_uterus",
        "Leiomyoma_of_uterus",
        "Female_genital_prolapse",
        "Hyperplasia_of_prostate",
        "Menorrhagia_and_polymenorrhoea",
        "Postmenopausal_bleeding"
    ]
    
    # Remove suffix if present to match base disease names
    base_name = disease_name.replace("_filtered", "")
    
    # If the disease is sex-specific, remove sex/gender features to prevent confounding
    if base_name in single_sex_bases:
        cols_to_drop = [col for col in ['31', 'sex', 'Sex', 'gender', 'Gender'] if col in clinical_data_filtered.columns]
        if cols_to_drop:
            print(f"[!] Warning: Single-sex disease ({base_name}) detected. Automatically dropping sex-related columns: {cols_to_drop}")
            clinical_data_filtered = clinical_data_filtered.drop(columns=cols_to_drop)
    # ==========================================
    
    # 7. Separate Continuous and Categorical Clinical Variables
    continuous_vars = ['874','884','904','864','1090','1080','1070','1160','1269','1279',
                       '1289','1299','1309','1319','1438','1458','1488','1498','1528','1050',
                       '1060','2277','2139','2149','age','bmi']
    
    available_continuous_vars = [var for var in continuous_vars if var in clinical_data_filtered.columns]
    categorical_vars = [var for var in clinical_data_filtered.columns if var not in available_continuous_vars]
    
    # Convert DataFrames to numpy arrays. Fallback to empty arrays if no features exist.
    continuous_features = clinical_data_filtered[available_continuous_vars].values if available_continuous_vars else np.zeros((len(clinical_data_filtered), 0))
    categorical_features = clinical_data_filtered[categorical_vars].values if categorical_vars else np.zeros((len(clinical_data_filtered), 0))
    
    # 8. Stratified Train-Validation-Test Split
    # Split out the test set
    train_val_indices, test_indices = train_test_split(
        range(len(patient_ids)), test_size=test_size, stratify=event_status, random_state=random_state
    )
    
    # Split the remaining data into train and validation sets
    train_val_events = event_status[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_events, random_state=random_state
    )
    
    # 9. Feature Scaling
    # To prevent data leakage, fit the scaler ONLY on the training data, then transform all sets
    if continuous_features.shape[1] > 0:
        continuous_scaler = StandardScaler()
        train_continuous = continuous_features[train_indices]
        continuous_scaler.fit(train_continuous)
        continuous_features_scaled = continuous_scaler.transform(continuous_features)
    else:
        continuous_features_scaled = continuous_features
    
    # Concatenate scaled continuous features with raw categorical features
    clinical_features_combined = np.hstack([continuous_features_scaled, categorical_features])
    clinical_dim = clinical_features_combined.shape[1]

    # 10. Construct PyTorch Geometric (PyG) Data Objects
    def create_data_list(indices):
        """Helper function to build PyG Data objects for a given set of indices."""
        data_list = []
        for i in indices:
            # Node features: Protein expression levels for patient 'i' (Shape: [num_nodes, 1])
            x = torch.tensor(adata.X[i].reshape(-1, 1), dtype=torch.float)
            
            # Graph-level features: Clinical data for patient 'i' (Shape: [1, clinical_dim])
            clinical_feat = torch.tensor(clinical_features_combined[i], dtype=torch.float).view(1, -1)
            
            # Construct graph Data object
            data = Data(
                x=x, 
                edge_index=edge_index,
                y=torch.tensor(survival_time[i], dtype=torch.float), # Survival time
                e=torch.tensor(event_status[i], dtype=torch.float),  # Event occurrence (censoring status)
                clinical=clinical_feat
            )
            data_list.append(data)
        return data_list

    # Generate data lists for each split
    train_data_list = create_data_list(train_indices)
    val_data_list = create_data_list(val_indices)
    test_data_list = create_data_list(test_indices)
   
    return train_data_list, val_data_list, test_data_list, protein_to_idx, adata.var_names.tolist(), clinical_dim