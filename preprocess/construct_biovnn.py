"""
==============================================================================
File: construct_biovnn.py
Description: 
    BioVNN (Biological Visible Neural Network) data dictionary construction script.
    Workflow:
    1. Reads gene/protein expression data (.h5ad) and PPI network (.csv).
    2. Filters low-confidence PPI edges (Conn < 0.5) and extracts valid proteins.
    3. Finds the intersection of proteins between the expression profile and PPI.
    4. Calls the utils_biovnn module to generate the hierarchical mapping 
       dictionary required by BioVNN based on biological pathway databases,
       and saves the result as a .pkl file for downstream deep learning models.
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import sys
import scanpy as sc

# ==========================================
# 1. Environment & Path Configuration
# ==========================================
# Append project root to system path to allow importing custom modules
sys.path.append('/home/xuln/olink_disease_predict/ProPathNet-github')

# Import the module responsible for constructing the Visible Neural Network (VNN)
import models.utils_biovnn as ub

# ==========================================
# 2. Data Loading
# ==========================================
# Read patient expression profile data (AnnData format)
adata_path = '/home/xuln/olink_disease_predict/ProPathNet-github/data/sample_data.h5ad'
adata = sc.read_h5ad(adata_path)

# Print all gene names in the expression data for preliminary inspection
print("[*] List of genes in expression data:")
print(adata.var['gene'].values)

# Read Protein-Protein Interaction (PPI) network data
ppi_path = '/home/xuln/olink_disease_predict/ProPathNet-github/data/human_ppi_intact.csv'
ppi_net = pd.read_csv(ppi_path)

# ==========================================
# 3. Data Filtering & Node Alignment
# ==========================================
# Filter PPI network: Retain only highly confident edges (weight/Conn >= 0.5)
filtered_ppi = ppi_net[ppi_net['Conn'] >= 0.5]  

# Extract all unique protein nodes appearing in the filtered PPI network
ppi_proteins = set(filtered_ppi['Source'].tolist() + filtered_ppi['Target'].tolist())

# Find intersection: Retain proteins present in BOTH expression data and PPI network
# This step ensures node alignment for Graph Attention Networks (GAT) and VNN.
common_proteins = [gene for gene in adata.var['gene'].values if gene in ppi_proteins]

print(f"[*] Final number of valid intersection proteins: {len(common_proteins)}")
print(f"[*] Preview of intersection proteins:\n{common_proteins[:20]} ...")

# ==========================================
# 4. Construct and Save BioVNN Dictionary
# ==========================================
# Convert intersection proteins to a list as base input features for BioVNN
genelist = list(common_proteins)

# Define output directory for the generated BioVNN dictionary (.pkl file)
result_dir = '/home/xuln/olink_disease_predict/ProPathNet-github/data'

print(f"\n[*] Starting BioVNN dictionary construction, please wait...")

# Initialize BioVNN preprocessing object
biovnn_pre = ub.BioVNN_pre(genelist, result_dir)

# Execute construction:
# This method looks up biological pathways based on the input genelist, 
# builds sparse matrix mappings, and automatically saves the result as a .pkl file.
biovnn_pre.perform() 

print(f"[*] ✅ BioVNN dictionary constructed successfully! File saved in: {result_dir}")