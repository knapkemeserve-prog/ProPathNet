"""
==============================================================================
File: process_intact_ppi.py
Description: 
    IntAct database human Protein-Protein Interaction (PPI) network preprocessing.
    This script is responsible for:
    1. Reading massive interaction data from the raw IntAct export (human.txt).
    2. Filtering and extracting pure human (Homo sapiens) PPI pairs.
    3. Cleaning missing values and abnormal IDs (pure digits, specific prefixes).
    4. Extracting Confidence values using regular expressions.
    5. Handling duplicate edges by retaining the record with the maximum confidence.
    6. Standardizing protein/gene names (uppercase, removing suffixes) and 
       exporting as a clean CSV network file.
==============================================================================
"""

import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Raw IntAct Data
# ==========================================
print("[*] Loading raw IntAct data file...")
file_path = "/bigdat2/user/xuln/human.txt"
# Use tab separator; low_memory=False prevents warnings due to mixed data types
human_ppi = pd.read_csv(file_path, sep='\t', low_memory=False)
print(f"[*] Raw data loaded successfully. Total rows: {len(human_ppi)}")

# ==========================================
# 2. Basic Cleaning: Remove Invalid Interaction Rows
# ==========================================
# Filter out rows where Alias(es) interactor A or B is NaN, empty string, or '-'
invalid_rows = human_ppi[
    human_ppi['Alias(es) interactor A'].isna() |
    (human_ppi['Alias(es) interactor A'] == '') |
    (human_ppi['Alias(es) interactor A'] == '-') |
    human_ppi['Alias(es) interactor B'].isna() |
    (human_ppi['Alias(es) interactor B'] == '') |
    (human_ppi['Alias(es) interactor B'] == '-')
]

# Remove these invalid rows from the main dataframe
human_ppi = human_ppi[~human_ppi.index.isin(invalid_rows.index)]
print(f"[*] Rows remaining after dropping invalid/missing aliases: {len(human_ppi)}")

# ==========================================
# 3. Species Filtering: Retain Only Human Interactions
# ==========================================
# Extract core columns needed: ID of A, ID of B, Species of A (9), Species of B (10), Confidence (14)
protein_interactor = human_ppi.iloc[:, [4, 5, 9, 10, 14]]

col_a = human_ppi.columns[9]  # Taxid interactor A
col_b = human_ppi.columns[10] # Taxid interactor B

def is_human(alias_str):
    """Check if the species string contains the human identifier."""
    return 'Homo sapiens' in str(alias_str)

# Apply filter: Both interactors must be human
human_only_ppi = protein_interactor[
    human_ppi[col_a].apply(is_human) & 
    human_ppi[col_b].apply(is_human)
]

# Keep only ID of A (0), ID of B (1), and Confidence value (4)
human_only_ppi = human_only_ppi.iloc[:, [0, 1, 4]]
print(f"[*] Rows remaining after pure human filtering: {len(human_only_ppi)}")

# ==========================================
# 4. Extract Confidence Weights
# ==========================================
def extract_float(x):
    """Extract float numbers from complex text strings using regex."""
    match = re.search(r'([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)', str(x))
    if match:
        return float(match.group(1))
    return float('nan')

# Apply extraction function and create a new 'Conn' (Connection weight) column
human_only_ppi['Conn'] = human_only_ppi['Confidence value(s)'].apply(extract_float)

# Filter out anomalous confidence values > 1.0 (typically bounded between 0 and 1)
human_only_ppi = human_only_ppi[human_only_ppi['Conn'] <= 1.0]

# ==========================================
# 5. Extract Standardized Protein/Gene Identifiers
# ==========================================
# Retain only rows containing the standard 'psi-mi:' format in their Alias
psi_mi_rows = human_only_ppi[
    human_only_ppi['Alias(es) interactor A'].str.contains('psi-mi:', na=False) |
    human_only_ppi['Alias(es) interactor B'].str.contains('psi-mi:', na=False)
]

def extract_psi_id(s):
    """Extract the core ID after 'psi-mi:' and truncate at the first parenthesis."""
    if 'psi-mi:' in str(s):
        return str(s).split('psi-mi:')[-1].split('(')[0]
    return None

# Store extracted IDs into Source and Target columns representing graph nodes
psi_mi_rows['Source'] = psi_mi_rows['Alias(es) interactor A'].apply(extract_psi_id)
psi_mi_rows['Target'] = psi_mi_rows['Alias(es) interactor B'].apply(extract_psi_id)

# Extract the final 3 required columns
human_ppi_intact = psi_mi_rows[['Source', 'Target', 'Conn']]

# ==========================================
# 6. First Round of Identifier Filtering & Deduplication
# ==========================================
# Filter out nodes containing specific non-target database prefixes (WUGSC, EBI, ENSG)
pattern = 'WUGSC:|EBI-|ENSG'
mask = ~(human_ppi_intact['Source'].str.contains(pattern, na=False) | 
         human_ppi_intact['Target'].str.contains(pattern, na=False))

filtered_df = human_ppi_intact[mask].copy()

# Deduplication Logic: For duplicate edges (same Source and Target), retain the one with max Conn
filtered_df = filtered_df.loc[filtered_df.groupby(['Source', 'Target'])['Conn'].idxmax()]
filtered_df = filtered_df.reset_index(drop=True)

# ==========================================
# 7. Second Round: Pure Numeric Identifier Filtering
# ==========================================
def is_pure_number(s):
    """Check if the string consists entirely of digits (invalid gene/protein name)."""
    if pd.isna(s):
        return False
    return bool(re.fullmatch(r'\d+', str(s)))

# Remove rows where either source or target node is purely numeric
mask_numeric = ~(filtered_df['Source'].apply(is_pure_number) | 
                 filtered_df['Target'].apply(is_pure_number))

filtered_df = filtered_df[mask_numeric].copy()

# Safely deduplicate based on weights once more
filtered_df = filtered_df.loc[filtered_df.groupby(['Source', 'Target'])['Conn'].idxmax()]
filtered_df = filtered_df.reset_index(drop=True)

# ==========================================
# 8. Format Standardization & Save Final Result
# ==========================================
# Remove redundant suffixes
filtered_df['Source'] = filtered_df['Source'].str.replace('_human', '', regex=False)
filtered_df['Target'] = filtered_df['Target'].str.replace('_human', '', regex=False)

# Convert all node identifiers to UPPERCASE for perfect matching with clinical expression data (H5AD)
filtered_df['Source'] = filtered_df['Source'].str.upper()
filtered_df['Target'] = filtered_df['Target'].str.upper()

# Export the final cleaned PPI network
output_path = '/home/xuln/GCN_network/human_ppi_intact.csv'
filtered_df.to_csv(output_path, index=False)

print(f"\n[*] Processing Complete!")
print(f"    - Final valid interaction edges (deduplicated): {len(filtered_df)}")
print(f"    - File successfully saved to: {output_path}")
print("\n[+] Preview of final data format:")
print(filtered_df.head())