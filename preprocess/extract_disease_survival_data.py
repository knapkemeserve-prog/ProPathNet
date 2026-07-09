"""
==============================================================================
File: extract_disease_survival_data.py
Description: 
    Disease event processing and survival time calculation module.
    This script is responsible for:
    1. Loading and merging UK Biobank disease diagnosis records with baseline 
       assessment dates and imputed Olink proteomics data.
    2. Applying Rank-Based Inverse Normal Transformation (INT) to protein levels.
    3. Calculating accurate follow-up time (in years), accounting for right-censoring 
       based on study end dates and the death registry.
    4. Filtering the cohort to extract specific pre-baseline prevalent cases 
       (e.g., diagnosed within 10 years prior to the baseline assessment).
==============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata

# ==============================================================================
# 1. Configuration Parameters
# ==============================================================================
# inc_yrs: Incident years threshold used to define the right-censoring date.

inc_yrs = 10  

# ==============================================================================
# 2. Helper Function: Inverse Normal Rank Transformation
# ==============================================================================
def rank_norm(x, k=0.375):
    """
    Strictly replicates the behavior of R's `RNOmni::RankNorm` function.
    
    Rules:
    - The input vector must not contain missing values (NaNs).
    - Calculates probability using the formula: (r - k) / (n - 2*k + 1).
    - Uses Blom's approximation by default (k = 0.375).
    - Uses 'average' tie-breaking method (consistent with R's default).
    - Returns the corresponding normal quantiles.
    
    Args:
        x (pd.Series or np.ndarray): Input numeric vector.
        k (float): Offset constant for transformation.
        
    Returns:
        np.ndarray: Rank-normalized vector.
    """
    if x.isnull().any():
        raise ValueError("RNOmni::RankNorm requires the input vector to have no missing values. Please impute first.")
    
    n = len(x)
    # Get ranks with average tie-breaking (same as R: rank(ties.method = "average"))
    r = rankdata(x, method='average')  
    
    # Calculate empirical probabilities
    p = (r - k) / (n - 2*k + 1)  
    
    # Clip probabilities slightly to avoid returning absolute infinity from the ppf function
    p = np.clip(p, 1e-12, 1 - 1e-12)
    
    # Map probabilities to the standard normal distribution quantiles
    q = norm.ppf(p)  
    return q

# ==============================================================================
# 3. Data Loading & Primary Merging
# ==============================================================================
print("[*] Loading disease event data...")
# Load diagnosis records for the specific disease (e.g., Hypertension)
disease_event = pd.read_csv('/bigdat2/user/xuln/olink_disease_predict/data/Hypertension_event.csv')
print(disease_event.head())

print("\n[*] Loading baseline assessment dates...")
# Load UKB Field 53 (Date of attending assessment centre)
baseline = pd.read_csv('/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p53_i0.txt')
baseline.rename(columns={'participant.p53_i0': 'date_baseline_assessment'}, inplace=True)

# Merge phenotype data (Left join to retain all disease event records)
pheno = pd.merge(disease_event, baseline, on='participant.eid', how='left')

print("\n[*] Loading and merging proteomics data...")
# Load the imputed Olink protein expression dataset
imputed_protein = pd.read_csv('/bigdat2/user/xuln/Imputed_protein_cleanoutlier.csv')
protein_list = [col for col in imputed_protein.columns if col != 'participant.eid']

# Merge phenotype with proteomics (Right join to retain all patients with protein data)
ukbb = pd.merge(pheno, imputed_protein, on='participant.eid', how='right')

# ==============================================================================
# 4. Protein Rank Normalization
# ==============================================================================
print("\n[*] Loading rank-normalized proteomics data...")
# Optional: Perform rank normalization dynamically
# ukbb[protein_list] = ukbb[protein_list].apply(rank_norm, axis=0)
# ukbb[protein_list].to_csv("/bigdat2/user/xuln/Imputed_protein_cleanoutlier_Ranknorm.csv", index=False)

# Directly load pre-computed rank-normalized protein data to save time
ukbb[protein_list] = pd.read_csv("/bigdat2/user/xuln/Imputed_protein_cleanoutlier_Ranknorm.csv")

# ==============================================================================
# 5. Date Parsing and Follow-up Time Calculation
# ==============================================================================
dz = 'flag_Hypertension'
dz_date = 'diagnosis_date_Hypertension'

# Convert date strings to Pandas datetime objects
ukbb['date_baseline_assessment'] = pd.to_datetime(ukbb['date_baseline_assessment'], errors='coerce')
ukbb[dz_date] = pd.to_datetime(ukbb[dz_date], errors='coerce')

# Calculate initial follow-up time (in years) for diagnosed cases
# Formula: (Diagnosis Date - Baseline Date) in weeks / 52.25 weeks per year
ukbb['fol'] = (ukbb[dz_date] - ukbb['date_baseline_assessment']).dt.days / 7 / 52.25

# ==============================================================================
# 6. Censoring Adjustments (Death & Study End Date)
# ==============================================================================
print("[*] Adjusting follow-up times for censoring (Death & End of study)...")
# Load death registry data (UKB Field 40000: Date of death)
death = pd.read_csv('/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p40000_i0.txt')
death.rename(columns={'participant.p40000_i0': 'date_of_death'}, inplace=True)
death['date_of_death'] = pd.to_datetime(death['date_of_death'], errors='coerce')

# Merge death dates into the main dataframe
ukbb = pd.merge(ukbb, death[['participant.eid', 'date_of_death']], on='participant.eid', how='left')

# Define the absolute censoring date based on the chosen follow-up window (inc_yrs)
if inc_yrs == 10:
    censor_date = pd.Timestamp('2020-12-31')
elif inc_yrs == 5:
    censor_date = pd.Timestamp('2016-05-31')
else:
    raise ValueError("inc_yrs must be either 5 or 10.")

# Data Cleaning: Remove cases where the event flag is 1 but the diagnosis date is missing
ukbb = ukbb[~((ukbb[dz] == 1) & ukbb[dz_date].isna())].copy()

# Adjust follow-up time for controls (individuals who never developed the disease)
# By default, their follow-up time spans from baseline to the study censoring date.
mask_no_event = (ukbb[dz] == 0)
ukbb.loc[mask_no_event, 'fol'] = (censor_date - ukbb.loc[mask_no_event, 'date_baseline_assessment']).dt.days / 7 / 52.25

# Account for death censoring:
# If a control patient died before the study's censor date, their follow-up time 
# should be truncated to their date of death.
mask_death_before_censor = (ukbb[dz] == 0) & (ukbb['date_of_death'] < censor_date)
ukbb.loc[mask_death_before_censor, 'fol'] = (
    ukbb.loc[mask_death_before_censor, 'date_of_death'] - 
    ukbb.loc[mask_death_before_censor, 'date_baseline_assessment']
).dt.days / 7 / 52.25

# ==============================================================================
# 7. Final Formatting and Pre-baseline Cohort Filtering
# ==============================================================================
# Extract the necessary columns and standardize names for downstream ML models
selected_data = ukbb[['participant.eid', dz, 'fol'] + protein_list].copy()
selected_data.rename(columns={dz: 'Event', 'fol': 'Time'}, inplace=True)

# Convert all protein column names to UPPERCASE to ensure consistency
selected_data.rename(columns={col: col.upper() for col in protein_list}, inplace=True)

print("\n[*] Filtering cohort to retain only prevalent cases within 10 years before baseline...")
# Crucial Filtering Step:
# We want to EXCLUDE cases (Event == 1) that occurred:
# 1. More than 10 years before baseline (Time < -10)
# 2. After baseline (Incident cases, Time > 0)
# This retains healthy controls (Event == 0) AND cases diagnosed within the [-10, 0] year window.
filtered_data = selected_data[~((selected_data['Event'] == 1) & ((selected_data['Time'] < -10) | (selected_data['Time'] > 0)))]

print("\n[+] Final filtered dataset preview:")
print(filtered_data.head())

# Save the final pre-baseline cohort to CSV
output_path = '/bigdat2/user/xuln/olink_disease_predict/data/Hypertension_before_baseline.csv'
filtered_data.to_csv(output_path, index=False)
print(f"\n[+] Successfully saved the pre-baseline dataset to: {output_path}")