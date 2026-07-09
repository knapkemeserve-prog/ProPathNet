"""
==============================================================================
File: process_ukb_lifestyle.py
Description: 
    UK Biobank (UKB) lifestyle and environmental data processing script.
    This script represents the complete, end-to-end optimized pipeline. 
    It is responsible for:
    1. Loading explicit core baseline demographics (Age, Sex, BMI, etc.).
    2. Batch loading 146 specific UKB lifestyle fields.
    3. Integrating both datasets into a comprehensive feature matrix.
    4. Handling UKB-specific negative encoded values (e.g., -1 for "Do not know", 
       -3 for "Prefer not to answer", -10 for "Less than one").
    5. Filtering out features with unacceptably high missing rates (> 20%).
    6. Imputing missing values (Median for continuous, One-Hot/Zero for categorical).
    7. Extracting the retained data dictionary from the metadata Excel file.
==============================================================================
"""

import pandas as pd
import numpy as np
import os
from functools import reduce

# ==============================================================================
# STEP 1: LOAD CORE DEMOGRAPHICS & BASIC LIFESTYLE FEATURES
# ==============================================================================
print("=" * 60)
print("[*] STEP 1: Loading Core Demographics and Baseline Features")
print("=" * 60)

# Define explicit paths for fundamental baseline traits
core_files = [
    ("Age", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p21003_i0.txt"),
    ("Sex", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p31.txt"),
    ("Ethnicity", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p22006.txt"),
    ("Smoking", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p20116_i0.txt"),
    ("Alcohol", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p1558_i0.txt"),
    ("Paternal_diseases", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p20107_i0.txt"),
    ("Maternal_diseases", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p20110_i0.txt"),
    ("BMI", "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p21001_i0.txt")
]

core_dfs = []
for name, file_path in core_files:
    try:
        df = pd.read_csv(file_path)
        # Rename the first column to 'participant.eid' and the second column to the specific feature name
        df.rename(columns={df.columns[0]: 'participant.eid', df.columns[1]: name}, inplace=True)
        core_dfs.append(df)
        print(f"    [+] Successfully loaded core feature: {name}")
    except Exception as e:
        print(f"    [x] Error loading {name} at {file_path}: {e}")

# Merge all core demographic dataframes together using an outer join
core_df = reduce(lambda left, right: pd.merge(left, right, on='participant.eid', how='outer'), core_dfs)
print(f"[*] Core demographics merged shape: {core_df.shape}")


# ==============================================================================
# STEP 2: BATCH DATA LOADING FOR LIFESTYLE FIELDS
# ==============================================================================
print("\n" + "=" * 60)
print("[*] STEP 2: Batch Loading UKB Lifestyle Fields")
print("=" * 60)

# Target UKB field IDs for extensive lifestyle/environmental profiling
file_numbers = [
    1100, 2634, 1021, 894, 10962, 3647, 1001, 914, 10971, 874, 10953, 981, 2624, 1011, 3637, 943, 991, 971, 884, 904, 864, 1090, 1080, 1070, 6164, 6162, 924,
    1110, 1120, 10749, 10016, 1130, 1140, 10886, 1150, 2237, 10105, 10114,
    1160, 1170, 1180, 1190, 1200, 1210, 1220,
    20160, 20162, 20161, 10895, 20116, 1239, 1249, 2644, 3436, 3446, 5959, 3456, 6194, 6183, 3466, 3476, 3486, 3496, 3506, 6158, 2867, 2877, 2887, 2897, 2907, 10827, 6157, 10115, 2926, 2936, 1259, 1269, 1279,
    1289, 1299, 1309, 1319, 1329, 1339, 1349, 1359, 1369, 1379, 1389, 3680, 6144, 10855, 1408, 1418, 1428, 2654, 10767, 1438, 1448, 10776, 1458, 1468, 1478, 1488, 1498, 1508, 1518, 1528, 1538, 1548, 10912,
    20117, 1558, 3731, 4407, 4418, 4429, 4440, 4451, 4462, 1568, 1578, 1588, 1598, 1608, 5364, 1618, 1628, 2664, 10818, 3859, 10853,
    1050, 1060, 1717, 1727, 1737, 1747, 1757, 2267, 2277,
    2129, 2139, 2149, 2159, 3669
]

# Exclude variables that represent multiple-choice arrays to avoid dimension explosion
multi_choice_vars = [6164, 6162, 6158, 6157, 6144, 10855]
file_numbers_filtered = [num for num in file_numbers if num not in multi_choice_vars]
print(f"[*] Total target batch fields to load: {len(file_numbers_filtered)}") 

batch_dataframes = {}
for num in file_numbers_filtered:
    file_path = f"/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p{num}_i0.txt"
    try:
        df = pd.read_csv(file_path)
        # Rename the second column to string format of the field ID
        df.rename(columns={df.columns[1]: str(num)}, inplace=True)
        batch_dataframes[str(num)] = df
    except Exception as e:
        print(f"    [x] Error loading Field {num}: {e}")

# Merge all batch DataFrames
print("\n[*] Merging all batch loaded UKB fields...")
batch_merged_df = None
for num in file_numbers_filtered:
    df_name = str(num)
    if df_name in batch_dataframes:
        if batch_merged_df is None:
            batch_merged_df = batch_dataframes[df_name]
        else:
            batch_merged_df = pd.merge(batch_merged_df, batch_dataframes[df_name], on=batch_merged_df.columns[0], how='outer')

print(f"[*] Batch dataframe shape: {batch_merged_df.shape}")


# ==============================================================================
# STEP 3: INTEGRATE CORE AND BATCH DATASETS
# ==============================================================================
print("\n" + "=" * 60)
print("[*] STEP 3: Integrating Core Demographics and Batch Features")
print("=" * 60)

# Merge explicit core features with the batch loaded features (Left join on patient IDs from batch)
merged_cleaned = pd.merge(core_df, batch_merged_df, on='participant.eid', how='right')

# Optionally drop Paternal and Maternal diseases if not needed (Uncomment to drop)
# merged_cleaned = merged_cleaned.drop(columns=['Paternal_diseases', 'Maternal_diseases'])

# Define subsets for processing
core_continuous = ['Age', 'BMI']
core_categorical = ['Sex', 'Ethnicity', 'Smoking', 'Alcohol', 'Paternal_diseases', 'Maternal_diseases']

batch_continuous_vars = [
    894, 914, 874, 884, 904, 864, 1090, 1080, 1070, 1160,
    20162, 20161, 3436, 3456, 6194, 6183, 2867, 2887, 2897, 1269, 1279, 2926,
    1289, 1299, 1309, 1319, 3680, 1438, 1458, 1488, 1498, 1528,
    4407, 4418, 4429, 4440, 4451, 4462, 1568, 1578, 1588, 1598, 1608, 5364,
    1050, 1060, 1737, 2277, 2139, 2149, 3669
]

continuous_vars = core_continuous + [str(num) for num in batch_continuous_vars]
categorical_vars = core_categorical + [str(num) for num in file_numbers_filtered if num not in batch_continuous_vars]

print(f"[*] Defined Continuous variables: {len(continuous_vars)}")
print(f"[*] Defined Categorical variables: {len(categorical_vars)}")


# ==============================================================================
# STEP 4: HANDLING UKB SPECIFIC ENCODINGS
# ==============================================================================
# Handling UKB Specific Encodings:
# -1 : "Do not know" -> Convert to NaN
# -3 : "Prefer not to answer" -> Convert to NaN
# -10: "Less than one" -> Impute as 0.5 (e.g., portions per week)
# -2 : Sometimes means 0 or None -> Convert to 0 for continuous vars

print("\n" + "=" * 60)
print("[*] STEP 4: Processing UKB specific negative encodings")
print("=" * 60)

minus_one_count_before = (merged_cleaned == -1).sum().sum()
minus_three_count_before = (merged_cleaned == -3).sum().sum()
minus_two_count_before = (merged_cleaned == -2).sum().sum()

print(f"    - Count of '-1' before processing: {minus_one_count_before}")
print(f"    - Count of '-3' before processing: {minus_three_count_before}")
print(f"    - Count of '-2' before processing: {minus_two_count_before}")

# Global replacements for missing data codes
merged_cleaned = merged_cleaned.replace([-1, -3], np.nan)
merged_cleaned = merged_cleaned.replace([-10], 0.5)
print("[+] Successfully converted '-1' and '-3' to NaN, and '-10' to 0.5")

# Continuous-specific replacements (-2 to 0)
for col in continuous_vars:
    if col in merged_cleaned.columns:
        merged_cleaned[col] = merged_cleaned[col].replace(-2, 0)

minus_two_count_after = 0
for col in continuous_vars:
    if col in merged_cleaned.columns:
        minus_two_count_after += (merged_cleaned[col] == -2).sum()
        
print(f"[+] Count of '-2' in continuous variables after processing: {minus_two_count_after}")

merged_cleaned.set_index('participant.eid', inplace=True)

# Identify which defined variables actually exist in the merged dataframe
existing_continuous = [var for var in continuous_vars if var in merged_cleaned.columns]
existing_categorical = [var for var in categorical_vars if var in merged_cleaned.columns]

print(f"[*] Actually existing continuous features to process: {len(existing_continuous)}")
print(f"[*] Actually existing categorical features to process: {len(existing_categorical)}")


# ==============================================================================
# STEP 5: FEATURE SELECTION (Missing Rate Filtering)
# ==============================================================================
print("\n" + "=" * 60)
print("[*] STEP 5: Feature Selection by Missing Rate")
print("=" * 60)

final_imputed = merged_cleaned.copy()
missing_ratio = final_imputed.isnull().sum() / len(final_imputed)

print("\n[*] Top 10 columns with the highest missing data ratios:")
print(missing_ratio.sort_values(ascending=False).head(10))

# Quality Control: Drop features missing more than 20% of their data
columns_to_drop = missing_ratio[missing_ratio > 0.2].index.tolist()
print(f"\n[*] Found {len(columns_to_drop)} columns with missing ratio > 20%")

final_imputed_cleaned = final_imputed.drop(columns=columns_to_drop)
print(f"[*] Shape before dropping: {final_imputed.shape}")
print(f"[*] Shape after dropping: {final_imputed_cleaned.shape}")

# Drop specific raw fields since we already explicitly loaded them as 'Smoking' and 'Alcohol'
columns_to_remove = ['20116', '1558']
for col in columns_to_remove:
    if col in final_imputed_cleaned.columns:
        final_imputed_cleaned = final_imputed_cleaned.drop(columns=[col])
        print(f"[+] Dropped redundant raw column to prevent duplication: {col}")

# Update lists of valid variables
remaining_continuous = [var for var in existing_continuous if var in final_imputed_cleaned.columns]
remaining_categorical = [var for var in existing_categorical if var in final_imputed_cleaned.columns]


# ==============================================================================
# STEP 6: IMPUTATION & ONE-HOT ENCODING
# ==============================================================================
print("\n" + "=" * 60)
print("[*] STEP 6: Missing Value Imputation and One-Hot Encoding")
print("=" * 60)

if remaining_continuous:
    print("    [-] Imputing continuous variables with median:")
    for var in remaining_continuous:
        missing_count = final_imputed_cleaned[var].isnull().sum()
        if missing_count > 0:
            median_value = final_imputed_cleaned[var].median()
            final_imputed_cleaned[var].fillna(median_value, inplace=True)
            print(f"        -> {var}: Filled {missing_count} missing values with median ({median_value:.4f})")

print("\n[*] Starting One-Hot Encoding for categorical variables...")
vars_to_encode = [var for var in remaining_categorical if var in final_imputed_cleaned.columns]
print(f"    [-] Categorical variables to be encoded: {len(vars_to_encode)}")

# Create dummy variables and cast to integer (0/1)
final_encoded = pd.get_dummies(final_imputed_cleaned, columns=vars_to_encode, 
                               prefix=vars_to_encode, dtype=int)

print("[+] One-Hot Encoding completed successfully!")
print(f"[*] Shape before encoding: {final_imputed_cleaned.shape}")
print(f"[*] Shape after encoding: {final_encoded.shape}")


# ==============================================================================
# STEP 7: DATA DICTIONARY META-DATA EXTRACTION
# ==============================================================================
print("\n" + "=" * 60)
print("[*] STEP 7: Data Dictionary Meta-data Extraction")
print("=" * 60)

# Final lists of retained categorical (group1) and continuous (group2) field IDs 
# corresponding to the 10.08 version selection
group1 = ['1100', '943', '924', '1110', '1120', '1130', '1140', '1150', '2237', 
          '1170', '1180', '1190', '1200', '1210', '1220', '20160', '1239', '1249', 
          '1259', '1329', '1339', '1349', '1359', '1369', '1379', '1389', '1408', 
          '1418', '1428', '1448', '1468', '1478', '1518', '1538', '1548', '20117', 
          '1628', '1717', '1727', '1747', '1757', '2267', '2129', '2159']

group2 = ['874', '884', '904', '864', '1090', '1080', '1070', '1160', '1269', 
          '1279', '1289', '1299', '1309', '1319', '1438', '1458', '1488', '1498', 
          '1528', '1050', '1060', '2277', '2139', '2149']

# Load the raw UKB data dictionary Excel file
file_path = "/home/xuln/生活习惯数据(version1).xlsx"
print(f"[*] Reading data dictionary from: {file_path}")

try:
    df_meta = pd.read_excel(file_path, sheet_name="Lifestyle and environment")
    
    # Rename the long URL column to a simpler 'A' (represents the UKB Field ID)
    df_meta = df_meta.rename(columns={'https://biobank.ctsu.ox.ac.uk/crystal/label.cgi?id=100050': 'A'})
    df_meta['A'] = df_meta['A'].astype(str)

    # Extract rows corresponding to retained categorical features
    group1_df = df_meta[df_meta['A'].isin(group1)]
    print(f"[*] Found {len(group1_df)} matching rows for categorical features (Group 1)")

    # Extract rows corresponding to retained continuous features
    group2_df = df_meta[df_meta['A'].isin(group2)]
    print(f"[*] Found {len(group2_df)} matching rows for continuous features (Group 2)")

    # Concatenate both groups
    combined_df = pd.concat([group1_df, group2_df], ignore_index=True)
    print(f"[*] Total rows combined: {len(combined_df)}")

    # Export the filtered metadata to a new Excel file
    output_file = "缺失率小于20%的生活习惯数据.xlsx"
    combined_df.to_excel(output_file, index=False)
    print(f"[+] Dictionary successfully saved to: {output_file}")
    
except Exception as e:
    print(f"[x] Error processing metadata Excel file: {e}")

print("\n[+] PIPELINE EXECUTION COMPLETELY FINISHED!")