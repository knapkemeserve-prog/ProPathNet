"""
==============================================================================
File: extract_and_impute_clinic_data.py
Description: 
    UK Biobank (UKB) Clinical Feature Extraction and Imputation Module.
    This script performs a two-step pipeline:
    1. Extraction: Batch reads 67 specific trait/clinical features from UKB raw 
       text files, automatically handling different file naming conventions, 
       and merges them into a single dataframe.
    2. Imputation: Cleans and imputes missing values in the extracted features. 
       Applies specific rules for certain biomarkers (e.g., sex-specific 
       imputation for fields 30800, 30850) and general median/mode imputation 
       for the rest.
==============================================================================
"""

import os
import pandas as pd
import numpy as np

# ==============================================================================
# PART 1: CONFIGURATION & PATH SETUP
# ==============================================================================
# 1. Directory configuration
DATA_DIR = "/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/"
OUT_PATH = '/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/'
OUT_FILE_RAW = "MILTON_features.csv"         # Output for extracted raw data
OUT_FILE_IMPUTED = "MILTON_features_imputed.csv" # Output for final imputed data

# 2. Target UKB Field IDs (67 specific traits)
raw_id_string = """
31 48 49 50 74 102 4079 4080 20150 20151 20258 21001 21003 
30000 30010 30020 30030 30040 30050 30060 30080 30100 
30120 30130 30140 30150 30160 30170 30250 30260 30270 
30280 30300 30500 30510 30520 30530 30600 30610 30620 
30630 30640 30650 30660 30670 30680 30690 30700 30710 
30720 30730 30740 30750 30760 30770 30780 30790 30800 
30810 30820 30830 30840 30850 30860 30870 30880 30890
"""

# Extract unique IDs and convert to a list
target_field_ids = list(set(raw_id_string.split()))

# ==============================================================================
# PART 2: DATA EXTRACTION FUNCTION
# ==============================================================================
def batch_read_ukb_fields(field_list, base_path):
    """
    Batch reads specified UKB field files and merges them by patient ID (eid).
    Automatically handles different UKB file naming suffixes.
    """
    df_list = []
    print(f"[*] Starting extraction process for {len(field_list)} fields...")

    for i, field_id in enumerate(field_list):
        # Define three possible file naming conventions in the UKB dataset
        path_i0 = os.path.join(base_path, f"participant.p{field_id}_i0.txt")
        path_i0_a0 = os.path.join(base_path, f"participant.p{field_id}_i0_a0.txt")
        path_static = os.path.join(base_path, f"participant.p{field_id}.txt")
        
        # Check which file variant exists
        target_path = None
        if os.path.exists(path_i0):
            target_path = path_i0
        elif os.path.exists(path_i0_a0):
            target_path = path_i0_a0
        elif os.path.exists(path_static):
            target_path = path_static
        
        # Skip if no corresponding file is found
        if target_path is None:
            print(f"  [!] [{i+1}/{len(field_list)}] ID {field_id}: File not found. Skipped.")
            continue
            
        try:
            # Read the file. usecols=[0, 1] drastically reduces memory consumption
            # by loading only the ID column and the primary value column.
            temp_df = pd.read_csv(target_path, sep=None, engine='python', usecols=[0, 1])
            
            cols = temp_df.columns.tolist()
            if len(cols) < 2:
                print(f"  [!] [{i+1}/{len(field_list)}] ID {field_id}: Insufficient columns (<2). Skipped.")
                continue

            curr_id_col = cols[0]
            curr_val_col = cols[1]
            
            # Standardize column names: 'eid' for index, field_id for the feature values
            temp_df.rename(columns={curr_id_col: 'eid', curr_val_col: field_id}, inplace=True)
            
            # Set 'eid' as index to facilitate horizontal concatenation later
            temp_df.set_index('eid', inplace=True)
            
            # Remove duplicated indices just in case (keep the first occurrence)
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            
            df_list.append(temp_df)
            
            file_name = os.path.basename(target_path)
            print(f"  [+] [{i+1}/{len(field_list)}] Successfully loaded: {field_id} (from: {file_name})")
            
        except Exception as e:
            print(f"  [x] [{i+1}/{len(field_list)}] Failed to load ID {field_id}: {e}")

    # Concatenate all dataframes horizontally using outer join
    if df_list:
        print("\n[*] Merging all extracted dataframes...")
        final_df = pd.concat(df_list, axis=1, join='outer')
        return final_df.reset_index()
    else:
        return pd.DataFrame()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # STEP 1: EXTRACTION
    # ---------------------------------------------------------
    print("=" * 60)
    print("STEP 1: UKB FEATURE EXTRACTION")
    print("=" * 60)
    
    if not os.path.exists(OUT_PATH):
        try:
            os.makedirs(OUT_PATH)
        except Exception as e:
            print(f"Failed to create output directory: {e}")

    # Execute extraction
    mydf = batch_read_ukb_fields(target_field_ids, DATA_DIR)

    # Save raw extracted data
    if not mydf.empty:
        raw_save_path = os.path.join(OUT_PATH, OUT_FILE_RAW)
        mydf.to_csv(raw_save_path, index=False)
        print(f"\n[+] Extraction Complete! Data shape: {mydf.shape}")
        print(f"[+] Raw data saved to: {raw_save_path}")
    else:
        print("[-] Failed to read any data. Please check paths and IDs.")
        exit()


    # ---------------------------------------------------------
    # STEP 2: IMPUTATION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: CLINICAL DATA IMPUTATION")
    print("=" * 60)

    # Load the newly extracted raw data
    df = pd.read_csv(raw_save_path)

    print("[*] Original data shape:", df.shape)
    
    # 1. Standardize the Gender/Sex column ('31' in UKB)
    gender_col = '31' 

    # Convert gender column to numeric (coerce errors to NaN)
    if df[gender_col].dtype not in ['int64', 'float64']:
        df[gender_col] = pd.to_numeric(df[gender_col], errors='coerce')

    # 2. Impute missing values in the gender column using the mode
    if df[gender_col].isnull().any():
        gender_mode = df[gender_col].mode()[0]
        df[gender_col].fillna(gender_mode, inplace=True)

    # 3. Create boolean masks for males and females based on encoding
    # Note: Assuming 1 = Male, 0 = Female based on the original logic
    male_mask = df[gender_col] == 1  
    female_mask = df[gender_col] == 0 

    # 4. Perform customized imputation for specific UKB fields
    # ---------------------------------------------------------
    # Field 30800: Sex-specific fixed values imputation
    if '30800' in df.columns:
        df.loc[male_mask & df['30800'].isnull(), '30800'] = 36.71
        df.loc[female_mask & df['30800'].isnull(), '30800'] = 110.13
        print("[+] Field 30800: Imputed Male=36.71, Female=110.13")

    # Field 30820: Fill all missing values with 0
    if '30820' in df.columns:
        missing_count = df['30820'].isnull().sum()
        df['30820'].fillna(0, inplace=True)
        print(f"[+] Field 30820: Imputed {missing_count} missing values with 0")

    # Field 30850: Sex-specific median imputation
    if '30850' in df.columns:
        male_median_30850 = df.loc[male_mask, '30850'].median()
        female_median_30850 = df.loc[female_mask, '30850'].median()
        
        male_missing = (male_mask & df['30850'].isnull()).sum()
        female_missing = (female_mask & df['30850'].isnull()).sum()
        
        df.loc[male_mask & df['30850'].isnull(), '30850'] = male_median_30850
        df.loc[female_mask & df['30850'].isnull(), '30850'] = female_median_30850
        
        print(f"[+] Field 30850: Imputed {male_missing} males (median={male_median_30850:.2f}), "
              f"{female_missing} females (median={female_median_30850:.2f})")

    # 5. General Imputation for remaining numeric columns (Global Median)
    # ---------------------------------------------------------
    excluded_cols = ['30800', '30820', '30850', gender_col]
    if 'eid' in df.columns:
        excluded_cols.append('eid') 

    # Identify remaining numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [col for col in numeric_cols if col not in excluded_cols]

    print(f"\n[*] Starting global median imputation for {len(cols_to_impute)} numeric columns...")

    for i, col in enumerate(cols_to_impute, 1):
        overall_median = df[col].median()
        df[col].fillna(overall_median, inplace=True)
        
        # Print progress every 10 columns to reduce clutter
        if i % 10 == 0 or i == len(cols_to_impute):
            print(f"  -> Processed {i}/{len(cols_to_impute)} numeric columns")

    # 6. General Imputation for non-numeric categorical columns (Mode)
    # ---------------------------------------------------------
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if gender_col in non_numeric_cols:
        non_numeric_cols.remove(gender_col)

    if non_numeric_cols:
        print(f"\n[*] Starting mode imputation for {len(non_numeric_cols)} categorical columns...")
        for col in non_numeric_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)

    # 7. Verification and Saving
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("IMPUTATION COMPLETED!")
    print("=" * 60)
    
    missing_after = df.isnull().sum()
    missing_cols = missing_after[missing_after > 0]
    
    if len(missing_cols) > 0:
        print("[-] Warning: Some columns still contain missing values:")
        print(missing_cols)
    else:
        print("[+] Success: All missing values have been successfully imputed!")

    # Save the final imputed data
    final_output_path = os.path.join(OUT_PATH, OUT_FILE_IMPUTED)
    df.to_csv(final_output_path, index=False)
    
    print(f"\n[+] Imputed data saved to: {final_output_path}")
    print(f"[+] Final data shape: {df.shape}")