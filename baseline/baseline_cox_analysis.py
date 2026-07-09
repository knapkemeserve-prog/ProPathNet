# %%
# Imputed 67 traits dataset
# /bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv

"""
==============================================================================
File: baseline_cox_analysis.py
Description: 
    Cox Proportional Hazards Model Analysis Pipeline for 67 Clinical Traits.
    This script is responsible for:
    1. Iterating through multiple disease categories and specific disease cohorts (.h5ad).
    2. Merging survival data with 67 imputed clinical trait features.
    3. Training and evaluating Cox models ('Base' demographics vs. 'Physical_examination' traits).
    4. Automatically searching for optimal penalizer terms if the model fails to converge 
       or performs poorly (C-index < 0.5).
    5. Saving individual test set risk predictions and generating comprehensive summary logs.
==============================================================================
"""

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# PART 1: Configuration Parameters
# ==============================================================================
# Parent directory path - Contains all disease category subdirectories
PARENT_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"

# Path to the imputed 67 traits dataset
PANEL_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"

# List of disease categories to process
DISEASE_CATEGORIES = [
    'Benign_neoplasm_or_Carcinoma_in_situ',
    'Cancers',
    'Cardiovascular',
    'Digestive',
    'Endocrine',
    'Eye',
    'Genitourinary',
    'Haematological_or_immunological',
    'Infections',
    'Musculoskeletal',
    'Neurological',
    'Psychiatric',
    'Respiratory',
    'Skin',
    'Ear'
]

# Output directory for the Cox analysis results
OUTPUT_DIR = "/bigdat2/user/xuln/olink_disease_predict/67traits_cox_analysis"

# Set random seed for reproducibility in dataset splitting
RANDOM_SEED = 42

# ==============================================================================
# PART 2: Feature Definitions
# ==============================================================================
# Sex feature indicator in UKB
categorical_features = ['31']

# Complete list of continuous trait variables
continuous_features = [
    '48', '49', '50', '74', '102', '4079', '4080', '20150', '20151', '20258',
    '21001', '21003', '30000', '30010', '30020', '30030', '30040', '30050', 
    '30060', '30080', '30100', '30120', '30130', '30140', '30150', '30160', 
    '30170', '30250', '30260', '30270', '30280', '30300', '30500', '30510', 
    '30520', '30530', '30600', '30610', '30620', '30630', '30640', '30650', 
    '30660', '30670', '30680', '30690', '30700', '30710', '30720', '30730', 
    '30740', '30750', '30760', '30770', '30780', '30790', '30800', '30810', 
    '30820', '30830', '30840', '30850', '30860', '30870', '30880', '30890'
]

# Model Configurations
# Base model: Age (21003) and Sex (31) only
Base_features = ['21003', '31']  

# Physical examination model: Base features + all 67 blood/urine/physical traits
Physical_examination_features = [
    '31', '48', '49', '50', '74', '102', '4079', '4080', '20150', '20151', 
    '20258', '21001', '21003', '30000', '30010', '30020', '30030', '30040', 
    '30050', '30060', '30080', '30100', '30120', '30130', '30140', '30150', 
    '30160', '30170', '30250', '30260', '30270', '30280', '30300', '30500', 
    '30510', '30520', '30530', '30600', '30610', '30620', '30630', '30640', 
    '30650', '30660', '30670', '30680', '30690', '30700', '30710', '30720', 
    '30730', '30740', '30750', '30760', '30770', '30780', '30790', '30800', 
    '30810', '30820', '30830', '30840', '30850', '30860', '30870', '30880', 
    '30890'
]

model_dict = {
    'Base': Base_features,
    'Physical_examination': Physical_examination_features
}

# Initial regularizer penalties assigned to each model
model_penalizers = {
    'Base': 0.0,                    # Standard unpenalized Cox for base features
    'Physical_examination': 0.01    # Light initial penalty for high-dimensional clinical traits
}

# Backup penalizers to search through if the model performs poorly (Val C-index < 0.5)
penalty_search_list = [0.01, 0.05, 0.1]

# ==============================================================================
# PART 3: Core Functions
# ==============================================================================

def save_test_predictions(model, model_name, test_df, feature_list, 
                          disease_name, category_name, output_dir):
    """
    Computes and saves the predicted risk scores for the test set into a CSV file.
    """
    # Create category and disease specific directories
    category_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_dir, exist_ok=True)
    
    disease_dir = os.path.join(category_dir, disease_name)
    os.makedirs(disease_dir, exist_ok=True)
    
    # Prepare the test dataset (Drop missing values to ensure prediction works)
    cols = feature_list + ['time', 'event']
    df_test_sub = test_df[cols].dropna().copy()
    
    if len(df_test_sub) == 0:
        print(f"    [!] Warning: Test set is empty after dropping missing values. Cannot generate predictions.")
        return None
    
    try:
        # Calculate the partial hazard (risk score)
        risk_scores = model.predict_partial_hazard(df_test_sub).values.flatten()
        
        # Construct the output DataFrame
        predictions_df = pd.DataFrame({
            'participant.eid': df_test_sub.index,
            'time': df_test_sub['time'].values,
            'event': df_test_sub['event'].values,
            'risk_score': risk_scores
        })
        
        # Sort by risk score in descending order
        predictions_df = predictions_df.sort_values('risk_score', ascending=False)
        
        # Save to CSV
        output_file = os.path.join(disease_dir, f"{model_name}_predictions.csv")
        predictions_df.to_csv(output_file, index=False)
        
        print(f"    [+] Test predictions saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"    [x] Error saving predictions: {e}")
        return None

def train_eval_cox_with_penalty_search(model_name, feature_list, train_df, val_df, test_df, 
                                       disease_name, category_name, output_dir):
    """
    Trains and evaluates the Cox model.
    Dynamically searches for the best L2 penalizer term if the validation C-index < 0.5.
    Also computes and saves test set risk predictions.
    """
    cols = feature_list + ['time', 'event']
    
    # Drop rows with missing values
    df_train_sub = train_df[cols].dropna().copy()
    df_val_sub = val_df[cols].dropna().copy()
    
    if len(df_train_sub) == 0:
        print(f"    [!] Training set is empty after dropping missing values. Skipping model.")
        return None, None
    
    # Get the default penalizer for this model type
    initial_penalizer = model_penalizers.get(model_name, 0.01)
    
    # Construct the search list prioritizing the initial penalizer
    penalizers_to_try = [initial_penalizer] + [p for p in penalty_search_list if p != initial_penalizer]
    
    best_model = None
    best_val_cindex = -np.inf
    best_penalizer = initial_penalizer
    
    print(f"    [*] Model '{model_name}' - Attempting penalizers: {penalizers_to_try}")
    
    for penalizer in penalizers_to_try:
        try:
            # Initialize and fit the Cox Proportional Hazards model
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df_train_sub, duration_col='time', event_col='event')
            
            # Evaluate on validation set
            c_val = cph.score(df_val_sub, scoring_method="concordance_index")
            
            print(f"        -> Penalizer {penalizer:.3f} | Val C-index: {c_val:.4f}")
            
            # If C-index is extremely poor (<0.5 indicates worse than random guessing), keep searching
            if c_val < 0.5:
                if c_val > best_val_cindex:
                    best_val_cindex = c_val
                    best_model = cph
                    best_penalizer = penalizer
                continue
            
            # If C-index >= 0.5, we accept this model and terminate the search
            best_model = cph
            best_val_cindex = c_val
            best_penalizer = penalizer
            break
            
        except Exception as e:
            print(f"        [x] Fitting failed for penalizer {penalizer:.3f}: {e}")
            continue
    
    # Handle edge case where all penalizers fail to yield a working model
    if best_model is None:
        print(f"    [!] All penalizers failed. Skipping model '{model_name}'.")
        return None, None
    
    print(f"    [+] Selected Penalizer {best_penalizer:.3f} | Best Val C-index: {best_val_cindex:.4f}")
    
    # Evaluate performance on the training set
    c_train = best_model.score(df_train_sub, scoring_method="concordance_index")
    
    # Evaluate performance on the test set
    df_test_sub = test_df[cols].copy()
    df_test_sub_clean = df_test_sub.dropna()
    
    if len(df_test_sub_clean) > 0:
        risk_scores = best_model.predict_partial_hazard(df_test_sub_clean).values.flatten()
        
        # Calculate test C-index (negative risk_scores are passed as lifelines expects 'actual survival times' or negative hazards)
        c_test = concordance_index(
            df_test_sub_clean['time'].values,
            -risk_scores,  
            df_test_sub_clean['event'].values
        )
        
        # Save prediction results
        save_test_predictions(
            best_model, model_name, test_df, feature_list,
            disease_name, category_name, output_dir
        )
    else:
        print(f"    [!] Test set is empty after dropping NA. Cannot compute Test C-index.")
        c_test = np.nan
    
    # Return model object and evaluation dictionary
    return best_model, {
        'Model': model_name, 
        'Penalizer': best_penalizer,
        'Train C-index': c_train, 
        'Val C-index': best_val_cindex, 
        'Test C-index': c_test
    }

def process_disease_category(category_name, base_path, panel_path, model_dict, output_dir):
    """
    Processes all diseases (.h5ad cohorts) within a specific disease category directory.
    """
    print(f"\n{'='*80}")
    print(f"[*] Starting processing for category: {category_name}")
    print(f"[*] Directory path: {base_path}")
    print(f"{'='*80}")
    
    # Scan for all .h5ad files in the category directory
    h5ad_files = glob.glob(os.path.join(base_path, "*.h5ad"))
    
    if not h5ad_files:
        print(f"[!] No .h5ad files found in {base_path}. Skipping this category.")
        return []
    
    print(f"[*] Found {len(h5ad_files)} .h5ad cohort files:")
    for file in h5ad_files[:10]:  
        print(f"  - {os.path.basename(file)}")
    if len(h5ad_files) > 10:
        print(f"  ... and {len(h5ad_files)-10} more files.")
    
    all_results = []
    successful_diseases = 0
    
    # Process each disease cohort
    for h5ad_file in h5ad_files:
        disease_name = os.path.splitext(os.path.basename(h5ad_file))[0]
        file_name = os.path.basename(h5ad_file)
        
        print(f"\n{'-'*60}")
        print(f"[*] Processing Disease: {disease_name}")
        print(f"[*] Target File: {file_name}")
        print(f"{'-'*60}")
        
        try:
            # 1. Load the survival data cohort
            print(f"    [-] Loading data: {h5ad_file}")
            adata = sc.read_h5ad(h5ad_file)
            print(f"    [-] Cohort shape: {adata.shape}")
            
            # Check for sufficient positive events
            event_counts = adata.obs['event'].value_counts()
            print(f"    [-] Event distribution: {event_counts.to_dict()}")
            
            if 1 not in event_counts or event_counts[1] < 5:
                print(f"    [!] Insufficient events ({event_counts.get(1, 0)} events). Skipping this disease.")
                continue
            
            # 2. Extract survival metrics and IDs
            survival_time = adata.obs['time'].values
            event_status = adata.obs['event'].values
            patient_ids = adata.obs.index.tolist()
            
            # 3. Stratified Train-Validation-Test Split
            indices = list(range(len(patient_ids)))
            train_val_idx, test_idx = train_test_split(
                indices, test_size=0.2, stratify=event_status, random_state=RANDOM_SEED
            )
            train_val_events = event_status[train_val_idx]
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=0.125, stratify=train_val_events, random_state=RANDOM_SEED
            )
            
            print(f"    [-] Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
            
            train_data = adata.obs.iloc[train_idx].copy()
            val_data = adata.obs.iloc[val_idx].copy()
            test_data = adata.obs.iloc[test_idx].copy()
            
            # 4. Load Clinical Traits Data and Merge
            print("    [-] Loading and merging clinical traits data...")
            Clinical = pd.read_csv(panel_path)
            Clinical['eid'] = Clinical['eid'].astype(str)
            
            train_data['eid'] = train_data.index.astype(str)
            val_data['eid'] = val_data.index.astype(str)
            test_data['eid'] = test_data.index.astype(str)
            
            train_data_merged = pd.merge(train_data, Clinical, on='eid', how='left').set_index('eid')
            val_data_merged = pd.merge(val_data, Clinical, on='eid', how='left').set_index('eid')
            test_data_merged = pd.merge(test_data, Clinical, on='eid', how='left').set_index('eid')
            
            print(f"    [-] Merged shapes: Train={train_data_merged.shape}, Val={val_data_merged.shape}, Test={test_data_merged.shape}")
            
            # 5. Data Standardization
            print("    [-] Standardizing continuous features...")
            exist_cont_features = [col for col in continuous_features if col in train_data_merged.columns]
            
            # Fit scaler only on training data to prevent data leakage
            scaler = StandardScaler()
            scaler.fit(train_data_merged[exist_cont_features])
            
            train_data_scaled = train_data_merged.copy()
            val_data_scaled = val_data_merged.copy()
            test_data_scaled = test_data_merged.copy()
            
            train_data_scaled[exist_cont_features] = scaler.transform(train_data_merged[exist_cont_features])
            val_data_scaled[exist_cont_features] = scaler.transform(val_data_merged[exist_cont_features])
            test_data_scaled[exist_cont_features] = scaler.transform(test_data_merged[exist_cont_features])
            
            # 6. Train and Evaluate Models
            disease_results = []
            for name, features in model_dict.items():
                valid_features = [f for f in features if f in train_data_scaled.columns]
                print(f"\n    [*] Training Model: {name} (Using {len(valid_features)} features)")
                
                model, res = train_eval_cox_with_penalty_search(
                    name, valid_features, train_data_scaled, val_data_scaled, test_data_scaled,
                    disease_name, category_name, output_dir
                )
                
                if res:
                    res['Disease'] = disease_name
                    res['Category'] = category_name
                    disease_results.append(res)
            
            # Append successful results
            if disease_results:
                all_results.extend(disease_results)
                successful_diseases += 1
                print(f"    [+] Disease '{disease_name}' processed successfully. Generated {len(disease_results)} model results.")
            else:
                print(f"    [!] Disease '{disease_name}' did not generate any valid model results.")
            
        except Exception as e:
            print(f"    [x] Error processing disease {disease_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save a separate summary file for this specific disease category
    if all_results:
        category_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        columns_order = ['Category', 'Disease', 'Model', 'Penalizer', 'Train C-index', 'Val C-index', 'Test C-index']
        existing_cols = [c for c in columns_order if c in category_df.columns]
        category_df = category_df[existing_cols]
        
        # Sort by Model name and Test C-index descending
        category_df = category_df.sort_values(['Model', 'Test C-index'], ascending=[True, False])
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to general output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(output_dir, f"{category_name}_cindex_summary_{timestamp}.csv")
        category_df.to_csv(csv_filename, index=False)
        
        print(f"\n[+] Category '{category_name}' processing complete!")
        print(f"    - Successfully processed diseases: {successful_diseases}/{len(h5ad_files)}")
        print(f"    - Total rows generated: {len(all_results)}")
        print(f"    - Summary CSV saved to: {csv_filename}")
        
        # Also save a clean copy inside the specific category folder
        category_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)
        category_summary_file = os.path.join(category_dir, f"{category_name}_cindex_summary.csv")
        category_df.to_csv(category_summary_file, index=False)
        print(f"    - Local directory summary saved to: {category_summary_file}")
    else:
        print(f"\n[!] Category '{category_name}' yielded no results.")
    
    return all_results

# ==============================================================================
# PART 4: Main Execution Block
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("🚀 Initializing Cross-Category Disease Data Processing Pipeline")
    print(f"[*] Parent Directory: {PARENT_PATH}")
    print(f"[*] Target Categories: {DISEASE_CATEGORIES}")
    print(f"[*] Output Directory: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    print(f"[*] Analysis Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_categories = len(DISEASE_CATEGORIES)
    all_results = []
    successful_categories = []
    failed_categories = []
    
    # Initialize the main progress log file
    log_file = os.path.join(OUTPUT_DIR, "analysis_progress.log")
    with open(log_file, 'w') as f:
        f.write(f"--- Disease Category Analysis Progress Log ---\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Categories to Process: {total_categories}\n")
        f.write(f"{'='*50}\n")
    
    # Iterate through all configured disease categories
    for i, category in enumerate(DISEASE_CATEGORIES, 1):
        print(f"\n>>> Progress: {i}/{total_categories} - Processing Category: {category}")
        
        category_path = os.path.join(PARENT_PATH, category)
        
        # Check if the directory actually exists
        if not os.path.exists(category_path):
            error_msg = f"Directory not found: {category_path}"
            print(f"    [!] {error_msg}. Skipping this category.")
            failed_categories.append(category)
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: {error_msg}\n")
            continue
        
        # Execute processing for the category
        try:
            category_results = process_disease_category(category, category_path, PANEL_PATH, model_dict, OUTPUT_DIR)
            
            if category_results:
                all_results.extend(category_results)
                successful_categories.append(category)
                
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: Success. Generated {len(category_results)} results.\n")
            else:
                failed_categories.append(category)
                
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: No valid results generated.\n")
                    
        except Exception as e:
            error_msg = f"Unexpected error while processing {category}: {str(e)}"
            print(f"    [x] {error_msg}")
            failed_categories.append(category)
            
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: Error - {str(e)}\n")
    
    # Record End Time and Duration
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    # Save the global concatenated summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        columns_order = ['Category', 'Disease', 'Model', 'Penalizer', 'Train C-index', 'Val C-index', 'Test C-index']
        existing_cols = [c for c in columns_order if c in results_df.columns]
        results_df = results_df[existing_cols]
        
        # Sort globally by Category -> Disease -> Model
        results_df = results_df.sort_values(['Category', 'Disease', 'Model'])
        
        summary_csv = os.path.join(OUTPUT_DIR, "ALL_DISEASES_CINDEX_SUMMARY.csv")
        results_df.to_csv(summary_csv, index=False)
        
        print(f"\n[+] Global Master Summary saved successfully:")
        print(f"    -> {summary_csv}")
    
    # Print Final Execution Summary
    print(f"\n{'='*80}")
    print("🎉 ALL DISEASE CATEGORIES PROCESSING COMPLETED!")
    print(f"{'='*80}")
    print(f"[*] Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Total Execution Time: {elapsed_time}")
    print(f"\n📊 Execution Statistics:")
    print(f"    - Successfully processed categories: {len(successful_categories)}/{total_categories}")
    
    if successful_categories:
        print(f"    - Successful list: {', '.join(successful_categories)}")
        
    print(f"    - Failed/Skipped categories: {len(failed_categories)}/{total_categories}")
    if failed_categories:
        print(f"    - Failed list: {', '.join(failed_categories)}")
    
    print(f"    - Total result rows generated: {len(all_results)}")
    print(f"    - Master Progress Log: {log_file}")
    print(f"    - Master Output Directory: {OUTPUT_DIR}")
    
    # Save text summary report
    summary_file = os.path.join(OUTPUT_DIR, "ANALYSIS_SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("--- Global Analysis Execution Summary ---\n")
        f.write("="*50 + "\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Execution Time: {elapsed_time}\n")
        f.write(f"Successfully processed categories: {len(successful_categories)}/{total_categories}\n")
        if successful_categories:
            f.write(f"Successful list: {', '.join(successful_categories)}\n")
        f.write(f"Failed/Skipped categories: {len(failed_categories)}/{total_categories}\n")
        if failed_categories:
            f.write(f"Failed list: {', '.join(failed_categories)}\n")
        f.write(f"Total result rows generated: {len(all_results)}\n")
        f.write(f"Progress Log: {log_file}\n")
        f.write(f"Output Directory: {OUTPUT_DIR}\n")
        if all_results:
            f.write(f"Master CSV File: {summary_csv}\n")
    
    print(f"\n[+] Execution text summary saved to: {summary_file}")
    
    # Display a preview of the final results
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n👀 Preview of final master results (First 20 rows):")
        print(results_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()

# ==============================================================================
# Terminal execution command (Background run):
# nohup python -u /home/xuln/olink_disease_predict/code/panel_cox_analysis.py > /home/xuln/olink_disease_predict/code/panel_cox_result_0124.log 2>&1 &
# ==============================================================================