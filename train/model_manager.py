"""
==============================================================================
File: model_manager.py
Description: 
    Disease Model Manager for ProPathNet.
    This module handles the automated routing of disease predictions. It ensures
    that single-sex diseases are properly mapped to their '_filtered' variants,
    retrieves optimal hyperparameters from the registry (CSV), and dynamically 
    locates the correct pre-trained `.pth` model weights.
==============================================================================
"""

import os
import glob
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DiseaseModelManager:
    def __init__(self, 
                 param_csv_path="result/disease_best_parameters.csv",
                 backup_base_path="best_models"):
        """
        Automated disease model registration and routing manager 
        (supports mandatory filtering/redirection for single-sex diseases).
        
        Args:
            param_csv_path (str): Path to the CSV registry containing optimal hyperparameters.
            backup_base_path (str): Root directory where the pre-trained `.pth` models are stored.
        """
        self.param_csv_path = param_csv_path
        self.backup_base_path = backup_base_path
        
        # Define a list of base names for single-sex diseases
        # (Diseases in this list MUST be forced to use the '_filtered' version to avoid sex-bias)
        self.single_sex_bases = [
            "Primary_Malignancy_Prostate",
            "Benign_neoplasm_and_polyp_of_uterus",
            "Leiomyoma_of_uterus",
            "Female_genital_prolapse",
            "Hyperplasia_of_prostate",
            "Menorrhagia_and_polymenorrhoea",
            "Postmenopausal_bleeding"
        ]
        
        # 1. Load the optimal hyperparameters database (Registry)
        if os.path.exists(self.param_csv_path):
            self.params_df = pd.read_csv(self.param_csv_path)
            # Set the disease name as the index for efficient direct retrieval
            self.params_df.set_index('Disease_Name', inplace=True)
            print(f"[*] Model registry loaded successfully. Total disease parameter records: {len(self.params_df)}")
        else:
            raise FileNotFoundError(f"[Error] Optimal disease parameters file not found: {self.param_csv_path}")

    def get_disease_info(self, disease_name):
        """
        Core routing function: Inputs a disease name and dynamically returns 
        the absolute path to its optimal model and complete hyperparameter configuration.
        
        Args:
            disease_name (str): The target disease name to query.
            
        Returns:
            dict: Contains the parsed disease_name, category, model_path, and config dictionary.
        """
        # ==========================================
        # Core Interception Logic: 
        # Mandatory redirection to '_filtered' for single-sex diseases
        # ==========================================
        
        # 1. If the user inputs the base name, automatically append the suffix
        if disease_name in self.single_sex_bases:
            filtered_name = f"{disease_name}_filtered"
            print(f"[*] Interception triggered: '{disease_name}' is a single-sex disease, forcibly redirected to -> '{filtered_name}'")
            disease_name = filtered_name
            
        # 2. If the user input already contains '_filtered' but is not in our known list, allow it safely with a warning
        elif disease_name.endswith("_filtered"):
            base_name = disease_name.replace("_filtered", "")
            if base_name not in self.single_sex_bases:
                print(f"[!] Warning: '{disease_name}' contains the '_filtered' suffix but is not in the known list of single-sex diseases.")

        # ==========================================
        
        # 2. Retrieve the category and hyperparameters from the CSV database
        if disease_name not in self.params_df.index:
            raise ValueError(f"[Error] Disease name '{disease_name}' is not registered in the parameter CSV database. Please check the spelling.")
            
        row_data = self.params_df.loc[disease_name]
        
        # If there are duplicate rows for the same disease, take the first one
        if isinstance(row_data, pd.DataFrame):
            row_data = row_data.iloc[0]
            
        category = row_data['Category']
        
        # 3. Convert the entire parameter row into a standard Python dictionary and remove metadata columns
        config_dict = row_data.to_dict()
        config_dict.pop('Category', None)

        # 4. Dynamically locate the model weights file in the backup directory
        # Path rule: backup_base_path / category / disease_name / *.pth
        target_folder = os.path.join(self.backup_base_path, category, disease_name)
        
        if not os.path.exists(target_folder):
            raise FileNotFoundError(f"[Error] Backup directory for the disease not found: {target_folder}")
            
        # Fuzzy match to find any .pth file within the target folder
        pth_files = glob.glob(os.path.join(target_folder, "*.pth"))
        
        if not pth_files:
            raise FileNotFoundError(f"[Error] No .pth model weights found in directory: {target_folder}")
            
        # By default, select the first matched optimal weight file
        best_model_path = pth_files[0]

        return {
            'disease_name': disease_name,  # This returns the correctly formatted name (with _filtered if applicable)
            'category': category,
            'model_path': best_model_path,
            'config': config_dict
        }