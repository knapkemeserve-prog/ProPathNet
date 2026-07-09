# ==============================================================================
# File Name: r_missforest_Inflammation.r
# Description: 
#     Proteomics data (NPX) missing value imputation script based on missForest.
#     This script precisely imputes a specific protein panel ("Inflammation"):
#     1. Enables multi-core parallel computing to accelerate random forest training.
#     2. Incorporates clinical baseline features (Age, Sex) to assist and improve
#        the prediction accuracy of missing protein expressions.
#     3. Saves the imputed result object as an .RData format for downstream scripts.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
library(missForest)  # Core package: Non-parametric imputation via random forests
library(dplyr)       # Core data processing: Efficient filtering, selection, and piping (%>%)
library(parallel)    # Provides underlying parallel computing support

# ------------------------------------------------------------------------------
# 2. Parallel Computing Configuration
# ------------------------------------------------------------------------------
# Register parallel computing backend, allocating 96 cores.
# Note: Random forest computation is highly time-consuming. Fully utilizing 
# multi-core server performance significantly reduces imputation time.
doParallel::registerDoParallel(cores = 96)
print("[*] Parallel computing environment initiated. Cores allocated: 96")

# ------------------------------------------------------------------------------
# 3. Data Loading
# ------------------------------------------------------------------------------
print("[*] Loading Olink proteomics expression profile and mapping dictionary...")
# data: Contains patient baseline info (Age, Sex) and NPX expression values for proteins
data <- read.csv("/bigdat2/user/xuln/final_combined_participant.csv")

# protein_panel: Dictionary mapping proteins to their respective target panels
protein_panel <- readRDS("/bigdat2/user/xuln/protein_panel.rds")

# ------------------------------------------------------------------------------
# 4. Target Panel Filtering & Feature Extraction
# ------------------------------------------------------------------------------
# Define the target panel to be imputed (can be modified to Cardiometabolic, Oncology, etc.)
panel.in <- "Inflammation" 
print(paste("[*] Currently processing protein panel:", panel.in))

# Extract all target protein column names belonging to the "Inflammation" panel
protein_columns <- protein_panel %>%
  filter(Protein.panel == panel.in) %>%
  pull(Assay.Target)

# Safety check: Intersect extracted proteins with actual column names in 'data'
# This prevents errors caused by version mismatches where a protein exists in 
# the dictionary but is missing in the actual dataset.
valid_protein_columns <- intersect(protein_columns, names(data))
print(paste("[*] Number of successfully matched valid proteins:", length(valid_protein_columns)))

# ------------------------------------------------------------------------------
# 5. Construct Standard Dataset for Random Forest Imputation
# ------------------------------------------------------------------------------
imp.data <- data %>%
  # Retain only: Unique patient ID, Age, Sex, and all valid proteins for this panel
  select(participant.eid, Age, Sex, all_of(valid_protein_columns)) %>%
  # CRITICAL STEP: When Random Forest processes categorical variables, they must be 
  # explicitly converted to 'Factor' type. Otherwise, the model treats Sex (e.g., 0/1) 
  # as a continuous numerical value for regression fitting, causing logical errors.
  mutate(Sex = as.factor(Sex))  

# Set patient IDs as row names to ensure patient identity is preserved post-imputation
row.names(imp.data) <- imp.data$participant.eid

# Remove participant.eid from feature columns since it's merely an identifier.
# (If not removed, the model will uselessly attempt to use IDs to predict protein levels,
# leading to severe overfitting and memory waste.)
imp.data$participant.eid <- NULL

# ------------------------------------------------------------------------------
# 6. Execute Core missForest Imputation
# ------------------------------------------------------------------------------
print("[*] Starting missForest random forest imputation (this may take considerable time)...")

m.forest <- missForest(
  xdata = imp.data,
  verbose = TRUE,         # Enable logging to monitor progress and OOB error changes
  replace = FALSE,        # Do not use sampling with replacement when building decision trees
  parallelize = "forests",# Parallelization strategy: compute different forests in parallel
  ntree = 50,             # Number of decision trees per forest (50 is a reasonable tradeoff for large omics data)
  maxiter = 1             # Maximum iterations. Often set to 1 or 2 for large datasets to control time costs
)

# ------------------------------------------------------------------------------
# 7. Result Persistence and Evaluation
# ------------------------------------------------------------------------------
output_file <- paste0("/bigdat2/user/xuln/Imputed_NPX_missForest_", panel.in, ".RData")
print(paste("[*] Imputation calculation complete! Saving results to:", output_file))

# Save the complete missForest object (includes the imputed data matrix $ximp and model state)
save(m.forest, file = output_file)

# Output Out-Of-Bag (OOB) Error
# OOB error is used to evaluate imputation reliability. 
# For continuous variables (NPX), it outputs NRMSE (Normalized Root Mean Squared Error).
# For categorical variables (Sex), it outputs PFC (Proportion of Falsely Classified).
print("\n[+] Out-Of-Bag (OOB) Error Evaluation:")
print(m.forest$OOBerror)
print("[+] Script execution successfully completed!")