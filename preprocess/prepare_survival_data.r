# ==============================================================================
# File: prepare_survival_data.R
# Description: 
#     UK Biobank survival analysis data preparation script.
#     This script is responsible for:
#     1. Extracting specific disease incidence flags and dates from ICD codes.
#     2. Merging disease phenotype data with baseline assessment dates.
#     3. Loading imputed Olink proteomics data and applying Inverse Rank Normal 
#        Transformation (INT) to handle skewed protein distributions.
#     4. Filtering out prevalent cases (and cases within the first 6 months).
#     5. Calculating precise follow-up time (in years), adjusting for right-censoring 
#        based on study end dates (5-year/10-year) and the death registry.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------------------------
library(data.table)  # Fast data manipulation and file reading (fread)
library(dplyr)       # Data manipulation and piping (%>%)
library(tidyr)       # Data tidying
library(caret)       # Machine learning utilities
library(survival)    # Survival analysis tools
library(RNOmni)      # Rank Normal Transformation (RankNorm)
library(glmnet)      # Regularized generalized linear models
library(ROSE)        # Random Over-Sampling Examples (class imbalance)

# ------------------------------------------------------------------------------
# 2. Global Configuration Parameters
# ------------------------------------------------------------------------------
inc.yrs <- 10                             # Follow-up threshold in years (e.g., 5 or 10)
target_disease <- "Hypertension"          # Target disease name
dz <- paste0("flag_", target_disease)     # Target event flag column name
dz.date <- paste0("diagnosis_date_", target_disease) # Target diagnosis date column name

# ------------------------------------------------------------------------------
# 3. Load Main Disease Wide Table
# ------------------------------------------------------------------------------
print("[*] Loading the comprehensive disease wide-format dataset...")
disease_wide <- fread('/bigdat2/user/xuln/disease_wide_all.csv')

# ------------------------------------------------------------------------------
# 4. Define Disease Diagnosis Extraction Function
# ------------------------------------------------------------------------------
add_diagnosis_flag_and_date <- function(data, icd_code, new_dz_name = "") {
  # If no specific disease name is provided, use the concatenated ICD codes
  if (is.null(new_dz_name) || new_dz_name == "") {
    new_dz_name <- paste(icd_code, collapse = "_")
  }
  
  # Construct standardized column names for the output
  date_col_name <- paste0("diagnosis_date_", new_dz_name)
  flag_col_name <- paste0("flag_", new_dz_name)
  
  # Ensure the input data is a data.table for efficient by-reference operations
  if (!data.table::is.data.table(data)) {
    data <- data.table::as.data.table(data)
  }
  
  # Search for date and flag columns matching the provided ICD codes
  date_pattern <- paste0("^diagnosis_date_(", paste(icd_code, collapse = "|"), ")")
  flag_pattern <- paste0("^flag_(", paste(icd_code, collapse = "|"), ")")
  
  date_cols <- names(data)[grepl(date_pattern, names(data))]
  flag_cols <- names(data)[grepl(flag_pattern, names(data))]
  
  if (length(date_cols) == 0 && length(flag_cols) == 0) {
    warning("No diagnosis date or flag columns found for ICD codes: ", paste(icd_code, collapse = ", "))
    return(data)
  }
  
  # 4.1 Create the comprehensive Event Flag column
  if (length(flag_cols) > 0) {
    # If any of the target ICD flags > 0, the participant is considered a case
    data[, (flag_col_name) := as.integer(rowSums(.SD, na.rm = TRUE) > 0), .SDcols = flag_cols]
  } else {
    data[, (flag_col_name) := 0L]
  }
  
  # 4.2 Process and extract the earliest Diagnosis Date
  if (length(date_cols) > 0) {
    # Create a temporary subset for melting
    temp_data <- data[, .SD, .SDcols = c("participant.eid", date_cols)]
    
    # Convert from wide to long format to easily find the earliest date
    diagnosis_dates_long <- data.table::melt(
      temp_data,
      id.vars = "participant.eid",
      variable.name = "source",
      value.name = "temp_date"
    )
    
    # Filter out invalid or missing dates (1970-01-01 is often a default zero-date in R)
    diagnosis_dates_long <- diagnosis_dates_long[
      !is.na(temp_date) & temp_date != as.IDate("1970-01-01")
    ]
    
    # If valid dates exist, extract the earliest (minimum) date per participant
    if (nrow(diagnosis_dates_long) > 0) {
      min_dates <- diagnosis_dates_long[
        order(participant.eid, temp_date)
      ][
        , .(min_date = min(temp_date)), by = participant.eid
      ]
      
      data.table::setnames(min_dates, "min_date", date_col_name)
      # Join the earliest date back to the main data table
      data[min_dates, (date_col_name) := get(date_col_name), on = "participant.eid"]
    } else {
      data[, (date_col_name) := as.IDate(NA)]
    }
  } else {
    data[, (date_col_name) := as.IDate(NA)]
  }
  
  return(data)
}

# Apply the function to extract Hypertension cases (ICD-10 code "I10")
print("[*] Extracting Hypertension diagnosis flags and dates...")
disease_wide <- add_diagnosis_flag_and_date(disease_wide, c("I10"), new_dz_name = target_disease)

# ------------------------------------------------------------------------------
# 5. Load Baseline Assessment Dates & Merge
# ------------------------------------------------------------------------------
print("[*] Loading baseline assessment dates and merging phenotypes...")
baseline_assessment <- fread("/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p53_i0.txt")
pheno <- merge(x = baseline_assessment, y = disease_wide, by = "participant.eid", all.y = TRUE)

# Rename the baseline date column (UKB Field 53) for readability
colnames(pheno)[which(colnames(pheno)=="participant.p53_i0")] <- "date_baseline_assessment"

# ------------------------------------------------------------------------------
# 6. Load Proteomics Data and Apply Rank Normalization
# ------------------------------------------------------------------------------
print("[*] Loading imputed Olink proteomics data...")
# Load the cleaned protein NPX dataset (45,066 samples after removing 94 outliers)
imputed_protein <- read.csv("/bigdat2/user/xuln/Imputed_protein_cleanoutlier.csv")

# Extract the list of protein column names
protein_list <- imputed_protein %>% select(-participant.eid) %>% names()

# Merge clinical phenotypes with protein expression data
ukbb <- merge(pheno, imputed_protein, by='participant.eid', all.y = TRUE)

print("[*] Applying Inverse Rank Normalization (INT) to all proteins...")
# Inverse rank normal transform NPX values to enforce a standard normal distribution,
# which helps linear models and neural networks converge better.
ukbb[,protein_list] <- sapply(ukbb[,protein_list], RankNorm)

# ------------------------------------------------------------------------------
# 7. Calculate Follow-up Time & Filter Prevalent Cases
# ------------------------------------------------------------------------------
print("[*] Calculating follow-up time and filtering prevalent/early-incident cases...")
ukbb <- ukbb %>% 
  # Calculate time difference between diagnosis date and baseline assessment in weeks, then convert to years
  mutate(fol = difftime(!!as.name(dz.date), date_baseline_assessment, units = "weeks") / 52.25) %>% 
  # EXCLUSION CRITERIA:
  # - Exclude prevalent cases (fol < 0)
  # - Exclude cases occurring within the first 6 months (fol < 0.5) to avoid reverse causality
  # - Exclude cases occurring after the specific follow-up window (fol > inc.yrs)
  # - Keep healthy controls (where diagnosis date is NA)
  filter((fol > 0.5 & fol < inc.yrs) | is.na(!!as.name(dz.date)))

# ------------------------------------------------------------------------------
# 8. Administrative Censoring (Study End Date)
# ------------------------------------------------------------------------------
print(paste("[*] Applying administrative censoring at", inc.yrs, "years..."))
if(inc.yrs == 10){
  ukbb <- ukbb %>%
    # Sanity check: exclude cases with event=1 but missing diagnosis dates
    filter(!(is.na(!!as.name(dz.date)) & !!as.name(dz) == 1)) %>% 
    # For healthy controls (Event=0), follow-up ends at the study end date (Dec 31, 2020)
    mutate(fol = ifelse(!!as.name(dz) == 0, 
                        difftime(as.Date("2020-12-31"), date_baseline_assessment, units = "weeks") / 52.25,
                        fol))
} else if(inc.yrs == 5){
  ukbb <- ukbb %>%
    filter(!(is.na(!!as.name(dz.date)) & !!as.name(dz) == 1)) %>% 
    mutate(fol = ifelse(!!as.name(dz) == 0, 
                        difftime(as.Date("2016-05-31"), date_baseline_assessment, units = "weeks") / 52.25,
                        fol))
} else {
  stop("Follow-up threshold (inc.yrs) must be explicitly defined (e.g., 5 or 10).")
}

# ------------------------------------------------------------------------------
# 9. Death Censoring
# ------------------------------------------------------------------------------
print("[*] Loading death registry and adjusting follow-up times...")
death.data <- fread("/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/all_pheno/participant.p40000_i0.txt")
death.data$date_of_death <- as.Date(death.data$participant.p40000_i0, format = "%Y-%m-%d")

# Merge death dates
ukbb <- merge(ukbb, death.data[,c("participant.eid", "date_of_death")], by="participant.eid", all.x=TRUE)

# Adjust follow-up time for patients who died before the study end date
# If a control patient (Event=0) died before the study end date, their follow-up is truncated to their death date
if(inc.yrs == 10){
  ukbb$fol_d <- ifelse(ukbb[, ..dz] == 0 & ukbb$date_of_death < as.Date("2020-12-31"),
                       difftime(ukbb$date_of_death, ukbb$date_baseline_assessment, units = "weeks") / 52.25,
                       ukbb$fol)
} else if(inc.yrs == 5){
  ukbb$fol_d <- ifelse(ukbb[, ..dz] == 0 & ukbb$date_of_death < as.Date("2016-05-31"),
                       difftime(ukbb$date_of_death, ukbb$date_baseline_assessment, units = "weeks") / 52.25,
                       ukbb$fol)
}

# Finalize the follow-up column (use death-adjusted follow-up if applicable)
ukbb$fol <- ifelse(is.na(ukbb$fol_d), ukbb$fol, ukbb$fol_d)

# ------------------------------------------------------------------------------
# 10. Format and Finalize Dataset
# ------------------------------------------------------------------------------
columns_to_keep <- c("participant.eid", dz, "fol", protein_list)

# Select target columns
selected_data <- ukbb[, ..columns_to_keep]

# Standardize target variable names for downstream ML tasks
setnames(selected_data, old = c(dz, "fol"), new = c("Event", "Time"))

print("[+] Data Preparation Complete! Preview of the final dataset:")
print(head(selected_data[, 1:5]))

# Optional: Export the final processed dataset to CSV
# write.csv(selected_data, file = paste0("/bigdat2/user/xuln/save_disease_data/", target_disease, ".csv"), row.names = FALSE)