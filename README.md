# ProPathNet

## Overview

ProPathNet is a deep learning framework that integrates multi-omics data for survival analysis and disease risk prediction.
By combining Protein-Protein Interaction (PPI) networks, Pathway-based Visible Neural Networks (VNN), and clinical covariates, ProPathNet achieves precise joint feature importance attribution and high-performance prognostic modeling.

<p align="center">
  <img src="流程图_model.drawio.svg" alt="模型流程图" width="800"/>
</p>

## Environment Setup


### Option A: Using propathnet.yml (Recommended)

This will automatically create a conda environment named propathnet_env with all necessary dependencies.

```
conda env create -f propathnet.yml
conda activate propathnet_env
```

### Option B: Using requirements.txt

The project requires Python (recommended 3.11). All necessary dependencies are listed in the **requirements.txt** file. If you prefer to create the environment manually:

```
conda create -n propathnet_env python=3.11
conda activate propathnet_env
pip install -r requirements.txt
```

(Note: For both options, please ensure you install the appropriate versions of PyTorch and PyTorch Geometric based on your hardware/CUDA specifications before running the script.)


## Quick Start

### 1. View Supported Diseases

You can check all the currently supported diseases (such as Acute_kidney_injury, Atrial_fibrillation, COPD, Diabetes_Type_II, and Fatty_Liver) in the **train/supported_diseases_list.txt** file.
To view the current list directly in your terminal, simply run:

```
cat train/supported_diseases_list.txt
```

### 2. Model Inference (Prediction)

You can easily calculate risk scores for new patient cohorts using the pre-trained models. Use the test_predict.py script and provide the disease name, expression profile data (.h5ad), and clinical features (.csv):

```
DISEASE="Hypertension"                                 
ADATA_PATH="data/sample_data.h5ad"                  
CLINICAL_PATH="data/sample_lifestyle_data.csv"               
OUTPUT_DIR="result"                         

nohup python -u train/disease_prediction.py \
    --disease "$DISEASE" \
    --adata_path "$ADATA_PATH" \
    --clinical_path "$CLINICAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    > log/predict_${DISEASE}.log 2>&1 &
```

The predicted risk scores will be saved as a CSV file in the specified output directory.

### 3. Model Training

If you want to train the ProPathNet model from scratch on your own dataset, you can use the training script provided in the train folder:

```
nohup python -u train/train_model.py \
    --adata_path data/sample_data.h5ad \
    --clinical_path data/sample_lifestyle_data.csv \
    --model_save_path checkpoints/best_test_model.pth \
    --disease Hypertension \
    > log/train.log 2>&1 &
```