#%%
#运行环境
#!/home/xuln/anaconda3/envs/xln_python311/bin/python
#激活环境，在终端运行命令：conda activate /home/xuln/anaconda3/envs/xln_python311
# -*- coding: utf-8 -*-

########################################################################################
##基于xgboost的基线(0年)的临床数据与筛选出的最优蛋白的预测

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, precision_score, recall_score, accuracy_score)
from xgboost import XGBClassifier
import os
import warnings
import json

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 基础大表路径与配置
# ==============================================================================
#(1).所有的蛋白表达数据
PROTEIN_PATH = "/bigdat2/user/xuln/Imputed_protein_cleanoutlier.csv"
#(2).基线的临床数据
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
#(3).最优的生活习惯与蛋白数
RESULT_CSV_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/final_combined_trait_ls_protein/finally111_diseases_with_combined_cindex.csv"

#(4).疾病的基线数据路径
BASE_CSV_DIR = "/bigdat2/user/xuln/olink_disease_predict/data/baseline_data"
#(5).蛋白的IG结果数据路径
BASE_IG_DIR = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT"
#(6).xgboost最优结果数据路径
BASE_MODEL_DIR = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0525"

RANDOM_SEED = 42
N_JOBS = 4  # 限制 CPU 使用数

#如果xgboost需要重新网格搜索可以使用以下参数空间
# XGBoost 参数网格
#XGB_PARAM_GRID = {
#    'xgbclassifier__n_estimators': [100, 200, 300],
#    'xgbclassifier__max_depth': [3, 4, 5],
#    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
#    'xgbclassifier__subsample': [0.7, 0.8, 0.9],
#    'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9],
#    'xgbclassifier__reg_alpha': [0, 0.1, 0.5],
#    'xgbclassifier__reg_lambda': [1.0, 2.0, 3.0]
#}


# ==============================================================================
# 2. 目标疾病与属性映射 (自动对应类型与是否单性别)
# ==============================================================================
TARGET_DISEASES = [
    'Primary_Malignancy_Prostate',
    'Osteoporosis',
    'Iron_deficiency_anaemia',
    'Heart_failure',
    'Hypertension'
]

DISEASE_INFO_MAP = {
    'Primary_Malignancy_Prostate': {'type': 'Cancers', 'is_single_sex': True},
    'Osteoporosis': {'type': 'Musculoskeletal', 'is_single_sex': False},
    'Iron_deficiency_anaemia': {'type': 'Haematological_or_immunological', 'is_single_sex': False},
    'Heart_failure': {'type': 'Cardiovascular', 'is_single_sex': False},
    'Hypertension': {'type': 'Cardiovascular', 'is_single_sex': False}
}

# ==============================================================================
# 3. 核心处理函数
# ==============================================================================
def process_single_disease(disease_name, global_data):
    protein_df, trait_df, trait_features, summary_results_df = global_data
    
    info = DISEASE_INFO_MAP[disease_name]
    d_type = info['type']
    is_single_sex = info['is_single_sex']
    
    print(f"\n{'='*70}")
    print(f"🚀 开始处理疾病: {disease_name}")
    print(f"📍 归属系统: {d_type} | 是否单性别: {'是' if is_single_sex else '否'}")
    
    # ---------------------------------------------------------
    # 路径动态拼接
    # ---------------------------------------------------------
    # 1. 疾病 CSV 路径 (兼容可能存在的 _filtered 后缀)
    csv_path_normal = os.path.join(BASE_CSV_DIR, f"{disease_name}_before_baseline.csv")
    csv_path_filtered = os.path.join(BASE_CSV_DIR, f"{disease_name}_filtered_before_baseline.csv")
    disease_csv_path = csv_path_filtered if is_single_sex and os.path.exists(csv_path_filtered) else csv_path_normal

    # 2. IG 重要性文件路径 (单性别强制加 _filtered)
    ig_folder = f"{disease_name}_filtered" if is_single_sex else disease_name
    prot_ig_file = os.path.join(BASE_IG_DIR, d_type, ig_folder, "joint_protein_importance.csv")

    # 3. 最优参数 JSON 路径
    params_json_path = os.path.join(BASE_MODEL_DIR, d_type, disease_name, "fig5_select_protein_xgb_prediction", "time_gt_minus10", "best_params_Trait_Protein.json")
    
    # 4. 输出路径
    output_dir = os.path.join(BASE_MODEL_DIR, d_type, disease_name, "fig5_select_protein_xgb_prediction", "time_gt_minus10_reloaded")

    # ---------------------------------------------------------
    # 文件检查
    # ---------------------------------------------------------
    if not os.path.exists(disease_csv_path):
        print(f"❌ 找不到疾病数据文件: {disease_csv_path}，跳过！")
        return
    if not os.path.exists(prot_ig_file):
        print(f"❌ 找不到 IG 特征文件: {prot_ig_file}，跳过！")
        return
    if not os.path.exists(params_json_path):
        print(f"❌ 找不到最优参数 JSON: {params_json_path}，跳过！")
        return

    # 从汇总表获取最佳蛋白数量 (注意把下划线换成空格来匹配表格)
    disease_key_for_csv = disease_name.replace('_', ' ')
    try:
        match = summary_results_df[summary_results_df['Disease'] == disease_key_for_csv]
        NUM_PROTEINS = int(match['Trait_Protein_Best_Count'].values[0]) if not match.empty else 5
    except:
        NUM_PROTEINS = 5

    # ---------------------------------------------------------
    # 数据加载与合并
    # ---------------------------------------------------------
    print("⏳ 正在加载队列数据并合并特征...")
    df_disease = pd.read_csv(disease_csv_path)
    df_disease.rename(columns={'Event': 'event', 'Time': 'time'}, inplace=True)
    df_disease['participant.eid'] = df_disease['participant.eid'].astype(str)

    df = pd.merge(df_disease, protein_df, on='participant.eid', how='right')
    df = pd.merge(df, trait_df, on='participant.eid', how='inner')
    df = df.dropna()
    df = df[df['time'] > -10].copy()
    
    print(f"📊 过滤后总样本: {len(df)} | 阳性(Case): {int(df['event'].sum())}")
    
    # 剔除性别特征 (如果是单性别疾病)
    current_trait_features = trait_features.copy()
    if is_single_sex:
        for sex_col in ['31', 31]:
            if sex_col in current_trait_features: 
                current_trait_features.remove(sex_col)
            if str(sex_col) in df.columns: 
                df.drop(columns=[str(sex_col)], inplace=True)
        print("💡 [系统提示]: 已自动剔除单性别疾病的性别特征 '31'")

    # ---------------------------------------------------------
    # 特征选择与数据切分
    # ---------------------------------------------------------
    top_proteins_all = [str(p).upper() for p in pd.read_csv(prot_ig_file).nlargest(200, 'importance')['protein'].tolist()]
    selected_proteins = [p for p in top_proteins_all if p in df.columns][:NUM_PROTEINS]
    
    train_idx, test_idx = train_test_split(df.index.tolist(), test_size=0.2, stratify=df['event'].values, random_state=RANDOM_SEED)
    train_df_raw, test_df_raw = df.loc[train_idx].copy(), df.loc[test_idx].copy()
    
    available_trait_features = [f for f in current_trait_features if f in train_df_raw.columns]
    feature_list = available_trait_features + selected_proteins
    
    print(f"📌 [特征选择] Trait 特征 {len(available_trait_features)}个 | Protein 特征 {len(selected_proteins)}个")

    # ---------------------------------------------------------
    # 读取最优参数并推理
    # ---------------------------------------------------------
    print("⚡ 加载 JSON 最优参数，开始极速推理...")
    with open(params_json_path, 'r') as f:
        best_params_raw = json.load(f)
        
    xgb_params = {}
    for k, v in best_params_raw.items():
        clean_k = k.replace('xgbclassifier__', '')
        xgb_params[clean_k] = v
            
    xgb_params['random_state'] = RANDOM_SEED
    xgb_params['eval_metric'] = 'logloss'
    xgb_params['n_jobs'] = N_JOBS 

    X_train, y_train = train_df_raw[feature_list].copy(), train_df_raw['event'].copy()
    X_test, y_test = test_df_raw[feature_list].copy(), test_df_raw['event'].copy()
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgbclassifier', XGBClassifier(**xgb_params))
    ])

    pipeline.fit(X_train, y_train)

    risk_scores = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # ---------------------------------------------------------
    # 结果评估与保存
    # ---------------------------------------------------------
    test_auc = roc_auc_score(y_test, risk_scores)
    test_auprc = average_precision_score(y_test, risk_scores)
    print(f"✅ 推理完毕 | Test AUC: {test_auc:.4f}, Test AUPRC: {test_auprc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    
    merged_risk = test_df_raw[['participant.eid', 'event']].copy()
    merged_risk['trait+protein_risk_score'] = risk_scores
    merged_risk.to_csv(os.path.join(output_dir, "Trait_Protein_risk_scores.csv"), index=False)
    
    metrics = [{
        'Model': 'Trait_Protein', 
        'Train_AUC': roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1]),
        'Test_AUC': test_auc, 
        'Test_AUPRC': test_auprc, 
        'Test_F1': f1_score(y_test, y_pred, zero_division=0),
        'Test_Precision': precision_score(y_test, y_pred, zero_division=0), 
        'Test_Recall': recall_score(y_test, y_pred, zero_division=0), 
        'Test_Accuracy': accuracy_score(y_test, y_pred)
    }]
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    
    print(f"🎉 结果已保存至: {output_dir}")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================
def main():
    print("🚀 开始批量执行 5 个疾病专属的极速预测任务 (纯 Trait + Protein)")
    
    print("⏳ 正在一次性预加载 Protein 与 Trait 大表数据 (请稍候)...")
    protein_df = pd.read_csv(PROTEIN_PATH)
    if 'participant_eid' in protein_df.columns:
        protein_df.rename(columns={'participant_eid': 'participant.eid'}, inplace=True)
    protein_df['participant.eid'] = protein_df['participant.eid'].astype(str)
    protein_df.columns = [col.upper() if col != 'participant.eid' else col for col in protein_df.columns]

    trait_df = pd.read_csv(TRAIT_PATH)
    if 'eid' in trait_df.columns:
        trait_df.rename(columns={'eid': 'participant.eid'}, inplace=True)
    trait_df['participant.eid'] = trait_df['participant.eid'].astype(str)
    trait_features = [col for col in trait_df.columns if col != 'participant.eid']

    try:
        summary_results_df = pd.read_csv(RESULT_CSV_PATH)
    except:
        summary_results_df = pd.DataFrame()
        
    global_data = (protein_df, trait_df, trait_features, summary_results_df)
    print("✅ 公共大表数据预加载完毕！\n")

    # 循环遍历 5 个目标疾病
    for disease in TARGET_DISEASES:
        process_single_disease(disease, global_data)
        
    print("\n" + "#"*70)
    print("🏁 所有指定疾病的批量推理任务已圆满结束！")
    print("#"*70)

if __name__ == "__main__":
    main()

#%%
#运行环境
#!/home/xuln/anaconda3/envs/xln_python311/bin/python
#激活环境，在终端运行命令：conda activate /home/xuln/anaconda3/envs/xln_python311
# -*- coding: utf-8 -*-

########################################################################################
##基于cox回归的(3、5、10年)的临床数据与筛选出的最优蛋白的预测

"""
==============================================================================
运行环境说明 (Environment Setup):
本脚本已在首行通过 Shebang 指定了专属的 Conda Python 解释器路径。
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
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 专属路径与全局配置
# ==============================================================================
#(1).所有的蛋白表达数据
PROTEIN_PATH = "/bigdat2/user/xuln/Imputed_protein_cleanoutlier.csv"
#(2).基线的临床数据
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
#(3).最优的生活习惯与蛋白数
RESULT_CSV_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/final_combined_trait_ls_protein/finally111_diseases_with_combined_cindex.csv"
#(4).疾病的预后数据路径
BASE_DATA_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"
#(5).蛋白的IG结果数据路径
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"

#(6).输出目录
OUTPUT_DIR = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0525/Cox_target5"

RANDOM_SEED = 42

# 基础 Trait 特征列表
CATEGORICAL_FEATURES = ['31']
CONTINUOUS_FEATURES = [
    '48', '49', '50', '74', '102', '4079', '4080', '20150', '20151', '20258',
    '21001', '21003', '30000', '30010', '30020', '30030', '30040', '30050', 
    '30060', '30080', '30100', '30120', '30130', '30140', '30150', '30160', 
    '30170', '30250', '30260', '30270', '30280', '30300', '30500', '30510', 
    '30520', '30530', '30600', '30610', '30620', '30630', '30640', '30650', 
    '30660', '30670', '30680', '30690', '30700', '30710', '30720', '30730', 
    '30740', '30750', '30760', '30770', '30780', '30790', '30800', '30810', 
    '30820', '30830', '30840', '30850', '30860', '30870', '30880', '30890'
]

# ==============================================================================
# 2. 目标疾病与属性映射 (自动处理 H5AD 文件名与单性别特征)
# ==============================================================================
TARGET_DISEASES = [
    'Primary Malignancy Prostate',
    'Osteoporosis',
    'Iron deficiency anaemia',
    'Heart failure',
    'Hypertension'
]

DISEASE_INFO_MAP = {
    'Primary Malignancy Prostate': {'type': 'Cancers', 'is_single_sex': True, 'h5ad': 'Primary Malignancy Prostate_filtered.h5ad'},
    'Osteoporosis': {'type': 'Musculoskeletal', 'is_single_sex': False, 'h5ad': 'Osteoporosis.h5ad'},
    'Iron deficiency anaemia': {'type': 'Haematological_or_immunological', 'is_single_sex': False, 'h5ad': 'Iron deficiency anaemia.h5ad'},
    'Heart failure': {'type': 'Cardiovascular', 'is_single_sex': False, 'h5ad': 'Heart failure.h5ad'},
    'Hypertension': {'type': 'Cardiovascular', 'is_single_sex': False, 'h5ad': 'Hypertension.h5ad'}
}

# ==============================================================================
# 3. Cox 模型核心函数
# ==============================================================================
def train_cox_model(model_name, feature_list, train_df, val_df, initial_penalizer=0.01):
    cols = feature_list + ['time', 'event']
    df_train_sub = train_df[cols].dropna().copy()
    df_val_sub = val_df[cols].dropna().copy()
    
    if len(df_train_sub) == 0:
        print(f"❌ 训练集为空 (所有样本均含缺失值)")
        return None
    
    # 搜索最佳惩罚项，防止 Cox 不收敛
    penalizers_to_try = [initial_penalizer, 0.05, 0.1, 0.5]
    best_model = None
    best_val_cindex = -np.inf
    best_penalizer = initial_penalizer
    
    for penalizer in penalizers_to_try:
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df_train_sub, duration_col='time', event_col='event')
            c_val = cph.score(df_val_sub, scoring_method="concordance_index")
            
            if c_val > best_val_cindex:
                best_val_cindex = c_val
                best_model = cph
                best_penalizer = penalizer
        except Exception:
            continue
            
    if best_model is None:
        return None
        
    c_train = best_model.score(df_train_sub, scoring_method="concordance_index")
    return best_model, best_penalizer, c_train, best_val_cindex

def calculate_test_cindex(model, test_df, features):
    cols = features + ['time', 'event']
    df_test_clean = test_df[cols].dropna()
    if len(df_test_clean) == 0 or df_test_clean['event'].sum() < 1:
        return None
    try:
        risk_scores = model.predict_partial_hazard(df_test_clean[features]).values.flatten()
        durations = df_test_clean['time'].values
        events = df_test_clean['event'].values
        return concordance_index(durations, -risk_scores, events)
    except Exception:
        return None

def train_and_evaluate_model(feature_list, train_df, val_df, test_df):
    model_result = train_cox_model("Trait_Protein", feature_list, train_df, val_df)
    if model_result is None:
        return None
        
    model, penalizer, train_cindex, val_cindex = model_result
    test_cindex = calculate_test_cindex(model, test_df, feature_list)
    
    if test_cindex is None:
        return None
        
    return {
        'Features_Count': len(feature_list),
        'Penalizer': penalizer,
        'Train_Cindex': train_cindex,
        'Val_Cindex': val_cindex,
        'Test_Cindex': test_cindex,
        'Model_Object': model
    }

# ==============================================================================
# 4. 单个疾病处理流程
# ==============================================================================
def process_single_disease(disease_name, global_data):
    protein_df, trait_df, summary_results_df = global_data
    info = DISEASE_INFO_MAP[disease_name]
    
    d_type = info['type']
    is_single_sex = info['is_single_sex']
    h5ad_filename = info['h5ad']
    
    print(f"\n{'='*70}")
    print(f"🚀 开始处理疾病: {disease_name}")
    print(f"📍 归属系统: {d_type} | H5AD文件: {h5ad_filename}")

    # 1. 确定最优蛋白数 (从已跑完的表格中拿)
    try:
        match = summary_results_df[summary_results_df['Disease'] == disease_name]
        NUM_PROTEINS = int(match['Trait_Protein_Best_Count'].values[0]) if not match.empty else 10
    except:
        NUM_PROTEINS = 10
    
    # 2. 读取 IG 蛋白重要性排名
    ig_folder = disease_name.replace(' ', '_')
    if is_single_sex:
        ig_folder += '_filtered'
    
    prot_ig_file = os.path.join(BASE_IG_RESULTS_PATH, d_type, ig_folder, "joint_protein_importance.csv")
    if not os.path.exists(prot_ig_file):
        print(f"❌ 找不到 IG 蛋白特征文件: {prot_ig_file}，跳过！")
        return None
        
    top_proteins_all = [str(p).upper() for p in pd.read_csv(prot_ig_file).nlargest(200, 'importance')['protein'].tolist()]

    # 3. 加载患者生存标签 (H5AD)
    h5ad_path = os.path.join(BASE_DATA_PATH, d_type, h5ad_filename)
    if not os.path.exists(h5ad_path):
        print(f"❌ 找不到 H5AD 文件: {h5ad_path}，跳过！")
        return None
        
    adata = sc.read_h5ad(h5ad_path)
    survival_df = adata.obs[['time', 'event']].copy()
    survival_df['participant.eid'] = survival_df.index.astype(str)

    # 4. 数据集合并 (以生存队列患者为主，合并蛋白和临床特征)
    print("⏳ 正在合并生存队列患者的 Trait 与 Protein 数据...")
    df_merged = pd.merge(survival_df, trait_df, on='participant.eid', how='left')
    df_merged = pd.merge(df_merged, protein_df, on='participant.eid', how='left')
    df_merged.set_index('participant.eid', inplace=True)

    # 5. 特征清洗与挑选
    current_cat_features = CATEGORICAL_FEATURES.copy()
    if is_single_sex:
        if '31' in current_cat_features:
            current_cat_features.remove('31')
        print("💡 [系统提示]: 单性别疾病，已自动剔除基础 Trait 特征 '31' (性别)")

    expected_traits = [str(c) for c in current_cat_features + CONTINUOUS_FEATURES]
    available_trait_features = [col for col in expected_traits if col in df_merged.columns]
    
    selected_proteins = [p for p in top_proteins_all if p in df_merged.columns][:NUM_PROTEINS]
    feature_list = available_trait_features + selected_proteins
    
    print(f"📌 [特征组装] 最终使用 Trait 特征 {len(available_trait_features)} 个, Protein 特征 {len(selected_proteins)} 个 (根据最优记录)")

    # 6. 切分数据集
    indices = df_merged.index.tolist()
    events = df_merged['event'].values
    
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=events, random_state=RANDOM_SEED)
    train_val_events = df_merged.loc[train_val_idx, 'event'].values
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, stratify=train_val_events, random_state=RANDOM_SEED)

    train_data = df_merged.loc[train_idx].copy()
    val_data = df_merged.loc[val_idx].copy()
    test_data = df_merged.loc[test_idx].copy()

    # 7. 标准化 (标准化连续的 Trait 特征 + 所有的蛋白质)
    print("⚡ 进行特征标准化并训练 Cox 模型...")
    features_to_scale = [col for col in CONTINUOUS_FEATURES if col in available_trait_features] + selected_proteins
    
    scaler = StandardScaler()
    scaler.fit(train_data[features_to_scale])
    train_data[features_to_scale] = scaler.transform(train_data[features_to_scale])
    val_data[features_to_scale] = scaler.transform(val_data[features_to_scale])
    test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])

    # 8. 直接训练一波流
    result = train_and_evaluate_model(feature_list, train_data, val_data, test_data)
    
    if result:
        print(f"✅ Trait+Protein 模型评估完成:")
        print(f"   - 最佳惩罚系数: {result['Penalizer']}")
        print(f"   - Train C-index: {result['Train_Cindex']:.4f}")
        print(f"   - Val C-index:   {result['Val_Cindex']:.4f}")
        print(f"   - Test C-index:  {result['Test_Cindex']:.4f}")
        
        return {
            'Disease': disease_name,
            'Disease_Type': d_type,
            'Total_Features': result['Features_Count'],
            'Protein_Count': len(selected_proteins),
            'Penalizer': result['Penalizer'],
            'Train_Cindex': result['Train_Cindex'],
            'Val_Cindex': result['Val_Cindex'],
            'Test_Cindex': result['Test_Cindex']
        }
    else:
        print(f"❌ 训练失败，可能是可用数据过少或模型不收敛。")
        return None

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    print("=" * 80)
    print("🚀 启动 5 大疾病专属 Cox 回归极速评估 (纯 Trait + Protein)")
    print("=" * 80)
    
    print("⏳ 正在一次性预加载 Protein 与 Trait 全局大表...")
    protein_df = pd.read_csv(PROTEIN_PATH)
    if 'participant_eid' in protein_df.columns:
        protein_df.rename(columns={'participant_eid': 'participant.eid'}, inplace=True)
    protein_df['participant.eid'] = protein_df['participant.eid'].astype(str)
    # 同步转为大写，完美匹配 IG 重要性文件
    protein_df.columns = [col.upper() if col != 'participant.eid' else col for col in protein_df.columns]

    trait_df = pd.read_csv(TRAIT_PATH)
    if 'eid' in trait_df.columns:
        trait_df.rename(columns={'eid': 'participant.eid'}, inplace=True)
    trait_df['participant.eid'] = trait_df['participant.eid'].astype(str)
    
    try:
        summary_results_df = pd.read_csv(RESULT_CSV_PATH)
    except:
        summary_results_df = pd.DataFrame()
        
    global_data = (protein_df, trait_df, summary_results_df)
    print("✅ 全局数据加载完毕！\n")
    
    all_results = []
    
    # 直接遍历目标 5 个疾病
    for disease in TARGET_DISEASES:
        res = process_single_disease(disease, global_data)
        if res:
            all_results.append(res)
            
    # 输出汇总结果
    if all_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"Cox_target5_results_{timestamp}.csv")
        results_df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 80)
        print("🏁 所有任务圆满完成！最终对比表如下：")
        print(results_df.to_string(index=False))
        print(f"\n📁 结果已持久化保存至: {output_file}")
        print("=" * 80)
    else:
        print("❌ 所有任务均未能成功生成结果。")

if __name__ == "__main__":
    main()


# %%
