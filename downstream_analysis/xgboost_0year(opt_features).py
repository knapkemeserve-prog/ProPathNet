# 网格搜索
# xgboost_multithreshold_cv.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             f1_score, precision_score, recall_score, accuracy_score)
from xgboost import XGBClassifier
import os
import warnings
import json
import argparse

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 基础路径与配置参数
# ==============================================================================
PROTEIN_PATH = "/bigdat2/user/xuln/Imputed_protein_cleanoutlier.csv"
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
LIFESTYLE_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/clinic_drop20_10_9.csv"

# 特征重要性与结果汇总路径
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"
RESULT_CSV_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/final_combined_trait_ls_protein/finally111_diseases_with_combined_cindex.csv"

# 模型输出路径
BASE_OUTPUT_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0525"

# 实验配置
RANDOM_SEED = 42
CV_FOLDS = 5
TIME_THRESHOLDS = [-10]
MIN_CASES_REQUIRED = 50  # 核心过滤条件：Case 数必须大于此值

# 🌟 新增：XGBoost 限制使用的最大 CPU 核心数
N_JOBS = 4 

# XGBoost 参数网格
XGB_PARAM_GRID = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 0.9],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 0.9],
    'xgbclassifier__reg_alpha': [0, 0.1, 0.5],
    'xgbclassifier__reg_lambda': [1.0, 2.0, 3.0]
}

# ==============================================================================
# 2. 特殊白名单与分类字典
# ==============================================================================
# 性别特异性疾病白名单 (用于自动寻找 _filtered 文件夹并剔除性别特征)
FILTERED_DISEASES = {
    'Primary Malignancy Prostate',
    'Benign neoplasm and polyp of uterus',
    'Leiomyoma of uterus',
    'Female genital prolapse',
    'Hyperplasia of prostate',
    'Menorrhagia and polymenorrhoea',
    'Postmenopausal bleeding'
}

# 疾病与归属系统大类的映射字典
DISEASE_TYPE_MAP = {
    'Benign neoplasm of colon, rectum, anus and anal canal': 'Benign_neoplasm_or_Carcinoma_in_situ',
    'Benign neoplasm and polyp of uterus': 'Benign_neoplasm_or_Carcinoma_in_situ',
    'Leiomyoma of uterus': 'Benign_neoplasm_or_Carcinoma_in_situ',
    'Primary Malignancy colorectal and anus': 'Cancers',
    'Primary Malignancy Breast': 'Cancers',
    'Primary Malignancy Lung and trachea': 'Cancers',
    'Primary Malignancy Prostate': 'Cancers',
    'Primary Malignancy Other Skin and subcutaneous tissue': 'Cancers',
    'Secondary Malignancy Bone': 'Cancers',
    'Secondary malignancy Liver and intrahepatic bile duct': 'Cancers',
    'Secondary Malignancy Lymph Nodes': 'Cancers',
    'Secondary Malignancy Lung': 'Cancers',
    'Secondary Malignancy Other organs': 'Cancers',
    'Secondary Malignancy retroperitoneum and peritoneum': 'Cancers',
    'Atrial fibrillation': 'Cardiovascular',
    'Coronary heart disease not otherwise specified': 'Cardiovascular',
    'Heart failure': 'Cardiovascular',
    'Hypertension': 'Cardiovascular',
    'Ischaemic stroke': 'Cardiovascular',
    'Left bundle branch block': 'Cardiovascular',
    'Multiple valve dz': 'Cardiovascular',
    'Myocardial infarction': 'Cardiovascular',
    'Nonrheumatic aortic valve disorders': 'Cardiovascular',
    'Nonrheumatic mitral valve disorders': 'Cardiovascular',
    'Pulmonary embolism': 'Cardiovascular',
    'Peripheral arterial disease': 'Cardiovascular',
    'Primary pulmonary hypertension': 'Cardiovascular',
    'Raynaud syndrome': 'Cardiovascular',
    'Right bundle branch block': 'Cardiovascular',
    'Stable angina': 'Cardiovascular',
    'Supraventricular tachycardia': 'Cardiovascular',
    'Transient ischaemic attack': 'Cardiovascular',
    'Unstable Angina': 'Cardiovascular',
    'Venous thromboembolic disease': 'Cardiovascular',
    'Anal fissure': 'Digestive',
    'Barrett oesophagus': 'Digestive',
    'Cholecystitis': 'Digestive',
    'Cholelithiasis': 'Digestive',
    'Diverticular disease of intestine': 'Digestive',
    'Fatty Liver': 'Digestive',
    'Gastritis and duodenitis': 'Digestive',
    'Gastro-oesophageal reflux disease': 'Digestive',
    'Abdominal Hernia': 'Digestive',
    'Diaphragmatic hernia': 'Digestive',
    'Irritable bowel syndrome': 'Digestive',
    'Oesophagitis and oesophageal ulcer': 'Digestive',
    'Peptic ulcer disease': 'Digestive',
    'Hearing loss': 'Ear',
    'Diabetes NOS': 'Endocrine',
    'Diabetes Type II': 'Endocrine',
    'Hypothyroidism': 'Endocrine',
    'Obesity': 'Endocrine',
    'Hypo or hyperthyroidism': 'Endocrine',
    'Cataract': 'Eye',
    'Diabetic ophthalmic complications': 'Eye',
    'Glaucoma': 'Eye',
    'Macular degeneration': 'Eye',
    'Retinal detachments and breaks': 'Eye',
    'Acute Kidney Injury': 'Genitourinary',
    'Hyperplasia of prostate': 'Genitourinary',
    'Chronic Kidney Disease': 'Genitourinary',
    'Female genital prolapse': 'Genitourinary',
    'Menorrhagia and polymenorrhoea': 'Genitourinary',
    'Obstructive and reflux uropathy': 'Genitourinary',
    'Postmenopausal bleeding': 'Genitourinary',
    'Urinary Incontinence': 'Genitourinary',
    'Urolithiasis': 'Genitourinary',
    'Agranulocytosis': 'Haematological_or_immunological',
    'Iron deficiency anaemia': 'Haematological_or_immunological',
    'Other anaemias': 'Haematological_or_immunological',
    'Secondary or other Thrombocytopaenia': 'Haematological_or_immunological',
    'Bacterial Diseases': 'Infections',
    'Infections of the digestive system': 'Infections',
    'Ear and Upper Respiratory Tract Infections': 'Infections',
    'Lower Respiratory Tract Infections': 'Infections',
    'Mycoses': 'Infections',
    'Septicaemia': 'Infections',
    'Infection of skin and subcutaneous tissues': 'Infections',
    'Urinary Tract Infections': 'Infections',
    'Viral diseases': 'Infections',
    'Carpal tunnel syndrome': 'Musculoskeletal',
    'Enthesopathies & synovial disorders': 'Musculoskeletal',
    'Fracture of hip': 'Musculoskeletal',
    'Fracture of wrist': 'Musculoskeletal',
    'Gout': 'Musculoskeletal',
    'Intervertebral disc disorders': 'Musculoskeletal',
    'Osteoarthritis': 'Musculoskeletal',
    'Osteoporosis': 'Musculoskeletal',
    'Polymyalgia Rheumatica': 'Musculoskeletal',
    'Rheumatoid Arthritis': 'Musculoskeletal',
    'Spinal stenosis': 'Musculoskeletal',
    'Spondylosis': 'Musculoskeletal',
    'Postviral fatigue syndrome, neurasthenia and fibromyalgia': 'Neurological',
    'Migraine': 'Neurological',
    'Motor neuron disease': 'Neurological',
    'Alcohol Problems': 'Psychiatric',
    'Anxiety disorders': 'Psychiatric',
    'Delirium, not induced by alcohol and other psychoactive substances': 'Psychiatric',
    'Dementia': 'Psychiatric',
    'Depression': 'Psychiatric',
    'Allergic and chronic rhinitis': 'Respiratory',
    'Asthma': 'Respiratory',
    'Bronchiectasis': 'Respiratory',
    'COPD': 'Respiratory',
    'Pleural effusion': 'Respiratory',
    'Pulmonary collapse': 'Respiratory',
    'Respiratory failure': 'Respiratory',
    'Sleep apnoea': 'Respiratory',
    'Actinic keratosis': 'Skin',
    'Dermatitis': 'Skin',
    'Psoriasis': 'Skin'
}

# ==============================================================================
# 3. 全局数据预加载 (程序启动时只执行一次)
# ==============================================================================
def preload_global_data():
    print("⏳ 正在预加载公共大表数据到内存中，请稍候...")
    protein_df = pd.read_csv(PROTEIN_PATH)
    
    if 'participant_eid' in protein_df.columns:
        protein_df.rename(columns={'participant_eid': 'participant.eid'}, inplace=True)
    
    protein_df['participant.eid'] = protein_df['participant.eid'].astype(str)

    # 🌟 核心修复：将所有的蛋白列名强制转换为大写，完美匹配 IG 文件
    protein_df.columns = [col.upper() if col != 'participant.eid' else col for col in protein_df.columns]

    trait_df = pd.read_csv(TRAIT_PATH)
    if 'eid' in trait_df.columns:
        trait_df.rename(columns={'eid': 'participant.eid'}, inplace=True)
    trait_df['participant.eid'] = trait_df['participant.eid'].astype(str)
    trait_features = [col for col in trait_df.columns if col != 'participant.eid']

    lifestyle_df = pd.read_csv(LIFESTYLE_PATH)
    if 'eid' in lifestyle_df.columns:
        lifestyle_df.rename(columns={'eid': 'participant.eid'}, inplace=True)
    lifestyle_df['participant.eid'] = lifestyle_df['participant.eid'].astype(str)

    try:
        summary_results_df = pd.read_csv(RESULT_CSV_PATH)
    except Exception:
        summary_results_df = pd.DataFrame()
        
    print("✅ 公共数据预加载完成（已自动将蛋白名称转为大写）！\n" + "="*70)
    return protein_df, trait_df, trait_features, lifestyle_df, summary_results_df

# ==============================================================================
# 4. 模型训练与评估函数
# ==============================================================================
def train_and_evaluate_model_with_cv(model_type, feature_list, train_df, test_df, model_name):
    X_train, y_train = train_df[feature_list].copy(), train_df['event'].copy()
    X_test, y_test = test_df[feature_list].copy(), test_df['event'].copy()

    train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    
    test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if len(X_train) == 0:
        return None

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgbclassifier', XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'))
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=XGB_PARAM_GRID, cv=cv,
        scoring='roc_auc', n_jobs=-1, verbose=0, refit=True
    )

    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ {model_type} 网格搜索失败: {e}")
        return None

    best_model = grid_search.best_estimator_
    risk_scores = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    result = {
        'Model_Type': model_type, 'Model_Name': model_name, 'Features_Count': len(feature_list),
        'Best_Params': grid_search.best_params_, 'CV_AUC': grid_search.best_score_,
        'Train_AUC': roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]),
        'Test_AUC': roc_auc_score(y_test, risk_scores), 'Test_AUPRC': average_precision_score(y_test, risk_scores),        
        'Test_F1': f1_score(y_test, y_pred, zero_division=0),              
        'Test_Precision': precision_score(y_test, y_pred, zero_division=0),
        'Test_Recall': recall_score(y_test, y_pred, zero_division=0),      
        'Test_Accuracy': accuracy_score(y_test, y_pred),  
        'Test_Risk_Scores': risk_scores, 'Test_Indices': X_test.index.tolist() 
    }
    print(f"  ✅ {model_type} 完成 | Test AUC: {result['Test_AUC']:.4f}, Test AUPRC: {result['Test_AUPRC']:.4f}")
    return result

# ==============================================================================
# 5. 核心疾病处理流程
# ==============================================================================
def process_single_disease(disease_file_path, time_threshold, global_data):
    protein_df, trait_df, trait_features, lifestyle_df, summary_results_df = global_data
    
    disease_name = os.path.basename(disease_file_path).replace('_before_baseline.csv', '')
    disease_key = disease_name.replace('_filtered', '').replace("_"," ")
    
    # 🌟 判断是否为性别特异性疾病
    is_filtered_disease = (disease_key in FILTERED_DISEASES) or ('_filtered' in disease_name)

    print(f"\n{'='*70}")
    print(f"🚀 处理疾病: {disease_name} | 时间阈值: > {time_threshold}")
    if is_filtered_disease:
        print("💡 [系统提示]: 该疾病为性别特异性疾病，将自动剔除性别特征并映射至 _filtered 文件夹。")
    
    # 获取系统分类
    disease_type = DISEASE_TYPE_MAP.get(disease_key)
    if not disease_type:
        print(f"⏭️ 警告: '{disease_key}' 未在分类字典中找到，跳过此任务。")
        return False
        
    print(f"📍 归属系统: {disease_type}")
    
    # 构建基础 IG 文件夹名，如果是性别特异性疾病，强行加上 _filtered
    ig_folder_name = disease_key.replace(',', '').replace('&', 'and').replace(' ', '_').replace('-', '_')
    if is_filtered_disease:
        ig_folder_name = f"{ig_folder_name}_filtered"
        
    disease_ig_path = os.path.join(BASE_IG_RESULTS_PATH, disease_type, ig_folder_name)

    # 读取合并数据
    df_disease = pd.read_csv(disease_file_path)
    df_disease.rename(columns={'Event': 'event', 'Time': 'time'}, inplace=True)
    df_disease['participant.eid'] = df_disease['participant.eid'].astype(str)

    df = pd.merge(df_disease, protein_df, on='participant.eid', how='right')
    df = pd.merge(df, trait_df, on='participant.eid', how='inner')
    df = pd.merge(df, lifestyle_df, on='participant.eid', how='left')
    df=  df.dropna()
    df = df[df['time'] > time_threshold].copy()
    
    # 核心过滤：Case 数量检查
    case_n = int(df['event'].sum())
    print(f"📊 过滤后总样本: {len(df)} | 阳性(Case): {case_n}")
    if case_n <= MIN_CASES_REQUIRED:
        print(f"⏭️ 阳性事件不足 (Case_N = {case_n} <= {MIN_CASES_REQUIRED})，自动跳过此疾病。")
        return False

    # 获取超参数与特征提取
    NUM_PROTEINS, NUM_LIFESTYLES = 5, 5
    if not summary_results_df.empty:
        match = summary_results_df[summary_results_df['Disease'] == disease_key]
        if not match.empty:
            NUM_PROTEINS, NUM_LIFESTYLES = int(match['Trait_Protein_Best_Count'].values[0]), int(match['Trait_LS_Best_Count'].values[0])

    current_trait_features = trait_features.copy()
    # 🌟 如果是性别特异性疾病，剔除性别特征
    if is_filtered_disease:
        for sex_col in ['31', 31]:
            if sex_col in current_trait_features: current_trait_features.remove(sex_col)
            if str(sex_col) in df.columns: df.drop(columns=[str(sex_col)], inplace=True)

    # 读取特征重要性列表 (由于 IG 中是大写，protein_df 列名也已转为大写，完美匹配)
    prot_ig_file = os.path.join(disease_ig_path, "joint_protein_importance.csv")
    if not os.path.exists(prot_ig_file): 
        print(f"⏭️ 找不到蛋白特征文件: {prot_ig_file}，跳过。")
        return False
    # 这里为了保险，也将提取出的 IG 特征转为大写
    top_proteins_all = [str(p).upper() for p in pd.read_csv(prot_ig_file).nlargest(200, 'importance')['protein'].tolist()]
    selected_proteins = [p for p in top_proteins_all if p in df.columns][:NUM_PROTEINS]

    ls_ig_file = os.path.join(disease_ig_path, "joint_clinical_importance.csv")
    top_lifestyles_all = []
    if os.path.exists(ls_ig_file):
        df_ls = pd.read_csv(ls_ig_file)
        col_name = 'protein' if 'protein' in df_ls.columns else ('feature' if 'feature' in df_ls.columns else 'lifestyle')
        exclude_set = {'age', 'sex', 'bmi'}
        top_lifestyles_all = [str(f) for f in df_ls.sort_values('importance', ascending=False)[col_name].tolist() if str(f).lower() not in exclude_set]
    
    available_ls = [p for p in top_lifestyles_all if p in df.columns]
    # 🌟 再次防御：从生活方式特征中剔除性别
    if is_filtered_disease and 'sex' in available_ls: available_ls.remove('sex')
    selected_lifestyles = available_ls[:NUM_LIFESTYLES]

    # 模型划分以获取可用基础特征
    train_idx, test_idx = train_test_split(df.index.tolist(), test_size=0.2, stratify=df['event'].values, random_state=RANDOM_SEED)
    train_df_raw, test_df_raw = df.loc[train_idx].copy(), df.loc[test_idx].copy()
    available_trait_features = [f for f in current_trait_features if f in train_df_raw.columns]

    # ==========================================================================
    # 🌟 新增：打印选定特征到 Log 🌟
    # ==========================================================================
    print(f"\n📌 [特征选择结果]")
    print(f"   ➤ 基础临床(Trait)特征 ({len(available_trait_features)}个): {available_trait_features}")
    print(f"   ➤ 筛选生活习惯特征 ({len(selected_lifestyles)}个): {selected_lifestyles}")
    print(f"   ➤ 筛选蛋白组学特征 ({len(selected_proteins)}个): {selected_proteins}\n")

    model_features = {
        'Trait_Only': available_trait_features,
        'Trait_Lifestyle': available_trait_features + selected_lifestyles,
        'Trait_Protein': available_trait_features + selected_proteins,
        'Trait_Lifestyle_Protein': available_trait_features + selected_lifestyles + selected_proteins
    }
    
    results_dict = {}
    print(f"⚡ 启动网格搜索训练 (XGBoost 将吃满全部已分配 CPU 核心)...")
    for m_name, features in model_features.items():
        results_dict[m_name] = train_and_evaluate_model_with_cv(m_name, features, train_df_raw, test_df_raw, m_name)

    # 结果持久化保存 (风险评分、最优参数、各项指标)
    merged_risk = test_df_raw[['participant.eid', 'event']].reset_index(drop=True).copy()
    col_mappings = {'Trait_Only': 'trait_risk_score', 'Trait_Lifestyle': 'trait+lifestyle_risk_score',
                    'Trait_Protein': 'trait+protein_risk_score', 'Trait_Lifestyle_Protein': 'trait+lifestyle+protein_risk_score'}
    
    for m_name, score_col in col_mappings.items():
        res = results_dict.get(m_name)
        if res:
            df_tmp = pd.DataFrame({'index': res['Test_Indices'], score_col: res['Test_Risk_Scores']})
            df_tmp['participant.eid'] = df.loc[df_tmp['index'], 'participant.eid'].values
            merged_risk = pd.merge(merged_risk, df_tmp[['participant.eid', score_col]], on='participant.eid', how='left')
    
    # 创建输出目录
    threshold_str = f"time_gt_{time_threshold}".replace('-', 'minus')
    output_subdir = os.path.join(BASE_OUTPUT_PATH, disease_type, disease_name, "fig5_select_protein_xgb_prediction", threshold_str)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存风险评分
    merged_risk.to_csv(os.path.join(output_subdir, f"4models_risk_scores_{disease_name}.csv"), index=False)
    
    # 保存参数与评估指标
    metrics_summary = []
    for m_name in col_mappings.keys():
        res = results_dict.get(m_name)
        if res:
            with open(os.path.join(output_subdir, f"best_params_{m_name}.json"), 'w') as f:
                json.dump(res['Best_Params'], f, indent=4)
            metrics_summary.append({
                'Model': m_name, 'CV_AUC': res['CV_AUC'], 'Train_AUC': res['Train_AUC'],
                'Test_AUC': res['Test_AUC'], 'Test_AUPRC': res['Test_AUPRC'], 'Test_F1': res['Test_F1'],
                'Test_Precision': res['Test_Precision'], 'Test_Recall': res['Test_Recall'], 'Test_Accuracy': res['Test_Accuracy']
            })
            
    if metrics_summary:
        pd.DataFrame(metrics_summary).to_csv(os.path.join(output_subdir, f"evaluation_metrics_{disease_name}.csv"), index=False)

    # ==========================================================================
    # 🌟 新增：保存特征列表到本地 JSON 文件 🌟
    # ==========================================================================
    feature_dict_to_save = {
        "Trait_Features": available_trait_features,
        "Lifestyle_Features": selected_lifestyles,
        "Protein_Features": selected_proteins
    }
    features_file_path = os.path.join(output_subdir, f"selected_features_{disease_name}.json")
    with open(features_file_path, 'w', encoding='utf-8') as f:
        json.dump(feature_dict_to_save, f, indent=4, ensure_ascii=False)
    
    print(f"🎉 特征清单已保存至: {features_file_path}")
    print(f"🎉 当前疾病指标已保存至: {output_subdir}")
    return True

# ==============================================================================
# 6. 主程序入口 (一次性接收 Bash 传来的列表，全局数据只读取 1 次)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量跑疾病预测的 XGBoost 模型")
    # nargs='+' 允许接收以空格分隔的文件列表
    parser.add_argument("--disease_files", type=str, nargs='+', required=True, help="疾病的 CSV 绝对路径列表")
    args = parser.parse_args()

    file_paths = args.disease_files
    print(f"📥 接收到 SLURM 派发的任务清单，共包含 {len(file_paths)} 个疾病文件。")

    # 全局数据只在这里读一次！
    global_data = preload_global_data()

    total_processed = 0
    total_skipped = 0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ 错误：找不到文件 {file_path}，跳过...")
            total_skipped += 1
            continue
            
        for thresh in TIME_THRESHOLDS:
            success = process_single_disease(file_path, thresh, global_data)
            if success:
                total_processed += 1
            else:
                total_skipped += 1

    print("\n" + "#"*70)
    print(f"🏁 批量任务圆满完结！成功完成训练: {total_processed} 次，因 Case 不足或数据缺失跳过: {total_skipped} 次。")
    print("#"*70)