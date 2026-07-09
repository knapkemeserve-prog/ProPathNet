import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import os
import glob
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

#性别类的疾病
#/bigdat2/user/xuln/olink_disease_predict/data/Cancers/Primary Malignancy Prostate_filtered.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Benign_neoplasm_or_Carcinoma_in_situ/Benign neoplasm and polyp of uterus_filtered.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Benign_neoplasm_or_Carcinoma_in_situ/Leiomyoma of uterus_filtered.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Female genital prolapse_filtered.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Hyperplasia of prostate_filtered.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Menorrhagia and polymenorrhoea_filtered.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Postmenopausal bleeding_filtered.h5ad

#要被过滤掉的文件
#/bigdat2/user/xuln/olink_disease_predict/data/Cancers/Primary Malignancy Prostate.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Benign_neoplasm_or_Carcinoma_in_situ/Benign neoplasm and polyp of uterus.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Benign_neoplasm_or_Carcinoma_in_situ/Leiomyoma of uterus.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Female genital prolapse.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Hyperplasia of prostate.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Menorrhagia and polymenorrhoea.h5ad
#/bigdat2/user/xuln/olink_disease_predict/data/Genitourinary/Postmenopausal bleeding.h5ad



# ==============================================================================
# 配置参数
# ==============================================================================

#DISEASE_TYPES = ['Cardiovascular','Benign_neoplasm_or_Carcinoma_in_situ','Ear','Endocrine','Eye','Neurological','Musculoskeletal']
DISEASE_TYPES = ['Cancers', 'Digestive', 'Infections', 'Psychiatric','Genitourinary','Haematological_or_immunological','Respiratory','Skin']



# 基础路径配置
#h5ad数据文件的路径
BASE_DATA_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"
#IG分析的重要蛋白和生活习惯的路径
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"
#对比模型Clinic model
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
#编码后的生活习惯的数据
LIFESTYLE_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/clinic_drop20_10_9.csv"
#对于每个疾病挑选的蛋白和生活习惯的数量
BASE_OUTPUT_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_lifestyle_0525_include_demographics"

# 停止规则参数
PATIENCE = 20           # 连续无显著提升的次数
IMPROVEMENT_THRESHOLD = 0.001  # 提升阈值

# 设置随机种子以确保可重复性
RANDOM_SEED = 42

# 要被过滤掉的单性别疾病或其他原始文件
SKIP_FILES = [
    'Primary Malignancy Prostate.h5ad',
    'Benign neoplasm and polyp of uterus.h5ad',
    'Leiomyoma of uterus.h5ad',
    'Female genital prolapse.h5ad',
    'Hyperplasia of prostate.h5ad',
    'Menorrhagia and polymenorrhoea.h5ad',
    'Postmenopausal bleeding.h5ad'
]

# 生活习惯中的连续变量（标准化时使用）
LIFESTYLE_CONTINUOUS_VARS = [
    '874','884','904','864','1090','1080','1070','1160','1269','1279',
    '1289','1299','1309','1319','1438','1458','1488','1498','1528','1050',
    '1060','2277','2139','2149','age','bmi'
]

# ==============================================================================
# 特征定义 (Trait相关)
# ==============================================================================
categorical_features = ['31']

# 连续变量列表（Trait里的所有变量）
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

# ==============================================================================
# 函数定义
# ==============================================================================

def train_cox_model(model_name, feature_list, train_df, val_df, initial_penalizer=0.01):
    cols = feature_list + ['time', 'event']
    df_train_sub = train_df[cols].dropna().copy()
    df_val_sub = val_df[cols].dropna().copy()
    
    if len(df_train_sub) == 0:
        print(f"❌ 训练集为空 (所有样本均含缺失值)")
        return None
    
    penalizers_to_try = [initial_penalizer, 0.05, 0.1]
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
        except Exception as e:
            continue
    
    if best_model is None:
        return None
    
    c_train = best_model.score(df_train_sub, scoring_method="concordance_index")
    return best_model, best_penalizer, c_train, best_val_cindex

def calculate_test_cindex(model, test_df, features):
    cols = features + ['time', 'event']
    df_test_clean = test_df[cols].dropna()
    
    if len(df_test_clean) == 0:
        return None
    
    if df_test_clean['event'].sum() < 1:
        return None
    
    try:
        risk_scores = model.predict_partial_hazard(df_test_clean[features]).values.flatten()
        durations = df_test_clean['time'].values
        events = df_test_clean['event'].values
        return concordance_index(durations, -risk_scores, events)
    except Exception as e:
        return None

def train_and_evaluate_model(model_type, feature_list, train_df, val_df, test_df, model_name):
    model_result = train_cox_model(model_name, feature_list, train_df, val_df, initial_penalizer=0.01)
    
    if model_result is None:
        return None
    
    model, penalizer, train_cindex, val_cindex = model_result
    test_cindex = calculate_test_cindex(model, test_df, feature_list)
    
    if test_cindex is None:
        return None
    
    result = {
        'Model_Type': model_type,
        'Model_Name': model_name,
        'Features_Count': len(feature_list),
        'Penalizer': penalizer,
        'Train_Cindex': train_cindex,
        'Val_Cindex': val_cindex,
        'Test_Cindex': test_cindex,
        'Model_Object': model
    }
    return result

def process_disease_all_models(disease_name, disease_file, data_path, trait_path, lifestyle_path, top_lifestyles_dict, output_path):
    print(f"\n{'='*60}")
    print(f"处理疾病: {disease_name}")
    print(f"文件: {disease_file}")
    
    try:
        # 1. 加载疾病标签数据 (提供 eid, time, event)
        h5ad_path = os.path.join(data_path, disease_file)
        adata = sc.read_h5ad(h5ad_path)
        
        if 'event' not in adata.obs.columns:
            print("❌ 错误: 疾病标签数据中没有 'event' 列")
            return None
            
        event_counts = adata.obs['event'].value_counts()
        if 1 not in event_counts or event_counts[1] < 5:
            print(f"⚠️ 警告: 疾病事件数量太少，跳过该疾病")
            return None
        
        # 2. 获取该疾病的生活习惯重要性列表
        disease_key = disease_name.replace('.h5ad', '').replace('_survival_data', '').replace(',', '').replace('&', 'and').replace(' ', '_').replace('-', '_')
        possible_keys = [
            disease_key,
            disease_key[0] + disease_key[1:].lower() if disease_key and len(disease_key) > 0 else disease_key,
            disease_key.lower(),
            disease_key.upper()
        ]
        
        top_lifestyles = []
        for key in possible_keys:
            if key in top_lifestyles_dict:
                top_lifestyles = top_lifestyles_dict[key]
                break
        
        if not top_lifestyles:
            print(f"⚠️ 警告: 疾病 {disease_name} 没有找到生活习惯重要性信息")
            return None
        
        # 3. 划分数据集
        indices = list(range(len(adata.obs)))
        event_status = adata.obs['event'].values
        
        train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=event_status, random_state=RANDOM_SEED)
        train_val_events = event_status[train_val_idx]
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, stratify=train_val_events, random_state=RANDOM_SEED)
        
        train_data = adata.obs.iloc[train_idx].copy()
        val_data = adata.obs.iloc[val_idx].copy()
        test_data = adata.obs.iloc[test_idx].copy()
        
        train_data['participant.eid'] = train_data.index.astype(str)
        val_data['participant.eid'] = val_data.index.astype(str)
        test_data['participant.eid'] = test_data.index.astype(str)
        
        # 4. 分别加载 Trait 和 Lifestyle 数据，并进行合并
        trait_df = pd.read_csv(trait_path)
        if 'eid' in trait_df.columns:
            trait_df = trait_df.rename(columns={'eid': 'participant.eid'})
        trait_df['participant.eid'] = trait_df['participant.eid'].astype(str)
        
        lifestyle_df = pd.read_csv(lifestyle_path)
        if 'eid' in lifestyle_df.columns:
            lifestyle_df = lifestyle_df.rename(columns={'eid': 'participant.eid'})
        lifestyle_df['participant.eid'] = lifestyle_df['participant.eid'].astype(str)
        
        # 外连接合并两个特征表以防部分患者仅在一个表中存在
        all_features_df = pd.merge(trait_df, lifestyle_df, on='participant.eid', how='outer')
        all_features_df.columns = [str(c) for c in all_features_df.columns] 
        
        # 将特征合并到 train, val, test 上
        train_data_merged = pd.merge(train_data, all_features_df, on='participant.eid', how='left').set_index('participant.eid')
        val_data_merged = pd.merge(val_data, all_features_df, on='participant.eid', how='left').set_index('participant.eid')
        test_data_merged = pd.merge(test_data, all_features_df, on='participant.eid', how='left').set_index('participant.eid')
        
        # 判断是否为需要过滤性别的特定疾病
        is_filtered_disease = '_filtered' in disease_name

        # 5. 严格挑选出有效的基础 Trait 特征
        current_cat_features = categorical_features.copy()
        if is_filtered_disease and '31' in current_cat_features:
            current_cat_features.remove('31')
            print("    [Info] 当前为单性别疾病 (_filtered)，已自动剔除基础 Trait 特征 '31' (Sex)")
            
        expected_traits = [str(c) for c in current_cat_features + continuous_features]
        trait_features = [col for col in expected_traits if col in all_features_df.columns]
        
        # 从大的临床表中挑选出有效的 Lifestyle 特征
        available_lifestyles = [str(p) for p in top_lifestyles if str(p) in all_features_df.columns]
        
        if is_filtered_disease and 'sex' in available_lifestyles:
            available_lifestyles.remove('sex')
            print("    [Info] 当前为单性别疾病 (_filtered)，已自动剔除生活习惯特征 'sex'")
            
        max_lifestyles = len(available_lifestyles)
        print(f"在整合表中找到 {max_lifestyles}/{len(top_lifestyles)} 个重要的生活习惯特征")
        
        if not available_lifestyles:
            print("❌ 没有找到任何可用的生活习惯特征，跳过该疾病")
            return None
        
        # 6. 测试模型
        all_results = []
        
        # =========== 模型A: 单独Trait模型 ===========
        print(f"\n📌 模型A: 单独Trait模型")
        train_trait = train_data_merged.copy()
        val_trait = val_data_merged.copy()
        test_trait = test_data_merged.copy()
        
        exist_cont_features = [col for col in continuous_features if col in train_trait.columns]
        if exist_cont_features:
            scaler_trait = StandardScaler()
            scaler_trait.fit(train_trait[exist_cont_features])
            train_trait[exist_cont_features] = scaler_trait.transform(train_trait[exist_cont_features])
            val_trait[exist_cont_features] = scaler_trait.transform(val_trait[exist_cont_features])
            test_trait[exist_cont_features] = scaler_trait.transform(test_trait[exist_cont_features])
            
        trait_result = train_and_evaluate_model('Trait_Only', trait_features, train_trait, val_trait, test_trait, 'Trait_Only')
        
        if trait_result:
            all_results.append(trait_result)
            trait_val_cindex = trait_result['Val_Cindex']
            trait_test_cindex = trait_result['Test_Cindex']
            trait_penalizer = trait_result['Penalizer'] # 获取基础模型惩罚项
            print(f"  结果: Val C-index={trait_val_cindex:.4f}, Test C-index={trait_test_cindex:.4f}, Penalizer={trait_penalizer}")
        else:
            trait_val_cindex = -np.inf
            trait_test_cindex = -np.inf
            trait_penalizer = None
        
        # =========== 模型B: Trait+生活习惯模型 (动态添加，带停止规则) ===========
        print(f"\n📌 模型B: Trait+生活习惯模型（动态添加，最高 {max_lifestyles} 个特征）")
        trait_lifestyle_results = []
        best_trait_ls_val_cindex = -np.inf
        best_trait_ls_count = 0
        best_trait_ls_result = None
        
        last_val_cindex = None
        no_improvement_count = 0
        
        for count in range(1, max_lifestyles + 1):
            selected_lifestyles = available_lifestyles[:count]
            
            train_tl = train_data_merged.copy()
            val_tl = val_data_merged.copy()
            test_tl = test_data_merged.copy()
            
            # 确定哪些特征需要标准化
            features_to_scale = []
            if exist_cont_features:
                features_to_scale.extend(exist_cont_features)
                
            if selected_lifestyles:
                cont_lifestyles = [p for p in selected_lifestyles if p in LIFESTYLE_CONTINUOUS_VARS]
                features_to_scale.extend(cont_lifestyles)
                
            if features_to_scale:
                scaler_tl = StandardScaler()
                scaler_tl.fit(train_tl[features_to_scale])
                train_tl[features_to_scale] = scaler_tl.transform(train_tl[features_to_scale])
                val_tl[features_to_scale] = scaler_tl.transform(val_tl[features_to_scale])
                test_tl[features_to_scale] = scaler_tl.transform(test_tl[features_to_scale])
            
            features = trait_features + selected_lifestyles
            
            tl_result = train_and_evaluate_model('Trait+Lifestyle', features, train_tl, val_tl, test_tl, f'Trait+Lifestyle_Top{count}')
            
            if tl_result:
                tl_result['Lifestyle_Count'] = count
                trait_lifestyle_results.append(tl_result)
                
                if tl_result['Val_Cindex'] > best_trait_ls_val_cindex:
                    best_trait_ls_val_cindex = tl_result['Val_Cindex']
                    best_trait_ls_count = count
                    best_trait_ls_result = tl_result
                
                current_val = tl_result['Val_Cindex']
                if last_val_cindex is not None:
                    improvement = current_val - last_val_cindex
                    if improvement < IMPROVEMENT_THRESHOLD:
                        no_improvement_count += 1
                        print(f"    提升 {improvement:.6f} < {IMPROVEMENT_THRESHOLD}，连续无提升次数: {no_improvement_count} (当前惩罚项: {tl_result['Penalizer']})")
                    else:
                        no_improvement_count = 0
                        print(f"    提升 {improvement:.6f} >= {IMPROVEMENT_THRESHOLD}，重置计数器 (当前惩罚项: {tl_result['Penalizer']})")
                    
                    if no_improvement_count >= PATIENCE:
                        print(f"    连续 {PATIENCE} 次提升小于 {IMPROVEMENT_THRESHOLD}，触发停止规则")
                        break
                last_val_cindex = current_val
        
        all_results.extend(trait_lifestyle_results)
        
        if not all_results:
            return None
        
        results_df = pd.DataFrame([{
            'Disease': disease_name,
            'Model_Type': r['Model_Type'],
            'Model_Name': r['Model_Name'],
            'Lifestyle_Count': r.get('Lifestyle_Count', 0),
            'Features_Count': r['Features_Count'],
            'Penalizer': r['Penalizer'],
            'Train_Cindex': r['Train_Cindex'],
            'Val_Cindex': r['Val_Cindex'],
            'Test_Cindex': r['Test_Cindex']
        } for r in all_results])
        
        tl_df = results_df[results_df['Model_Type'] == 'Trait+Lifestyle'].copy()
        tl_plateau_count = tl_df['Lifestyle_Count'].max() if len(tl_df) > 0 else 0
        
        # =========== 汇总输出字典 (这里增加了最优惩罚系数) ===========
        summary = {
            'Disease_Type': data_path.split('/')[-2] if data_path.split('/')[-2] != 'data' else data_path.split('/')[-1],
            'Disease': disease_name,
            'Total_Available_Lifestyles': max_lifestyles,
            'Trait_Only_Val_Cindex': trait_val_cindex,
            'Trait_Only_Test_Cindex': trait_test_cindex,
            'Trait_Only_Penalizer': trait_penalizer,
            'Trait_LS_Best_Count': best_trait_ls_count,
            'Trait_LS_Best_Val_Cindex': best_trait_ls_val_cindex,
            'Trait_LS_Best_Test_Cindex': best_trait_ls_result['Test_Cindex'] if best_trait_ls_result else None,
            'Trait_LS_Best_Penalizer': best_trait_ls_result['Penalizer'] if best_trait_ls_result else None,
            'Trait_LS_Plateau_Count': tl_plateau_count,
            'Trait_LS_Max_Test_Cindex': tl_df['Test_Cindex'].max() if len(tl_df) > 0 else None,
            'Trait_LS_vs_Trait_Improvement': (best_trait_ls_val_cindex - trait_val_cindex) if best_trait_ls_result else None
        }
        
        # =========== 在日志中直观展示该疾病的核心对比结果 ===========
        print(f"\n📊 [{disease_name}] 最终结果总结:")
        print(f"  ▶ 仅 Trait 模型:")
        print(f"    - Val C-index:  {trait_val_cindex:.4f}" if trait_val_cindex != -np.inf else "    - Val C-index:  None")
        print(f"    - Test C-index: {trait_test_cindex:.4f}" if trait_test_cindex != -np.inf else "    - Test C-index: None")
        print(f"    - Penalizer:    {trait_penalizer}")
        
        if best_trait_ls_result:
            print(f"  ▶ 最优 Trait + Lifestyle 模型:")
            print(f"    - 最优生活习惯数: {best_trait_ls_count}")
            print(f"    - Val C-index:  {best_trait_ls_val_cindex:.4f}")
            print(f"    - Test C-index: {best_trait_ls_result['Test_Cindex']:.4f}")
            print(f"    - Penalizer:    {best_trait_ls_result['Penalizer']}")
            print(f"    - 相对 Trait 提升: {(best_trait_ls_val_cindex - trait_val_cindex):.4f}")
        else:
            print(f"  ▶ 最优 Trait + Lifestyle 模型: 无")
        print(f"{'='*60}\n")
        
        os.makedirs(output_path, exist_ok=True)
        detailed_file = os.path.join(output_path, f"all_models_comparison_{disease_name}.csv")
        results_df.to_csv(detailed_file, index=False)
        
        create_comparison_plot(results_df, disease_name, output_path)
        return summary, results_df
        
    except Exception as e:
        print(f"处理疾病 {disease_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_comparison_plot(results_df, disease_name, output_path):
    try:
        trait_ls_df = results_df[results_df['Model_Type'] == 'Trait+Lifestyle'].copy()
        trait_only_df = results_df[results_df['Model_Type'] == 'Trait_Only'].copy()
        
        plt.figure(figsize=(10, 6))
        
        if len(trait_ls_df) > 0:
            trait_ls_df = trait_ls_df.sort_values('Lifestyle_Count')
            plt.plot(trait_ls_df['Lifestyle_Count'], trait_ls_df['Test_Cindex'], 's-', color='red', linewidth=2, markersize=8, label='Trait+Lifestyle')
            
        if len(trait_only_df) > 0:
            trait_test_cindex = trait_only_df['Test_Cindex'].iloc[0]
            plt.axhline(y=trait_test_cindex, color='green', linestyle='--', linewidth=2, label='Trait Only')
            
        plt.xlabel('Number of Top Lifestyle Features', fontsize=14)
        plt.ylabel('Test C-index', fontsize=14)
        plt.title(f'Model Comparison for {disease_name}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        #plt.savefig(os.path.join(output_path, f"model_comparison_{disease_name}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"创建比较图时出错: {e}")

def process_disease_type_all_models(disease_type):
    data_path = os.path.join(BASE_DATA_PATH, disease_type)
    ig_results_path = os.path.join(BASE_IG_RESULTS_PATH, disease_type)
    output_path = os.path.join(BASE_OUTPUT_PATH, disease_type, "all_models_analysis")
    os.makedirs(output_path, exist_ok=True)
    
    top_lifestyles_dict = {}
    exclude_folders = ["protein_frequency_analysis_results", "lifestyle_frequency_analysis_results"]
    
    try:
        disease_folders = [f for f in os.listdir(ig_results_path) if os.path.isdir(os.path.join(ig_results_path, f)) and f not in exclude_folders]
        for disease in sorted(disease_folders):
            file_path = os.path.join(ig_results_path, disease, "joint_clinical_importance.csv")
            if not os.path.exists(file_path):
                continue
            try:
                df = pd.read_csv(file_path)
                col_name = 'protein' if 'protein' in df.columns else ('feature' if 'feature' in df.columns else 'lifestyle')
                top_features = df.sort_values('importance', ascending=False)[col_name].tolist()
                
                top_lifestyles_dict[disease] = [f for f in top_features]
            except Exception as e:
                continue
    except Exception as e:
        print(f"扫描重要性文件夹出错: {e}")
        return None, None
    
    h5ad_files = glob.glob(os.path.join(data_path, "*.h5ad"))
    all_summaries = []
    all_detailed_results = []
    
    for h5ad_file in sorted(h5ad_files):
        file_name = os.path.basename(h5ad_file)
        
        # 过滤未处理的基础文件
        if file_name in SKIP_FILES:
            print(f"⏭️ 根据规则跳过未过滤的文件: {file_name}")
            continue
            
        disease_name = os.path.splitext(file_name)[0]
        summary, detailed_results = process_disease_all_models(disease_name, file_name, data_path, TRAIT_PATH, LIFESTYLE_PATH, top_lifestyles_dict, output_path)
        
        if summary is not None:
            all_summaries.append(summary)
            detailed_results['Disease_Type'] = disease_type
            all_detailed_results.append(detailed_results)
            
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        detailed_df = pd.concat(all_detailed_results, ignore_index=True)
        summary_df.to_csv(os.path.join(output_path, f"all_models_summary_{disease_type}.csv"), index=False)
        detailed_df.to_csv(os.path.join(output_path, f"all_models_detailed_{disease_type}.csv"), index=False)
        return summary_df, detailed_df
    return None, None

def create_final_summary_table(all_summaries):
    if not all_summaries:
        return None
    
    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    final_summary = []
    
    for disease_type in DISEASE_TYPES:
        type_results = all_summaries_df[all_summaries_df['Disease_Type'] == disease_type]
        if len(type_results) == 0:
            continue
            
        type_stats = {
            'Disease_Type': disease_type,
            'Number_of_Diseases': len(type_results),
            'Avg_Trait_Only_Test_Cindex': type_results['Trait_Only_Test_Cindex'].mean(),
            'Avg_Trait_LS_Best_Count': type_results['Trait_LS_Best_Count'].mean(),
            'Median_Trait_LS_Best_Count': type_results['Trait_LS_Best_Count'].median(),
            'Std_Trait_LS_Best_Count': type_results['Trait_LS_Best_Count'].std(),
            'Avg_Trait_LS_Test_Cindex': type_results['Trait_LS_Best_Test_Cindex'].mean(),
            'Trait_LS_Better_Than_Trait': (type_results['Trait_LS_Best_Test_Cindex'] > type_results['Trait_Only_Test_Cindex']).sum(),
            'Avg_Trait_LS_vs_Trait_Improvement': type_results['Trait_LS_vs_Trait_Improvement'].mean(),
        }
        final_summary.append(type_stats)
    
    overall_stats = {
        'Disease_Type': 'Overall',
        'Number_of_Diseases': len(all_summaries_df),
        'Avg_Trait_Only_Test_Cindex': all_summaries_df['Trait_Only_Test_Cindex'].mean(),
        'Avg_Trait_LS_Best_Count': all_summaries_df['Trait_LS_Best_Count'].mean(),
        'Median_Trait_LS_Best_Count': all_summaries_df['Trait_LS_Best_Count'].median(),
        'Std_Trait_LS_Best_Count': all_summaries_df['Trait_LS_Best_Count'].std(),
        'Avg_Trait_LS_Test_Cindex': all_summaries_df['Trait_LS_Best_Test_Cindex'].mean(),
        'Trait_LS_Better_Than_Trait': (all_summaries_df['Trait_LS_Best_Test_Cindex'] > all_summaries_df['Trait_Only_Test_Cindex']).sum(),
        'Avg_Trait_LS_vs_Trait_Improvement': all_summaries_df['Trait_LS_vs_Trait_Improvement'].mean()
    }
    final_summary.append(overall_stats)
    return pd.DataFrame(final_summary)

def main():
    print("=" * 100)
    print("开始测试 Trait 与 Trait+Lifestyle 模型的 Cox回归性能")
    print("特征添加范围: 将根据每个疾病实际可用的生活习惯特征数自动推断")
    print(f"停止规则: 连续 {PATIENCE} 次提升小于 {IMPROVEMENT_THRESHOLD} 时停止")
    print("=" * 100)
    
    all_summaries = []
    all_detailed_results = []
    
    for disease_type in DISEASE_TYPES:
        summary_df, detailed_df = process_disease_type_all_models(disease_type)
        if summary_df is not None:
            all_summaries.append(summary_df)
            all_detailed_results.append(detailed_df)
            
    if all_summaries:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        all_summaries_df = pd.concat(all_summaries, ignore_index=True)
        all_summaries_df.to_csv(os.path.join(BASE_OUTPUT_PATH, f"all_diseases_summary_{timestamp}.csv"), index=False)
        
        all_detailed_df = pd.concat(all_detailed_results, ignore_index=True)
        all_detailed_df.to_csv(os.path.join(BASE_OUTPUT_PATH, f"all_diseases_detailed_{timestamp}.csv"), index=False)
        
        final_summary_table = create_final_summary_table(all_summaries)
        if final_summary_table is not None:
            final_summary_table.to_csv(os.path.join(BASE_OUTPUT_PATH, f"final_summary_table_{timestamp}.csv"), index=False)
            print(f"\n最终汇总表格:\n{final_summary_table.to_string(index=False)}")

if __name__ == "__main__":
    main()


#nohup python -u /home/xuln/olink_disease_predict/code/fig4_67trait_with_select_clinic_life_0525.py > /home/xuln/olink_disease_predict/code/fig4_67trait_with_select_clinic_life_0525_2.log 2>&1 &