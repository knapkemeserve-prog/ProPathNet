#%%
##############################################################
###逐步挑选蛋白
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

# ==============================================================================
# 配置参数
# ==============================================================================
# 定义四种疾病类型
#DISEASE_TYPES = ['Cancers', 'Digestive', 'Infections', 'Psychiatric']
#DISEASE_TYPES = ['Benign_neoplasm_or_Carcinoma_in_situ','Cardiovascular','Ear','Endocrine','Eye','Neurological','Musculoskeletal','Genitourinary','Haematological_or_immunological','Respiratory','Skin']
DISEASE_TYPES = ['Cardiovascular','Benign_neoplasm_or_Carcinoma_in_situ','Ear','Endocrine','Eye','Neurological','Musculoskeletal','Genitourinary','Haematological_or_immunological','Respiratory','Skin']

# 基础路径配置
BASE_DATA_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
BASE_OUTPUT_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_0309"

# 蛋白数量配置：从1到2506（覆盖Olink平台所有可能蛋白数）
PROTEIN_COUNTS = list(range(1, 2507))  # 1,2,3,...,2506

# 停止规则参数
PATIENCE = 20           # 连续无显著提升的次数
IMPROVEMENT_THRESHOLD = 0.005  # 提升阈值

# 设置随机种子以确保可重复性
RANDOM_SEED = 42

# 💡 新增：要被过滤掉的单性别疾病或其他原始文件
SKIP_FILES = [
    'Primary Malignancy Prostate.h5ad',
    'Benign neoplasm and polyp of uterus.h5ad',
    'Leiomyoma of uterus.h5ad',
    'Female genital prolapse.h5ad',
    'Hyperplasia of prostate.h5ad',
    'Menorrhagia and polymenorrhoea.h5ad',
    'Postmenopausal bleeding.h5ad'
]

# ==============================================================================
# 特征定义
# ==============================================================================
categorical_features = ['31']

# 连续变量列表（所有变量）
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
    """
    训练Cox模型，使用验证集选择最佳正则化系数
    """
    cols = feature_list + ['time', 'event']
    
    # 预处理：删除缺失值
    df_train_sub = train_df[cols].dropna().copy()
    df_val_sub = val_df[cols].dropna().copy()
    
    if len(df_train_sub) == 0:
        print(f"❌ 训练集为空 (所有样本均含缺失值)")
        return None
    
    # 尝试的正则化系数列表
    penalizers_to_try = [initial_penalizer, 0.05, 0.1]
    
    best_model = None
    best_val_cindex = -np.inf
    best_penalizer = initial_penalizer
    
    print(f"  训练模型 '{model_name}' - 尝试正则化系数: {penalizers_to_try}")
    
    for penalizer in penalizers_to_try:
        try:
            # 训练模型
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df_train_sub, duration_col='time', event_col='event')
            
            # 计算验证集C-index
            c_val = cph.score(df_val_sub, scoring_method="concordance_index")
            
            print(f"    正则化系数 {penalizer:.3f} - 验证集C-index: {c_val:.4f}")
            
            # 选择最佳模型（即使C-index<0.5也选择最好的）
            if c_val > best_val_cindex:
                best_val_cindex = c_val
                best_model = cph
                best_penalizer = penalizer
            
        except Exception as e:
            print(f"    正则化系数 {penalizer:.3f} 训练失败: {e}")
            continue
    
    if best_model is None:
        print(f"    所有正则化系数均失败")
        return None
    
    print(f"    选择正则化系数 {best_penalizer:.3f} - 验证集C-index: {best_val_cindex:.4f}")
    
    # 计算训练集C-index
    c_train = best_model.score(df_train_sub, scoring_method="concordance_index")
    print(f"    训练集C-index: {c_train:.4f}")
    
    return best_model, best_penalizer, c_train, best_val_cindex

def calculate_test_cindex(model, test_df, features):
    """
    计算模型在测试集上的C-index（不使用Bootstrap）
    """
    cols = features + ['time', 'event']
    
    # 删除缺失值
    df_test_clean = test_df[cols].dropna()
    
    if len(df_test_clean) == 0:
        print("    错误: 测试集在删除缺失值后为空！")
        return None
    
    # 检查事件数量
    cleaned_events = df_test_clean['event'].sum()
    if cleaned_events < 1:
        print(f"    警告: 测试集没有'事件'(Event=1)样本！")
        return None
    
    # 获取模型的预测风险评分
    try:
        risk_scores = model.predict_partial_hazard(df_test_clean[features]).values.flatten()
        durations = df_test_clean['time'].values
        events = df_test_clean['event'].values
        
        # 计算C-index
        c_index = concordance_index(durations, -risk_scores, events)
        return c_index
        
    except Exception as e:
        print(f"    错误: 计算C-index失败: {e}")
        return None

def train_and_evaluate_model(model_type, feature_list, train_df, val_df, test_df, model_name):
    """
    训练并评估一个模型
    """
    print(f"\n  {model_type}模型 '{model_name}'...")
    
    # 训练模型
    model_result = train_cox_model(model_name, feature_list, train_df, val_df, initial_penalizer=0.01)
    
    if model_result is None:
        print(f"  模型训练失败")
        return None
    
    model, penalizer, train_cindex, val_cindex = model_result
    
    # 在测试集上评估
    test_cindex = calculate_test_cindex(model, test_df, feature_list)
    
    if test_cindex is None:
        print(f"  测试集评估失败")
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
    
    print(f"  结果: Val C-index={val_cindex:.4f}, Test C-index={test_cindex:.4f}")
    
    return result

def process_disease_all_models(disease_name, disease_file, data_path, trait_path, top_proteins_dict, output_path):
    """
    处理单个疾病的数据，测试三种模型：
    1. 单独Trait模型
    2. 单独蛋白模型（动态添加，带停止规则）
    3. Trait+蛋白模型（动态添加，带停止规则）
    """
    print(f"\n{'='*60}")
    print(f"处理疾病: {disease_name}")
    print(f"文件: {disease_file}")
    print(f"{'='*60}")
    
    try:
        # 1. 加载数据
        h5ad_path = os.path.join(data_path, disease_file)
        print(f"加载数据: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        print(f"数据形状: {adata.shape}")
        
        # 检查事件列
        if 'event' not in adata.obs.columns:
            print(f"错误: 数据中没有'event'列")
            return None
            
        event_counts = adata.obs['event'].value_counts()
        print(f"事件统计: {event_counts.to_dict()}")
        
        # 检查是否有足够的事件
        if 1 not in event_counts or event_counts[1] < 5:
            print(f"警告: 事件数量太少 ({event_counts.get(1, 0)})，跳过该疾病")
            return None
        
        # 2. 获取该疾病的蛋白重要性列表
        disease_key = disease_name.replace('.h5ad', '').replace('_survival_data', '').replace(',', '').replace('&', 'and').replace(' ', '_').replace('-', '_')
        
        # 尝试不同格式的疾病键
        possible_keys = [
            disease_key,
            disease_key[0] + disease_key[1:].lower() if disease_key and len(disease_key) > 0 else disease_key,
            disease_key.lower(),
            disease_key.upper()
        ]
        
        top_proteins = []
        found_key = None
        for key in possible_keys:
            if key in top_proteins_dict:
                top_proteins = top_proteins_dict[key]
                found_key = key
                print(f"找到 {len(top_proteins)} 个蛋白 (使用键: {found_key})")
                break
        
        if not top_proteins:
            print(f"警告: 疾病 {disease_name} 没有找到蛋白重要性信息")
            return None
        
        # 3. 划分数据集
        indices = list(range(len(adata.obs)))
        event_status = adata.obs['event'].values
        
        # 尝试分层抽样，失败则使用随机抽样
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=event_status, random_state=RANDOM_SEED
        )
        train_val_events = event_status[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.125, stratify=train_val_events, random_state=RANDOM_SEED
        )
        
        print(f"训练集样本数: {len(train_idx)}")
        print(f"验证集样本数: {len(val_idx)}")
        print(f"测试集样本数: {len(test_idx)}")
        
        train_data = adata.obs.iloc[train_idx]
        val_data = adata.obs.iloc[val_idx]
        test_data = adata.obs.iloc[test_idx]
        
        # 4. 加载Trait数据并合并
        print("加载Trait数据并合并...")
        trait_df = pd.read_csv(trait_path)
        trait_df = trait_df.rename(columns={'eid': 'participant.eid'})
        trait_df['participant.eid'] = trait_df['participant.eid'].astype(str)
        
        # 获取Trait特征列表（排除ID列）
        trait_features = [col for col in trait_df.columns if col != 'participant.eid']
        
        # 💡 新增：针对单性别疾病剔除性别特征 '31'
        is_filtered_disease = '_filtered' in disease_name
        if is_filtered_disease and '31' in trait_features:
            trait_features.remove('31')
            print("    [Info] 当前为单性别疾病 (_filtered)，已自动剔除基础 Trait 特征 '31' (Sex)")
        
        # 准备数据
        train_data = train_data.copy()
        val_data = val_data.copy()
        test_data = test_data.copy()
        
        train_data['participant.eid'] = train_data.index.astype(str)
        val_data['participant.eid'] = val_data.index.astype(str)
        test_data['participant.eid'] = test_data.index.astype(str)
        
        # 合并Trait数据
        train_data_merged = pd.merge(train_data, trait_df, on='participant.eid', how='left')
        val_data_merged = pd.merge(val_data, trait_df, on='participant.eid', how='left')
        test_data_merged = pd.merge(test_data, trait_df, on='participant.eid', how='left')
        
        train_data_merged.set_index('participant.eid', inplace=True)
        val_data_merged.set_index('participant.eid', inplace=True)
        test_data_merged.set_index('participant.eid', inplace=True)
        
        print(f"合并后形状: Train={train_data_merged.shape}, Val={val_data_merged.shape}, Test={test_data_merged.shape}")
        
        # 5. 添加蛋白表达数据
        # 从adata.X中提取蛋白表达数据
        if hasattr(adata.X, 'toarray'):
            protein_matrix = adata.X.toarray()
        else:
            protein_matrix = adata.X
        
        protein_df = pd.DataFrame(
            protein_matrix,
            index=adata.obs.index,
            columns=adata.var_names
        )
        
        # 获取所有可用的蛋白（按重要性排序）
        available_proteins = [p for p in top_proteins if p in protein_df.columns]
        print(f"在表达数据中找到 {len(available_proteins)}/{len(top_proteins)} 个蛋白")
        
        if not available_proteins:
            print("没有找到可用的蛋白，跳过该疾病")
            return None
        
        # 6. 测试三种模型
        all_results = []
        
        # 模型A: 单独Trait模型
        print(f"\n{'='*40}")
        print("模型A: 单独Trait模型")
        print(f"{'='*40}")
        
        # 准备单独Trait数据
        train_trait = train_data_merged.copy()
        val_trait = val_data_merged.copy()
        test_trait = test_data_merged.copy()
        
        # 标准化Trait特征
        exist_cont_features = [col for col in continuous_features if col in train_trait.columns]
        if exist_cont_features:
            scaler_trait = StandardScaler()
            scaler_trait.fit(train_trait[exist_cont_features])
            
            train_trait_scaled = train_trait.copy()
            val_trait_scaled = val_trait.copy()
            test_trait_scaled = test_trait.copy()
            
            train_trait_scaled[exist_cont_features] = scaler_trait.transform(train_trait[exist_cont_features])
            val_trait_scaled[exist_cont_features] = scaler_trait.transform(val_trait[exist_cont_features])
            test_trait_scaled[exist_cont_features] = scaler_trait.transform(test_trait[exist_cont_features])
        else:
            train_trait_scaled = train_trait.copy()
            val_trait_scaled = val_trait.copy()
            test_trait_scaled = test_trait.copy()
        
        # 训练单独Trait模型
        trait_result = train_and_evaluate_model(
            'Trait_Only', 
            trait_features, 
            train_trait_scaled, 
            val_trait_scaled, 
            test_trait_scaled, 
            'Trait_Only'
        )
        
        if trait_result:
            all_results.append(trait_result)
            trait_val_cindex = trait_result['Val_Cindex']
            trait_test_cindex = trait_result['Test_Cindex']
        else:
            trait_val_cindex = -np.inf
            trait_test_cindex = -np.inf
        
        # 模型B: 单独蛋白模型（动态添加，带停止规则）
        print(f"\n{'='*40}")
        print("模型B: 单独蛋白模型（动态添加，带停止规则）")
        print(f"{'='*40}")
        
        protein_only_results = []
        best_protein_only_val_cindex = -np.inf
        best_protein_only_count = 0
        best_protein_only_result = None
        
        # 停止规则相关变量
        last_val_cindex = None
        no_improvement_count = 0
        
        # 逐个添加蛋白，从1到全部可用
        max_proteins = len(available_proteins)
        for protein_count in range(1, max_proteins + 1):
            # 选择前protein_count个蛋白
            selected_proteins = available_proteins[:protein_count]
            
            # 准备蛋白数据
            train_protein = train_data_merged.copy()
            val_protein = val_data_merged.copy()
            test_protein = test_data_merged.copy()
            
            if selected_proteins:
                train_proteins = protein_df.loc[train_protein.index, selected_proteins].copy()
                val_proteins = protein_df.loc[val_protein.index, selected_proteins].copy()
                test_proteins = protein_df.loc[test_protein.index, selected_proteins].copy()
                
                # 重命名蛋白列
                protein_columns = [f"Protein_{p}" for p in selected_proteins]
                train_proteins.columns = protein_columns
                val_proteins.columns = protein_columns
                test_proteins.columns = protein_columns
                
                # 合并蛋白数据
                train_protein = pd.concat([train_protein, train_proteins], axis=1)
                val_protein = pd.concat([val_protein, val_proteins], axis=1)
                test_protein = pd.concat([test_protein, test_proteins], axis=1)
            
            # 标准化蛋白特征
            if selected_proteins:
                scaler_protein = StandardScaler()
                scaler_protein.fit(train_protein[protein_columns])
                
                train_protein_scaled = train_protein.copy()
                val_protein_scaled = val_protein.copy()
                test_protein_scaled = test_protein.copy()
                
                train_protein_scaled[protein_columns] = scaler_protein.transform(train_protein[protein_columns])
                val_protein_scaled[protein_columns] = scaler_protein.transform(val_protein[protein_columns])
                test_protein_scaled[protein_columns] = scaler_protein.transform(test_protein[protein_columns])
            else:
                train_protein_scaled = train_protein.copy()
                val_protein_scaled = val_protein.copy()
                test_protein_scaled = test_protein.copy()
            
            # 训练模型
            protein_result = train_and_evaluate_model(
                'Protein_Only',
                protein_columns if selected_proteins else [],
                train_protein_scaled,
                val_protein_scaled,
                test_protein_scaled,
                f'Protein_Only_Top{protein_count}'
            )
            
            if protein_result:
                protein_result['Protein_Count'] = protein_count
                protein_only_results.append(protein_result)
                
                # 更新最佳结果
                if protein_result['Val_Cindex'] > best_protein_only_val_cindex:
                    best_protein_only_val_cindex = protein_result['Val_Cindex']
                    best_protein_only_count = protein_count
                    best_protein_only_result = protein_result
                
                # 停止规则检查
                current_val = protein_result['Val_Cindex']
                if last_val_cindex is not None:
                    improvement = current_val - last_val_cindex
                    if improvement < IMPROVEMENT_THRESHOLD:
                        no_improvement_count += 1
                        print(f"    提升 {improvement:.6f} < {IMPROVEMENT_THRESHOLD}，连续无提升次数: {no_improvement_count}")
                    else:
                        no_improvement_count = 0
                        print(f"    提升 {improvement:.6f} >= {IMPROVEMENT_THRESHOLD}，重置计数器")
                    
                    # 如果连续无提升达到阈值，停止添加更多蛋白
                    if no_improvement_count >= PATIENCE:
                        print(f"    连续 {PATIENCE} 次提升小于 {IMPROVEMENT_THRESHOLD}，停止添加更多蛋白")
                        break
                last_val_cindex = current_val
            else:
                # 如果当前数量训练失败，仍然继续尝试下一个数量（可能因为特征缺失等原因）
                print(f"    蛋白数量 {protein_count} 训练失败，继续尝试下一个数量")
        
        # 将蛋白模型结果加入总结果
        all_results.extend(protein_only_results)
        
        # 模型C: Trait+蛋白模型（动态添加，带停止规则）
        print(f"\n{'='*40}")
        print("模型C: Trait+蛋白模型（动态添加，带停止规则）")
        print(f"{'='*40}")
        
        trait_protein_results = []
        best_trait_protein_val_cindex = -np.inf
        best_trait_protein_count = 0
        best_trait_protein_result = None
        
        # 重置停止规则变量
        last_val_cindex = None
        no_improvement_count = 0
        
        for protein_count in range(1, max_proteins + 1):
            # 选择前protein_count个蛋白
            selected_proteins = available_proteins[:protein_count]
            
            # 准备Trait+蛋白数据
            train_trait_protein = train_data_merged.copy()
            val_trait_protein = val_data_merged.copy()
            test_trait_protein = test_data_merged.copy()
            
            if selected_proteins:
                train_proteins = protein_df.loc[train_trait_protein.index, selected_proteins].copy()
                val_proteins = protein_df.loc[val_trait_protein.index, selected_proteins].copy()
                test_proteins = protein_df.loc[test_trait_protein.index, selected_proteins].copy()
                
                # 重命名蛋白列
                protein_columns = [f"Protein_{p}" for p in selected_proteins]
                train_proteins.columns = protein_columns
                val_proteins.columns = protein_columns
                test_proteins.columns = protein_columns
                
                # 合并蛋白数据
                train_trait_protein = pd.concat([train_trait_protein, train_proteins], axis=1)
                val_trait_protein = pd.concat([val_trait_protein, val_proteins], axis=1)
                test_trait_protein = pd.concat([test_trait_protein, test_proteins], axis=1)
            
            # 标准化所有特征
            features_to_scale = []
            if exist_cont_features:
                features_to_scale.extend(exist_cont_features)
            if selected_proteins:
                features_to_scale.extend(protein_columns)
            
            if features_to_scale:
                scaler_trait_protein = StandardScaler()
                scaler_trait_protein.fit(train_trait_protein[features_to_scale])
                
                train_trait_protein_scaled = train_trait_protein.copy()
                val_trait_protein_scaled = val_trait_protein.copy()
                test_trait_protein_scaled = test_trait_protein.copy()
                
                train_trait_protein_scaled[features_to_scale] = scaler_trait_protein.transform(train_trait_protein[features_to_scale])
                val_trait_protein_scaled[features_to_scale] = scaler_trait_protein.transform(val_trait_protein[features_to_scale])
                test_trait_protein_scaled[features_to_scale] = scaler_trait_protein.transform(test_trait_protein[features_to_scale])
            else:
                train_trait_protein_scaled = train_trait_protein.copy()
                val_trait_protein_scaled = val_trait_protein.copy()
                test_trait_protein_scaled = test_trait_protein.copy()
            
            # 创建特征列表
            if selected_proteins:
                features = trait_features + protein_columns
            else:
                features = trait_features
            
            # 训练模型
            trait_protein_result = train_and_evaluate_model(
                'Trait+Protein',
                features,
                train_trait_protein_scaled,
                val_trait_protein_scaled,
                test_trait_protein_scaled,
                f'Trait+Protein_Top{protein_count}'
            )
            
            if trait_protein_result:
                trait_protein_result['Protein_Count'] = protein_count
                trait_protein_results.append(trait_protein_result)
                
                # 更新最佳结果
                if trait_protein_result['Val_Cindex'] > best_trait_protein_val_cindex:
                    best_trait_protein_val_cindex = trait_protein_result['Val_Cindex']
                    best_trait_protein_count = protein_count
                    best_trait_protein_result = trait_protein_result
                
                # 停止规则检查
                current_val = trait_protein_result['Val_Cindex']
                if last_val_cindex is not None:
                    improvement = current_val - last_val_cindex
                    if improvement < IMPROVEMENT_THRESHOLD:
                        no_improvement_count += 1
                        print(f"    提升 {improvement:.6f} < {IMPROVEMENT_THRESHOLD}，连续无提升次数: {no_improvement_count}")
                    else:
                        no_improvement_count = 0
                        print(f"    提升 {improvement:.6f} >= {IMPROVEMENT_THRESHOLD}，重置计数器")
                    
                    if no_improvement_count >= PATIENCE:
                        print(f"    连续 {PATIENCE} 次提升小于 {IMPROVEMENT_THRESHOLD}，停止添加更多蛋白")
                        break
                last_val_cindex = current_val
            else:
                print(f"    蛋白数量 {protein_count} 训练失败，继续尝试下一个数量")
        
        # 将Trait+蛋白模型结果加入总结果
        all_results.extend(trait_protein_results)
        
        # 7. 汇总结果
        if not all_results:
            print("没有成功训练任何模型，跳过该疾病")
            return None
        
        # 创建结果DataFrame
        results_list = []
        for result in all_results:
            row = {
                'Disease': disease_name,
                'Model_Type': result['Model_Type'],
                'Model_Name': result['Model_Name'],
                'Protein_Count': result.get('Protein_Count', 0),
                'Features_Count': result['Features_Count'],
                'Penalizer': result['Penalizer'],
                'Train_Cindex': result['Train_Cindex'],
                'Val_Cindex': result['Val_Cindex'],
                'Test_Cindex': result['Test_Cindex']
            }
            results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        
        # 确定性能平台期（基于停止规则，但这里可以记录停止时的数量）
        protein_only_df = results_df[results_df['Model_Type'] == 'Protein_Only'].copy()
        if len(protein_only_df) > 0:
            protein_plateau_count = protein_only_df['Protein_Count'].max()
        else:
            protein_plateau_count = 0
        
        trait_protein_df = results_df[results_df['Model_Type'] == 'Trait+Protein'].copy()
        if len(trait_protein_df) > 0:
            trait_protein_plateau_count = trait_protein_df['Protein_Count'].max()
        else:
            trait_protein_plateau_count = 0
        
        # 8. 创建汇总
        summary = {
            'Disease_Type': data_path.split('/')[-2] if data_path.split('/')[-2] != 'data' else data_path.split('/')[-1],
            'Disease': disease_name,
            'Total_Available_Proteins': len(available_proteins),
            
            # Trait Only结果
            'Trait_Only_Val_Cindex': trait_val_cindex,
            'Trait_Only_Test_Cindex': trait_test_cindex,
            
            # Protein Only结果
            'Protein_Only_Best_Count': best_protein_only_count,
            'Protein_Only_Best_Val_Cindex': best_protein_only_val_cindex,
            'Protein_Only_Best_Test_Cindex': best_protein_only_result['Test_Cindex'] if best_protein_only_result else None,
            'Protein_Only_Plateau_Count': protein_plateau_count,
            'Protein_Only_Max_Test_Cindex': protein_only_df['Test_Cindex'].max() if len(protein_only_df) > 0 else None,
            
            # Trait+Protein结果
            'Trait_Protein_Best_Count': best_trait_protein_count,
            'Trait_Protein_Best_Val_Cindex': best_trait_protein_val_cindex,
            'Trait_Protein_Best_Test_Cindex': best_trait_protein_result['Test_Cindex'] if best_trait_protein_result else None,
            'Trait_Protein_Plateau_Count': trait_protein_plateau_count,
            'Trait_Protein_Max_Test_Cindex': trait_protein_df['Test_Cindex'].max() if len(trait_protein_df) > 0 else None,
            
            # 比较
            'Protein_vs_Trait_Improvement': (best_protein_only_val_cindex - trait_val_cindex) if best_protein_only_result else None,
            'Trait_Protein_vs_Trait_Improvement': (best_trait_protein_val_cindex - trait_val_cindex) if best_trait_protein_result else None,
            'Trait_Protein_vs_Protein_Improvement': (best_trait_protein_val_cindex - best_protein_only_val_cindex) if (best_trait_protein_result and best_protein_only_result) else None
        }
        
        for count in PROTEIN_COUNTS:
            # Protein Only
            protein_row = protein_only_df[protein_only_df['Protein_Count'] == count]
            if len(protein_row) > 0:
                summary[f'Protein_Only_Top{count}_Test'] = protein_row.iloc[0]['Test_Cindex']
            else:
                summary[f'Protein_Only_Top{count}_Test'] = None
            
            # Trait+Protein
            trait_protein_row = trait_protein_df[trait_protein_df['Protein_Count'] == count]
            if len(trait_protein_row) > 0:
                summary[f'Trait_Protein_Top{count}_Test'] = trait_protein_row.iloc[0]['Test_Cindex']
            else:
                summary[f'Trait_Protein_Top{count}_Test'] = None
        
        print(f"\n{'='*60}")
        print(f"疾病: {disease_name}")
        print(f"{'='*60}")
        print(f"1. Trait Only: Val={trait_val_cindex:.4f}, Test={trait_test_cindex:.4f}")
        print(f"2. Protein Only最佳: {best_protein_only_count}个蛋白, Val={best_protein_only_val_cindex:.4f}, Test={best_protein_only_result['Test_Cindex']:.4f}" if best_protein_only_result else "2. Protein Only: 无结果")
        print(f"3. Trait+Protein最佳: {best_trait_protein_count}个蛋白, Val={best_trait_protein_val_cindex:.4f}, Test={best_trait_protein_result['Test_Cindex']:.4f}" if best_trait_protein_result else "3. Trait+Protein: 无结果")
        print(f"   Protein Only停止数量: {protein_plateau_count}, Trait+Protein停止数量: {trait_protein_plateau_count}")
        
        # 9. 保存结果
        os.makedirs(output_path, exist_ok=True)
        
        detailed_file = os.path.join(output_path, f"all_models_comparison_{disease_name}.csv")
        results_df.to_csv(detailed_file, index=False)
        print(f"\n详细结果已保存到: {detailed_file}")
        
        create_comparison_plot(results_df, disease_name, output_path)
        
        return summary, results_df
        
    except Exception as e:
        print(f"处理疾病 {disease_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_comparison_plot(results_df, disease_name, output_path):
    """
    创建模型比较图
    """
    try:
        protein_only_df = results_df[results_df['Model_Type'] == 'Protein_Only'].copy()
        trait_protein_df = results_df[results_df['Model_Type'] == 'Trait+Protein'].copy()
        trait_only_df = results_df[results_df['Model_Type'] == 'Trait_Only'].copy()
        
        plt.figure(figsize=(12, 8))
        
        if len(protein_only_df) > 0:
            protein_only_df = protein_only_df.sort_values('Protein_Count')
            plt.plot(protein_only_df['Protein_Count'], protein_only_df['Test_Cindex'], 
                    'o-', color='blue', linewidth=2, markersize=8, label='Protein Only')
        
        if len(trait_protein_df) > 0:
            trait_protein_df = trait_protein_df.sort_values('Protein_Count')
            plt.plot(trait_protein_df['Protein_Count'], trait_protein_df['Test_Cindex'], 
                    's-', color='red', linewidth=2, markersize=8, label='Trait+Protein')
        
        if len(trait_only_df) > 0:
            trait_test_cindex = trait_only_df['Test_Cindex'].iloc[0]
            plt.axhline(y=trait_test_cindex, color='green', linestyle='--', 
                       linewidth=2, label='Trait Only')
        
        plt.xlabel('Number of Top Proteins', fontsize=14)
        plt.ylabel('Test C-index', fontsize=14)
        plt.title(f'Model Comparison for {disease_name}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        max_count = max(protein_only_df['Protein_Count'].max() if len(protein_only_df)>0 else 0,
                        trait_protein_df['Protein_Count'].max() if len(trait_protein_df)>0 else 0)
        if max_count > 0:
            plt.xticks(range(0, max_count+1, max(1, max_count//10)), fontsize=12)
        plt.yticks(fontsize=12)
        
        plot_file = os.path.join(output_path, f"model_comparison_{disease_name}.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"创建比较图时出错: {e}")

def process_disease_type_all_models(disease_type):
    """
    处理一个疾病类型的所有疾病，测试所有三种模型
    """
    print(f"\n{'='*80}")
    print(f"开始处理疾病类型: {disease_type}")
    print(f"{'='*80}")
    
    data_path = os.path.join(BASE_DATA_PATH, disease_type)
    ig_results_path = os.path.join(BASE_IG_RESULTS_PATH, disease_type)
    output_path = os.path.join(BASE_OUTPUT_PATH, disease_type, "all_models_analysis")
    
    os.makedirs(output_path, exist_ok=True)
    
    print("\n步骤1: 获取各疾病的蛋白重要性信息")
    print("-" * 40)
    
    exclude_folders = ["protein_frequency_analysis_results"]
    top_proteins_dict = {}
    
    try:
        disease_folders = [f for f in os.listdir(ig_results_path) 
                         if os.path.isdir(os.path.join(ig_results_path, f)) and f not in exclude_folders]
        
        for disease in sorted(disease_folders):
            file_path = os.path.join(ig_results_path, disease, "joint_protein_importance.csv")
            if not os.path.exists(file_path):
                continue
                
            try:
                df = pd.read_csv(file_path)
                top_proteins = df.sort_values('importance', ascending=False)['protein'].tolist()
                top_proteins_dict[disease] = top_proteins
            except Exception as e:
                continue
    except Exception as e:
        print(f"扫描文件夹时出错: {e}")
        return None, None
    
    print("\n" + "=" * 80)
    print("步骤2: 处理h5ad文件并测试所有模型")
    print("=" * 80)
    
    h5ad_files = glob.glob(os.path.join(data_path, "*.h5ad"))
    
    if not h5ad_files:
        print(f"在目录 {data_path} 中没有找到任何.h5ad文件！")
        return None, None
    
    all_summaries = []
    all_detailed_results = []
    processed_count = 0
    
    for h5ad_file in sorted(h5ad_files):
        file_name = os.path.basename(h5ad_file)
        
        # 💡 新增：在这里触发 SKIP_FILES 判断
        if file_name in SKIP_FILES:
            print(f"\n⏭️ 根据规则跳过未过滤的文件: {file_name}")
            continue
            
        disease_name = os.path.splitext(file_name)[0]
        summary, detailed_results = process_disease_all_models(
            disease_name, file_name, data_path, TRAIT_PATH, top_proteins_dict, output_path
        )
        
        if summary is not None:
            all_summaries.append(summary)
            detailed_results['Disease_Type'] = disease_type
            all_detailed_results.append(detailed_results)
            processed_count += 1
    
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        detailed_df = pd.concat(all_detailed_results, ignore_index=True)
        
        summary_file = os.path.join(output_path, f"all_models_summary_{disease_type}.csv")
        summary_df.to_csv(summary_file, index=False)
        
        detailed_file = os.path.join(output_path, f"all_models_detailed_{disease_type}.csv")
        detailed_df.to_csv(detailed_file, index=False)
        
        return summary_df, detailed_df
    else:
        return None, None

def create_final_summary_table(all_summaries):
    """
    创建最终的汇总表格
    """
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
            'Avg_Protein_Only_Best_Count': type_results['Protein_Only_Best_Count'].mean(),
            'Std_Protein_Only_Best_Count': type_results['Protein_Only_Best_Count'].std(),
            'Median_Protein_Only_Best_Count': type_results['Protein_Only_Best_Count'].median(),
            'Avg_Protein_Only_Test_Cindex': type_results['Protein_Only_Best_Test_Cindex'].mean(),
            'Protein_Only_Better_Than_Trait': (type_results['Protein_Only_Best_Test_Cindex'] > type_results['Trait_Only_Test_Cindex']).sum(),
            'Protein_Only_Better_Percent': (type_results['Protein_Only_Best_Test_Cindex'] > type_results['Trait_Only_Test_Cindex']).sum() / len(type_results) * 100,
            'Avg_Trait_Protein_Best_Count': type_results['Trait_Protein_Best_Count'].mean(),
            'Std_Trait_Protein_Best_Count': type_results['Trait_Protein_Best_Count'].std(),
            'Median_Trait_Protein_Best_Count': type_results['Trait_Protein_Best_Count'].median(),
            'Avg_Trait_Protein_Test_Cindex': type_results['Trait_Protein_Best_Test_Cindex'].mean(),
            'Trait_Protein_Better_Than_Trait': (type_results['Trait_Protein_Best_Test_Cindex'] > type_results['Trait_Only_Test_Cindex']).sum(),
            'Trait_Protein_Better_Percent': (type_results['Trait_Protein_Best_Test_Cindex'] > type_results['Trait_Only_Test_Cindex']).sum() / len(type_results) * 100,
            'Trait_Protein_Better_Than_Protein': (type_results['Trait_Protein_Best_Test_Cindex'] > type_results['Protein_Only_Best_Test_Cindex']).sum(),
            'Trait_Protein_Better_Than_Protein_Percent': (type_results['Trait_Protein_Best_Test_Cindex'] > type_results['Protein_Only_Best_Test_Cindex']).sum() / len(type_results) * 100,
            'Avg_Protein_vs_Trait_Improvement': type_results['Protein_vs_Trait_Improvement'].mean(),
            'Avg_Trait_Protein_vs_Trait_Improvement': type_results['Trait_Protein_vs_Trait_Improvement'].mean(),
            'Avg_Trait_Protein_vs_Protein_Improvement': type_results['Trait_Protein_vs_Protein_Improvement'].mean()
        }
        
        protein_only_counts = type_results['Protein_Only_Best_Count']
        trait_protein_counts = type_results['Trait_Protein_Best_Count']
        
        for count in PROTEIN_COUNTS:
            type_stats[f'Protein_Only_Count_{count}'] = (protein_only_counts == count).sum()
            type_stats[f'Trait_Protein_Count_{count}'] = (trait_protein_counts == count).sum()
        
        final_summary.append(type_stats)
    
    overall_stats = {
        'Disease_Type': 'Overall',
        'Number_of_Diseases': len(all_summaries_df),
        'Avg_Trait_Only_Test_Cindex': all_summaries_df['Trait_Only_Test_Cindex'].mean(),
        'Avg_Protein_Only_Best_Count': all_summaries_df['Protein_Only_Best_Count'].mean(),
        'Std_Protein_Only_Best_Count': all_summaries_df['Protein_Only_Best_Count'].std(),
        'Median_Protein_Only_Best_Count': all_summaries_df['Protein_Only_Best_Count'].median(),
        'Avg_Protein_Only_Test_Cindex': all_summaries_df['Protein_Only_Best_Test_Cindex'].mean(),
        'Protein_Only_Better_Than_Trait': (all_summaries_df['Protein_Only_Best_Test_Cindex'] > all_summaries_df['Trait_Only_Test_Cindex']).sum(),
        'Protein_Only_Better_Percent': (all_summaries_df['Protein_Only_Best_Test_Cindex'] > all_summaries_df['Trait_Only_Test_Cindex']).sum() / len(all_summaries_df) * 100,
        'Avg_Trait_Protein_Best_Count': all_summaries_df['Trait_Protein_Best_Count'].mean(),
        'Std_Trait_Protein_Best_Count': all_summaries_df['Trait_Protein_Best_Count'].std(),
        'Median_Trait_Protein_Best_Count': all_summaries_df['Trait_Protein_Best_Count'].median(),
        'Avg_Trait_Protein_Test_Cindex': all_summaries_df['Trait_Protein_Best_Test_Cindex'].mean(),
        'Trait_Protein_Better_Than_Trait': (all_summaries_df['Trait_Protein_Best_Test_Cindex'] > all_summaries_df['Trait_Only_Test_Cindex']).sum(),
        'Trait_Protein_Better_Percent': (all_summaries_df['Trait_Protein_Best_Test_Cindex'] > all_summaries_df['Trait_Only_Test_Cindex']).sum() / len(all_summaries_df) * 100,
        'Trait_Protein_Better_Than_Protein': (all_summaries_df['Trait_Protein_Best_Test_Cindex'] > all_summaries_df['Protein_Only_Best_Test_Cindex']).sum(),
        'Trait_Protein_Better_Than_Protein_Percent': (all_summaries_df['Trait_Protein_Best_Test_Cindex'] > all_summaries_df['Protein_Only_Best_Test_Cindex']).sum() / len(all_summaries_df) * 100,
        'Avg_Protein_vs_Trait_Improvement': all_summaries_df['Protein_vs_Trait_Improvement'].mean(),
        'Avg_Trait_Protein_vs_Trait_Improvement': all_summaries_df['Trait_Protein_vs_Trait_Improvement'].mean(),
        'Avg_Trait_Protein_vs_Protein_Improvement': all_summaries_df['Trait_Protein_vs_Protein_Improvement'].mean()
    }
    
    protein_only_counts = all_summaries_df['Protein_Only_Best_Count']
    trait_protein_counts = all_summaries_df['Trait_Protein_Best_Count']
    
    for count in PROTEIN_COUNTS:
        overall_stats[f'Protein_Only_Count_{count}'] = (protein_only_counts == count).sum()
        overall_stats[f'Trait_Protein_Count_{count}'] = (trait_protein_counts == count).sum()
    
    final_summary.append(overall_stats)
    
    final_summary_df = pd.DataFrame(final_summary)
    
    return final_summary_df

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    print("=" * 100)
    print("开始测试所有三种模型的Cox回归性能（动态添加蛋白，带停止规则）")
    print(f"蛋白添加范围: 1 到 {max(PROTEIN_COUNTS)}")
    print(f"停止规则: 连续 {PATIENCE} 次提升小于 {IMPROVEMENT_THRESHOLD} 时停止")
    print(f"处理疾病类型: {', '.join(DISEASE_TYPES)}")
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
        all_summaries_file = os.path.join(BASE_OUTPUT_PATH, f"all_diseases_all_models_summary_{timestamp}.csv")
        all_summaries_df.to_csv(all_summaries_file, index=False)
        
        all_detailed_df = pd.concat(all_detailed_results, ignore_index=True)
        all_detailed_file = os.path.join(BASE_OUTPUT_PATH, f"all_diseases_all_models_detailed_{timestamp}.csv")
        all_detailed_df.to_csv(all_detailed_file, index=False)
        
        final_summary_table = create_final_summary_table(all_summaries)
        if final_summary_table is not None:
            final_summary_file = os.path.join(BASE_OUTPUT_PATH, f"final_all_models_summary_table_{timestamp}.csv")
            final_summary_table.to_csv(final_summary_file, index=False)
            
            print(f"\n\n{'='*100}")
            print("所有疾病类型处理完成！")
            print(f"{'='*100}")
            print(f"\n汇总结果文件:")
            print(f"1. 所有疾病汇总结果: {all_summaries_file}")
            print(f"2. 所有疾病详细结果: {all_detailed_file}")
            print(f"3. 最终汇总表格: {final_summary_file}")
            
if __name__ == "__main__":
    main()


#%%
##############################################################
#逐步挑选lifestyle
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
#DISEASE_TYPES = ['Cardiovascular','Benign_neoplasm_or_Carcinoma_in_situ','Ear','Endocrine','Eye','Neurological','Musculoskeletal','Genitourinary','Haematological_or_immunological','Respiratory','Skin']
#DISEASE_TYPES = ['Cancers', 'Digestive', 'Infections', 'Psychiatric']
#DISEASE_TYPES = ['Cardiovascular','Benign_neoplasm_or_Carcinoma_in_situ','Ear','Endocrine','Eye','Neurological','Musculoskeletal']
DISEASE_TYPES = ['Cancers', 'Digestive', 'Infections', 'Psychiatric','Genitourinary','Haematological_or_immunological','Respiratory','Skin']



# 基础路径配置
BASE_DATA_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
LIFESTYLE_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/clinic_drop20_10_9.csv"
BASE_OUTPUT_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_lifestyle_0320"

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
            print(f"  结果: Val C-index={trait_val_cindex:.4f}, Test C-index={trait_test_cindex:.4f}")
        else:
            trait_val_cindex = -np.inf
            trait_test_cindex = -np.inf
        
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
                        print(f"    提升 {improvement:.6f} < {IMPROVEMENT_THRESHOLD}，连续无提升次数: {no_improvement_count}")
                    else:
                        no_improvement_count = 0
                        print(f"    提升 {improvement:.6f} >= {IMPROVEMENT_THRESHOLD}，重置计数器")
                    
                    if no_improvement_count >= PATIENCE:
                        print(f"    连续 {PATIENCE} 次提升小于 {IMPROVEMENT_THRESHOLD}，触发停止规则")
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
        
        summary = {
            'Disease_Type': data_path.split('/')[-2] if data_path.split('/')[-2] != 'data' else data_path.split('/')[-1],
            'Disease': disease_name,
            'Total_Available_Lifestyles': max_lifestyles,
            'Trait_Only_Val_Cindex': trait_val_cindex,
            'Trait_Only_Test_Cindex': trait_test_cindex,
            'Trait_LS_Best_Count': best_trait_ls_count,
            'Trait_LS_Best_Val_Cindex': best_trait_ls_val_cindex,
            'Trait_LS_Best_Test_Cindex': best_trait_ls_result['Test_Cindex'] if best_trait_ls_result else None,
            'Trait_LS_Plateau_Count': tl_plateau_count,
            'Trait_LS_Max_Test_Cindex': tl_df['Test_Cindex'].max() if len(tl_df) > 0 else None,
            'Trait_LS_vs_Trait_Improvement': (best_trait_ls_val_cindex - trait_val_cindex) if best_trait_ls_result else None
        }
        
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
                
                # 过滤掉 age, sex, bmi，避免这些直接被拉到动态添加队列
                exclude_set = {'age', 'sex', 'bmi'}
                top_lifestyles_dict[disease] = [f for f in top_features if str(f).lower() not in exclude_set]
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


#nohup python -u /home/xuln/olink_disease_predict/code/fig4_67trait_with_select_clinic_life_0320.py > /home/xuln/olink_disease_predict/code/fig4_67trait_with_select_clinic_life_0320.log 2>&1 &

#%%
#混合clinic+lifestyle+protein

# fig4_combined_trait_lifestyle_protein.py
# fig4_combined_trait_lifestyle_protein.py

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

# ==============================================================================
# 1. 配置参数
# ==============================================================================
DISEASE_TYPES = [
    'Cardiovascular', 'Benign_neoplasm_or_Carcinoma_in_situ', 'Ear', 'Endocrine', 
    'Eye', 'Neurological', 'Musculoskeletal', 'Genitourinary', 
    'Haematological_or_immunological', 'Respiratory', 'Skin',
    'Cancers', 'Digestive', 'Infections', 'Psychiatric'
]

# 基础路径配置
BASE_DATA_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"
TRAIT_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"
LIFESTYLE_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/clinic_drop20_10_9.csv"

# 之前合并好的最佳特征数量文件
BEST_COUNTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_lifestyle_0320/finally111_diseases_best_counts_lifestyle_and_protein.csv"

# 最终输出路径
BASE_OUTPUT_PATH = "/bigdat2/user/xuln/olink_disease_predict/save_models/final_combined_trait_ls_protein"

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

# ==============================================================================
# 2. 特征定义
# ==============================================================================
categorical_features = ['31']

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

LIFESTYLE_CONTINUOUS_VARS = [
    '874','884','904','864','1090','1080','1070','1160','1269','1279',
    '1289','1299','1309','1319','1438','1458','1488','1498','1528','1050',
    '1060','2277','2139','2149','age','bmi'
]

# ==============================================================================
# 3. 核心函数定义
# ==============================================================================

def train_cox_model(model_name, feature_list, train_df, val_df, initial_penalizer=0.01):
    cols = feature_list + ['time', 'event']
    df_train_sub = train_df[cols].dropna().copy()
    df_val_sub = val_df[cols].dropna().copy()
    
    if len(df_train_sub) == 0:
        return None
    
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

def train_and_evaluate_model(model_type, feature_list, train_df, val_df, test_df, model_name):
    model_result = train_cox_model(model_name, feature_list, train_df, val_df)
    if model_result is None:
        return None
        
    model, penalizer, train_cindex, val_cindex = model_result
    test_cindex = calculate_test_cindex(model, test_df, feature_list)
    
    if test_cindex is None:
        return None
        
    return {
        'Model_Type': model_type,
        'Model_Name': model_name,
        'Features_Count': len(feature_list),
        'Penalizer': penalizer,
        'Train_Cindex': train_cindex,
        'Val_Cindex': val_cindex,
        'Test_Cindex': test_cindex
    }

def get_feature_importances(disease_name, disease_type):
    """读取并返回该疾病的生活习惯和蛋白重要性排序列表"""
    disease_key = disease_name.replace('.h5ad', '').replace('_survival_data', '').replace(',', '').replace('&', 'and').replace(' ', '_').replace('-', '_')
    possible_keys = [
        disease_key,
        disease_key[0] + disease_key[1:].lower() if disease_key else disease_key,
        disease_key.lower(),
        disease_key.upper()
    ]
    
    ls_features, prot_features = [], []
    
    # 查找生活习惯特征
    for key in possible_keys:
        ls_file = os.path.join(BASE_IG_RESULTS_PATH, disease_type, key, "joint_clinical_importance.csv")
        if os.path.exists(ls_file):
            df_ls = pd.read_csv(ls_file)
            col_name = 'feature' if 'feature' in df_ls.columns else 'lifestyle'
            exclude_set = {'age', 'sex', 'bmi'}
            all_ls = df_ls.sort_values('importance', ascending=False)[col_name].tolist()
            ls_features = [str(f) for f in all_ls if str(f).lower() not in exclude_set]
            break
            
    # 查找蛋白特征
    for key in possible_keys:
        prot_file = os.path.join(BASE_IG_RESULTS_PATH, disease_type, key, "joint_protein_importance.csv")
        if os.path.exists(prot_file):
            df_prot = pd.read_csv(prot_file)
            prot_features = df_prot.sort_values('importance', ascending=False)['protein'].tolist()
            break
            
    return ls_features, prot_features

# ==============================================================================
# 4. 主流程逻辑
# ==============================================================================

def main():
    print("=" * 80)
    print("🚀 开始执行最终联合模型: Trait + Lifestyle + Protein")
    print("=" * 80)
    
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    
    # 1. 读取最佳特征数量文件
    best_counts_df = pd.read_csv(BEST_COUNTS_PATH)
    print(f"✅ 成功读取最佳特征数量文件，包含 {len(best_counts_df)} 个疾病配置。")
    
    # 2. 读取基础特征数据
    print("📦 正在加载全局 Trait 和 Lifestyle 数据...")
    trait_df = pd.read_csv(TRAIT_PATH)
    trait_df.rename(columns={'eid': 'participant.eid'}, inplace=True)
    trait_df['participant.eid'] = trait_df['participant.eid'].astype(str)
    
    lifestyle_df = pd.read_csv(LIFESTYLE_PATH)
    if 'eid' in lifestyle_df.columns:
        lifestyle_df.rename(columns={'eid': 'participant.eid'}, inplace=True)
    lifestyle_df['participant.eid'] = lifestyle_df['participant.eid'].astype(str)
    
    # 合并基础特征
    global_features_df = pd.merge(trait_df, lifestyle_df, on='participant.eid', how='outer')
    global_features_df.columns = [str(c) for c in global_features_df.columns]
    print(f"✅ 全局特征合并完成，形状: {global_features_df.shape}")
    
    all_results = []
    
    # 3. 遍历疾病
    for disease_type in DISEASE_TYPES:
        data_path = os.path.join(BASE_DATA_PATH, disease_type)
        if not os.path.exists(data_path):
            continue
            
        h5ad_files = glob.glob(os.path.join(data_path, "*.h5ad"))
        for h5ad_file in h5ad_files:
            file_name = os.path.basename(h5ad_file)
            if file_name in SKIP_FILES:
                continue
                
            disease_name = os.path.splitext(file_name)[0]
            
            # 从合并表中寻找该疾病的配置
            disease_config = best_counts_df[best_counts_df['Disease'] == disease_name]
            # --- 新增逻辑：如果没有找到，并且名字里带有 '_filtered'，就去掉后缀再查一次 ---
            if len(disease_config) == 0 and '_filtered' in disease_name:
                alt_disease_name = disease_name.replace('_filtered', '')
                disease_config = best_counts_df[best_counts_df['Disease'] == alt_disease_name]
                if len(disease_config) > 0:
                    print(f"🔍 提示: 使用备用名称 '{alt_disease_name}' 成功匹配到配置。")
            
            # 如果去掉了后缀还是找不到（或者本来就不带后缀也找不到），则跳过
            if len(disease_config) == 0:
                print(f"⚠️ 跳过 {disease_name}：在配置表中未找到该疾病。")
                continue
                
            ls_target_count = int(disease_config.iloc[0]['Trait_LS_Best_Count'])
            prot_target_count = int(disease_config.iloc[0]['Trait_Protein_Best_Count'])
            
            print(f"\n[{disease_type}] -> {disease_name}")
            print(f"   目标配置: 生活习惯 {ls_target_count} 个, 蛋白 {prot_target_count} 个")
            
            # 读取特征排序
            ls_ranked, prot_ranked = get_feature_importances(disease_name, disease_type)
            if not ls_ranked or not prot_ranked:
                print(f"   ❌ 无法找到特征重要性文件，跳过。")
                continue
            
            # 读取 h5ad
            adata = sc.read_h5ad(h5ad_file)
            if 'event' not in adata.obs.columns or adata.obs['event'].sum() < 5:
                print(f"   ❌ 事件数不足，跳过。")
                continue
                
            # 划分数据集
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
            
            # 合并全局特征
            train_merged = pd.merge(train_data, global_features_df, on='participant.eid', how='left').set_index('participant.eid')
            val_merged = pd.merge(val_data, global_features_df, on='participant.eid', how='left').set_index('participant.eid')
            test_merged = pd.merge(test_data, global_features_df, on='participant.eid', how='left').set_index('participant.eid')
            
            # 提取蛋白特征
            protein_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            protein_df = pd.DataFrame(protein_matrix, index=adata.obs.index, columns=adata.var_names)
            
            # 组装最终特征列表
            is_filtered_disease = '_filtered' in disease_name
            current_cat_features = categorical_features.copy()
            if is_filtered_disease and '31' in current_cat_features:
                current_cat_features.remove('31')
                
            available_traits = [str(c) for c in current_cat_features + continuous_features if str(c) in global_features_df.columns]
            
            available_ls = [p for p in ls_ranked if p in global_features_df.columns]
            if is_filtered_disease and 'sex' in available_ls:
                available_ls.remove('sex')
            selected_ls = available_ls[:ls_target_count]
            
            available_prot = [p for p in prot_ranked if p in protein_df.columns]
            selected_prot = available_prot[:prot_target_count]
            
            # 将蛋白合并进数据
            if selected_prot:
                train_merged = pd.concat([train_merged, protein_df.loc[train_merged.index, selected_prot]], axis=1)
                val_merged = pd.concat([val_merged, protein_df.loc[val_merged.index, selected_prot]], axis=1)
                test_merged = pd.concat([test_merged, protein_df.loc[test_merged.index, selected_prot]], axis=1)
                
            all_final_features = available_traits + selected_ls + selected_prot
            
            # 确定哪些需要标准化
            features_to_scale = []
            features_to_scale.extend([f for f in continuous_features if f in train_merged.columns])
            features_to_scale.extend([p for p in selected_ls if p in LIFESTYLE_CONTINUOUS_VARS])
            features_to_scale.extend(selected_prot) # 蛋白全标准化
            
            # 唯一化，防止重复
            features_to_scale = list(set(features_to_scale))
            
            # 执行标准化
            if features_to_scale:
                scaler = StandardScaler()
                train_merged[features_to_scale] = scaler.fit_transform(train_merged[features_to_scale])
                val_merged[features_to_scale] = scaler.transform(val_merged[features_to_scale])
                test_merged[features_to_scale] = scaler.transform(test_merged[features_to_scale])
                
            # 训练模型：Trait Only (作为基线对比)
            trait_result = train_and_evaluate_model('Trait_Only', available_traits, train_merged, val_merged, test_merged, 'Baseline_Trait')
            
            # 训练模型：Trait + LS + Protein
            combined_result = train_and_evaluate_model('Trait+LS+Protein', all_final_features, train_merged, val_merged, test_merged, 'Final_Combined_Model')
            
            if trait_result and combined_result:
                print(f"   ✔️ 训练完成！")
                print(f"      Trait Only Test C-index: {trait_result['Test_Cindex']:.4f}")
                print(f"      Trait+LS+Protein Test C-index: {combined_result['Test_Cindex']:.4f}")
                
                # =============== 主要修改部分：增加了 Penalizer 字段 ===============
                all_results.append({
                    'Disease_Type': disease_type,
                    'Disease': disease_name,
                    'LS_Count': ls_target_count,
                    'Protein_Count': prot_target_count,
                    'Total_Features_Count': len(all_final_features),
                    'Trait_Only_Penalizer': trait_result['Penalizer'],     # 新增字段：基础模型正则化系数
                    'Trait_Only_Val_Cindex': trait_result['Val_Cindex'],
                    'Trait_Only_Test_Cindex': trait_result['Test_Cindex'],
                    'Combined_Penalizer': combined_result['Penalizer'],    # 新增字段：联合模型正则化系数
                    'Combined_Val_Cindex': combined_result['Val_Cindex'],
                    'Combined_Test_Cindex': combined_result['Test_Cindex'],
                    'Improvement_over_Trait': combined_result['Test_Cindex'] - trait_result['Test_Cindex']
                })
                # =================================================================
            else:
                print(f"   ❌ 模型拟合失败。")

    # 5. 保存最终汇总结果
    if all_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        final_df = pd.DataFrame(all_results)
        output_file = os.path.join(BASE_OUTPUT_PATH, f"final_combined_models_summary_{timestamp}.csv")
        final_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print("🎉 所有任务执行完毕！")
        print(f"共成功运行 {len(final_df)} 个疾病的终极模型。")
        print(f"平均 C-index (Trait Only): {final_df['Trait_Only_Test_Cindex'].mean():.4f}")
        print(f"平均 C-index (联合模型): {final_df['Combined_Test_Cindex'].mean():.4f}")
        print(f"结果已保存至: {output_file}")
        print("="*80)
    else:
        print("\n没有成功生成任何结果。")

if __name__ == "__main__":
    main()


#nohup python -u /home/xuln/olink_disease_predict/code/fig4_run.py > /home/xuln/olink_disease_predict/code/fig4_select_trait_lifestyle_protein_0514.log 2>&1 &
