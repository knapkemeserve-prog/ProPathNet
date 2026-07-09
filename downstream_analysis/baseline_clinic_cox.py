# %%
#插补后的67个trait
#/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv

# Cox比例风险模型分析
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

# ==============================================================================
# 配置参数 - 根据需要修改这些参数
# ==============================================================================
# 父目录路径 - 所有疾病类型都在这个目录下
PARENT_PATH = "/bigdat2/user/xuln/olink_disease_predict/data/"
PANEL_PATH = "/bigdat2/user/xuln/olink_disease_predict/comparision_with_clinical_predictor/MILTON_features_imputed.csv"

# 需要处理的疾病类型列表
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

# 输出目录 
OUTPUT_DIR = "/bigdat2/user/xuln/olink_disease_predict/67traits_cox_analysis"

# 设置随机种子以确保可重复性
RANDOM_SEED = 42

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

# 模型特征定义
Base_features = ['21003', '31']  # 基础特征：年龄和性别

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

# 模型对应的初始正则化系数
model_penalizers = {
    'Base': 0.0,      # Base模型初始正则化系数为0
    'Physical_examination': 0.01     # 体检模型初始正则化系数为0.01
}

# 如果C-index小于0.5时尝试的正则化系数列表
penalty_search_list = [0.01, 0.05, 0.1]

# ==============================================================================
# 函数定义
# ==============================================================================

def save_test_predictions(model, model_name, test_df, feature_list, 
                         disease_name, category_name, output_dir):
    """
    保存测试集的预测风险值到CSV文件
    """
    # 创建疾病类型目录
    category_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_dir, exist_ok=True)
    
    # 创建疾病目录
    disease_dir = os.path.join(category_dir, disease_name)
    os.makedirs(disease_dir, exist_ok=True)
    
    # 准备测试集数据（删除缺失值）
    cols = feature_list + ['time', 'event']
    df_test_sub = test_df[cols].dropna().copy()
    
    if len(df_test_sub) == 0:
        print(f"    警告: 测试集数据为空（所有样本均含缺失值），无法生成预测")
        return None
    
    try:
        # 获取预测风险值（部分风险）
        risk_scores = model.predict_partial_hazard(df_test_sub).values.flatten()
        
        # 创建包含所有信息的DataFrame
        predictions_df = pd.DataFrame({
            'participant.eid': df_test_sub.index,
            'time': df_test_sub['time'].values,
            'event': df_test_sub['event'].values,
            'risk_score': risk_scores
        })
        
        # 按照风险值排序
        predictions_df = predictions_df.sort_values('risk_score', ascending=False)
        
        # 保存到CSV文件
        output_file = os.path.join(disease_dir, f"{model_name}_predictions.csv")
        predictions_df.to_csv(output_file, index=False)
        
        print(f"    预测结果已保存到: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"    保存预测结果时出错: {e}")
        return None

def train_eval_cox_with_penalty_search(model_name, feature_list, train_df, val_df, test_df, 
                                     disease_name, category_name, output_dir):
    """
    训练Cox模型并评估性能，根据模型名称使用不同的正则化系数
    如果验证集C-index小于0.5，尝试不同的正则化系数
    同时保存测试集的预测风险值
    """
    cols = feature_list + ['time', 'event']
    
    # 预处理：删除缺失值
    df_train_sub = train_df[cols].dropna().copy()
    df_val_sub = val_df[cols].dropna().copy()
    
    if len(df_train_sub) == 0:
        print(f"    训练集为空 (所有样本均含缺失值)，跳过。")
        return None, None
    
    # 获取初始正则化系数
    initial_penalizer = model_penalizers.get(model_name, 0.01)
    
    # 尝试的正则化系数列表（初始系数优先）
    penalizers_to_try = [initial_penalizer] + [p for p in penalty_search_list if p != initial_penalizer]
    
    best_model = None
    best_val_cindex = -np.inf
    best_penalizer = initial_penalizer
    
    print(f"    模型 '{model_name}' - 尝试正则化系数: {penalizers_to_try}")
    
    for penalizer in penalizers_to_try:
        try:
            # 训练模型
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df_train_sub, duration_col='time', event_col='event')
            
            # 计算验证集C-index
            c_val = cph.score(df_val_sub, scoring_method="concordance_index")
            
            print(f"    正则化系数 {penalizer:.3f} - 验证集C-index: {c_val:.4f}")
            
            # 如果C-index小于0.5，继续尝试其他系数
            if c_val < 0.5:
                # 但仍记录最佳模型
                if c_val > best_val_cindex:
                    best_val_cindex = c_val
                    best_model = cph
                    best_penalizer = penalizer
                continue
            
            # 如果C-index大于等于0.5，使用当前模型
            best_model = cph
            best_val_cindex = c_val
            best_penalizer = penalizer
            break
            
        except Exception as e:
            print(f"    正则化系数 {penalizer:.3f} 训练失败: {e}")
            continue
    
    # 如果没有找到C-index>=0.5的模型，使用最佳模型（即使C-index<0.5）
    if best_model is None:
        print(f"    所有正则化系数均失败，跳过模型 '{model_name}'")
        return None, None
    
    print(f"    选择正则化系数 {best_penalizer:.3f} - 验证集C-index: {best_val_cindex:.4f}")
    
    # 计算训练集C-index
    c_train = best_model.score(df_train_sub, scoring_method="concordance_index")
    
    # 计算测试集C-index
    df_test_sub = test_df[cols].copy()
    df_test_sub_clean = df_test_sub.dropna()
    
    if len(df_test_sub_clean) > 0:
        # 获取预测风险值（部分风险）
        risk_scores = best_model.predict_partial_hazard(df_test_sub_clean).values.flatten()
        
        # 计算C-index
        c_test = concordance_index(
            df_test_sub_clean['time'].values,
            -risk_scores,  # 注意：concordance_index需要负的风险分数
            df_test_sub_clean['event'].values
        )
        
        # 保存测试集预测结果
        save_test_predictions(
            best_model, model_name, test_df, feature_list,
            disease_name, category_name, output_dir
        )
    else:
        print(f"    测试集数据为空（所有样本均含缺失值），无法计算C-index")
        c_test = np.nan
    
    return best_model, {
        'Model': model_name, 
        'Penalizer': best_penalizer,
        'Train C-index': c_train, 
        'Val C-index': best_val_cindex, 
        'Test C-index': c_test
    }

def process_disease_category(category_name, base_path, panel_path, model_dict, output_dir):
    """
    处理单个疾病类型（目录）下的所有疾病
    """
    print(f"\n{'='*80}")
    print(f"开始处理疾病类型: {category_name}")
    print(f"路径: {base_path}")
    print(f"{'='*80}")
    
    # 扫描目录下的所有h5ad文件
    h5ad_files = glob.glob(os.path.join(base_path, "*.h5ad"))
    
    if not h5ad_files:
        print(f"在目录 {base_path} 中没有找到任何.h5ad文件！跳过该类型。")
        return []
    
    print(f"找到 {len(h5ad_files)} 个h5ad文件:")
    for file in h5ad_files[:10]:  # 只显示前10个文件
        print(f"  - {os.path.basename(file)}")
    if len(h5ad_files) > 10:
        print(f"  ... 还有 {len(h5ad_files)-10} 个文件")
    
    all_results = []
    successful_diseases = 0
    
    # 处理每个疾病文件
    for h5ad_file in h5ad_files:
        # 从文件名提取疾病名称（去除扩展名）
        disease_name = os.path.splitext(os.path.basename(h5ad_file))[0]
        file_name = os.path.basename(h5ad_file)
        
        print(f"\n{'='*60}")
        print(f"处理疾病: {disease_name}")
        print(f"文件: {file_name}")
        print(f"{'='*60}")
        
        try:
            # 1. 加载数据
            print(f"加载数据: {h5ad_file}")
            adata = sc.read_h5ad(h5ad_file)
            print(f"数据形状: {adata.shape}")
            
            # 检查是否有足够的事件
            event_counts = adata.obs['event'].value_counts()
            print(f"事件统计: {event_counts.to_dict()}")
            
            # 如果事件太少，跳过这个疾病
            if 1 not in event_counts or event_counts[1] < 5:  # 降低阈值到5个事件
                print(f"⚠️  事件数太少 ({event_counts.get(1, 0)}个事件)，跳过这个疾病")
                continue
            
            # 2. 准备生存数据
            survival_time = adata.obs['time'].values
            event_status = adata.obs['event'].values
            patient_ids = adata.obs.index.tolist()
            
            # 3. 划分数据集
            indices = list(range(len(patient_ids)))
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
            
            # 4. 加载临床数据并合并
            print("加载临床数据并合并...")
            Clinical = pd.read_csv(panel_path)
            Clinical['eid'] = Clinical['eid'].astype(str)
            
            train_data = train_data.copy()
            val_data = val_data.copy()
            test_data = test_data.copy()
            
            train_data['eid'] = train_data.index.astype(str)
            val_data['eid'] = val_data.index.astype(str)
            test_data['eid'] = test_data.index.astype(str)
            
            train_data_merged = pd.merge(train_data, Clinical, on='eid', how='left')
            val_data_merged = pd.merge(val_data, Clinical, on='eid', how='left')
            test_data_merged = pd.merge(test_data, Clinical, on='eid', how='left')
            
            train_data_merged.set_index('eid', inplace=True)
            val_data_merged.set_index('eid', inplace=True)
            test_data_merged.set_index('eid', inplace=True)
            
            print(f"合并后形状: Train={train_data_merged.shape}, Val={val_data_merged.shape}, Test={test_data_merged.shape}")
            
            # 5. 数据标准化
            print("数据标准化...")
            exist_cont_features = [col for col in continuous_features if col in train_data_merged.columns]
            
            scaler = StandardScaler()
            scaler.fit(train_data_merged[exist_cont_features])
            
            train_data_scaled = train_data_merged.copy()
            val_data_scaled = val_data_merged.copy()
            test_data_scaled = test_data_merged.copy()
            
            train_data_scaled[exist_cont_features] = scaler.transform(train_data_merged[exist_cont_features])
            val_data_scaled[exist_cont_features] = scaler.transform(val_data_merged[exist_cont_features])
            test_data_scaled[exist_cont_features] = scaler.transform(test_data_merged[exist_cont_features])
            
            # 6. 训练和评估模型
            disease_results = []
            for name, features in model_dict.items():
                valid_features = [f for f in features if f in train_data_scaled.columns]
                print(f"\n训练模型: {name}, 使用特征数: {len(valid_features)}")
                
                model, res = train_eval_cox_with_penalty_search(
                    name, valid_features, train_data_scaled, val_data_scaled, test_data_scaled,
                    disease_name, category_name, output_dir
                )
                
                if res:
                    res['Disease'] = disease_name
                    res['Category'] = category_name
                    disease_results.append(res)
            
            if disease_results:
                all_results.extend(disease_results)
                successful_diseases += 1
                print(f"✅ 疾病 '{disease_name}' 处理完成，获得 {len(disease_results)} 个模型结果")
            else:
                print(f"⚠️  疾病 '{disease_name}' 没有生成任何模型结果")
            
        except Exception as e:
            print(f"处理疾病 {disease_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 为该疾病类型保存单独的结果文件
    if all_results:
        # 创建DataFrame
        category_df = pd.DataFrame(all_results)
        
        # 重新排列列的顺序
        columns_order = ['Category', 'Disease', 'Model', 'Penalizer', 'Train C-index', 'Val C-index', 'Test C-index']
        
        # 只保留实际存在的列
        existing_cols = [c for c in columns_order if c in category_df.columns]
        category_df = category_df[existing_cols]
        
        # 排序
        category_df = category_df.sort_values(['Model', 'Test C-index'], ascending=[True, False])
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个疾病类型生成单独的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(output_dir, f"{category_name}_cindex_summary_{timestamp}.csv")
        
        # 保存到CSV文件
        category_df.to_csv(csv_filename, index=False)
        
        print(f"\n✅ {category_name} 处理完成!")
        print(f"   成功处理的疾病数: {successful_diseases}/{len(h5ad_files)}")
        print(f"   总结果数: {len(all_results)}")
        print(f"   C-index汇总文件: {csv_filename}")
        
        # 同时在该疾病类型的文件夹中也保存一份汇总文件
        category_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)
        category_summary_file = os.path.join(category_dir, f"{category_name}_cindex_summary.csv")
        category_df.to_csv(category_summary_file, index=False)
        print(f"   疾病类型文件夹内汇总文件: {category_summary_file}")
    else:
        print(f"\n⚠️  {category_name} 没有生成任何结果")
    
    return all_results

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("开始处理所有疾病类型数据...")
    print(f"父目录: {PARENT_PATH}")
    print(f"疾病类型列表: {DISEASE_CATEGORIES}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"分析开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_categories = len(DISEASE_CATEGORIES)
    all_results = []
    successful_categories = []
    failed_categories = []
    
    # 创建进度日志文件
    log_file = os.path.join(OUTPUT_DIR, "analysis_progress.log")
    with open(log_file, 'w') as f:
        f.write(f"疾病类型分析进度日志\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"疾病类型总数: {total_categories}\n")
        f.write(f"{'='*50}\n")
    
    for i, category in enumerate(DISEASE_CATEGORIES, 1):
        print(f"\n处理进度: {i}/{total_categories} - {category}")
        
        # 构建疾病类型的完整路径
        category_path = os.path.join(PARENT_PATH, category)
        
        if not os.path.exists(category_path):
            error_msg = f"目录不存在: {category_path}"
            print(f"⚠️  {error_msg}，跳过该类型。")
            failed_categories.append(category)
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: {error_msg}\n")
            continue
        
        # 处理该疾病类型下的所有疾病
        try:
            category_results = process_disease_category(category, category_path, PANEL_PATH, model_dict, OUTPUT_DIR)
            
            if category_results:
                all_results.extend(category_results)
                successful_categories.append(category)
                
                # 记录到日志
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: 成功，获得 {len(category_results)} 个结果\n")
            else:
                failed_categories.append(category)
                
                # 记录到日志
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: 无结果\n")
                    
        except Exception as e:
            error_msg = f"处理 {category} 时发生错误: {str(e)}"
            print(f"❌ {error_msg}")
            failed_categories.append(category)
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {category}: 错误 - {str(e)}\n")
    
    # 记录结束时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    # 保存汇总结果（所有疾病类型的结果）
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # 重新排列列的顺序
        columns_order = ['Category', 'Disease', 'Model', 'Penalizer', 'Train C-index', 'Val C-index', 'Test C-index']
        
        # 只保留实际存在的列
        existing_cols = [c for c in columns_order if c in results_df.columns]
        results_df = results_df[existing_cols]
        
        # 排序
        results_df = results_df.sort_values(['Category', 'Disease', 'Model'])
        
        # 保存汇总文件
        summary_csv = os.path.join(OUTPUT_DIR, "ALL_DISEASES_CINDEX_SUMMARY.csv")
        results_df.to_csv(summary_csv, index=False)
        
        print(f"\n✅ 所有疾病汇总文件已保存:")
        print(f"   CSV汇总文件: {summary_csv}")
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("所有疾病类型处理完成！")
    print(f"{'='*80}")
    print(f"分析开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"分析结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed_time}")
    print(f"\n处理统计:")
    print(f"  成功处理的疾病类型: {len(successful_categories)}/{total_categories}")
    if successful_categories:
        print(f"  成功的类型: {', '.join(successful_categories)}")
    
    print(f"  失败的疾病类型: {len(failed_categories)}/{total_categories}")
    if failed_categories:
        print(f"  失败的类型: {', '.join(failed_categories)}")
    
    print(f"  总结果数: {len(all_results)}")
    print(f"  日志文件: {log_file}")
    print(f"  输出目录: {OUTPUT_DIR}")
    
    # 保存最终总结到文件
    summary_file = os.path.join(OUTPUT_DIR, "ANALYSIS_SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("疾病类型分析总结\n")
        f.write("="*50 + "\n")
        f.write(f"分析开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed_time}\n")
        f.write(f"成功处理的疾病类型: {len(successful_categories)}/{total_categories}\n")
        if successful_categories:
            f.write(f"成功的类型: {', '.join(successful_categories)}\n")
        f.write(f"失败的疾病类型: {len(failed_categories)}/{total_categories}\n")
        if failed_categories:
            f.write(f"失败的类型: {', '.join(failed_categories)}\n")
        f.write(f"总结果数: {len(all_results)}\n")
        f.write(f"日志文件: {log_file}\n")
        f.write(f"输出目录: {OUTPUT_DIR}\n")
        if all_results:
            f.write(f"汇总文件: {summary_csv}\n")
    
    print(f"\n总结已保存到: {summary_file}")
    
    # 显示汇总结果的前几行
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n汇总结果预览 (前20行):")
        print(results_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()



#nohup python -u /home/xuln/olink_disease_predict/code/panel_cox_result_0124.py > /home/xuln/olink_disease_predict/code/panel_cox_result_0124.log 2>&1 &

