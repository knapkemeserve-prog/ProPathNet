# %%
#跑所有疾病在四个模型上的bootstrap

import os
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

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

# 定义bootstrap函数
def bootstrap_cindex_comparison(df, n_bootstrap=1000):
    """
    对合并数据进行bootstrap分析
    """
    # 准备数据
    T = df['survival_time'].values
    E = df['event_status'].values
    scores = {
        'base': df['risk_score_base'].values,
        'traits': df['risk_score_67traits'].values,
        'mymodel': df['risk_score_mymodel'].values,
        'traits_riskscore': df['risk_score_67traits_riskscore'].values
    }
    
    n_samples = len(df)
    
    # 存储bootstrap结果
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # 抽取样本
        T_boot = T[indices]
        E_boot = E[indices]
        
        # 计算每个模型的C-index
        cindex_vals = {}
        for name, score_array in scores.items():
            score_boot = score_array[indices]
            try:
                cindex = concordance_index(T_boot, -score_boot, E_boot)
                cindex_vals[name] = cindex
            except:
                cindex_vals[name] = np.nan
        
        # 计算差值
        diff_mymodel_base = cindex_vals.get('mymodel', np.nan) - cindex_vals.get('base', np.nan)
        diff_mymodel_traits = cindex_vals.get('mymodel', np.nan) - cindex_vals.get('traits', np.nan)
        diff_traits_riskscore_mymodel = cindex_vals.get('traits_riskscore', np.nan) - cindex_vals.get('mymodel', np.nan)
        diff_traits_riskscore_traits = cindex_vals.get('traits_riskscore', np.nan) - cindex_vals.get('traits', np.nan)
        
        # 存储结果
        bootstrap_results.append({
            'bootstrap_iteration': i+1,
            'cindex_base': cindex_vals.get('base', np.nan),
            'cindex_traits': cindex_vals.get('traits', np.nan),
            'cindex_mymodel': cindex_vals.get('mymodel', np.nan),
            'cindex_traits_riskscore': cindex_vals.get('traits_riskscore', np.nan),
            'diff_mymodel_base': diff_mymodel_base,
            'diff_mymodel_traits': diff_mymodel_traits,
            'diff_traits_riskscore_mymodel': diff_traits_riskscore_mymodel,
            'diff_traits_riskscore_traits': diff_traits_riskscore_traits
        })
    
    return pd.DataFrame(bootstrap_results)

# 计算汇总统计量
def calculate_summary_statistics(bootstrap_df):
    """计算bootstrap结果的汇总统计量"""
    summary = {}
    
    # 各模型C-index的统计量
    models = ['cindex_base', 'cindex_traits', 'cindex_mymodel', 'cindex_traits_riskscore']
    for model in models:
        values = bootstrap_df[model].dropna()
        if len(values) > 0:
            summary[f'{model}_mean'] = values.mean()
            summary[f'{model}_median'] = values.median()
            summary[f'{model}_std'] = values.std()
            summary[f'{model}_ci_lower'] = np.percentile(values, 2.5)
            summary[f'{model}_ci_upper'] = np.percentile(values, 97.5)
        else:
            summary[f'{model}_mean'] = np.nan
            summary[f'{model}_median'] = np.nan
            summary[f'{model}_std'] = np.nan
            summary[f'{model}_ci_lower'] = np.nan
            summary[f'{model}_ci_upper'] = np.nan
    
    # 差值统计量
    diffs = ['diff_mymodel_base', 'diff_mymodel_traits', 
             'diff_traits_riskscore_mymodel', 'diff_traits_riskscore_traits']
    for diff in diffs:
        values = bootstrap_df[diff].dropna()
        if len(values) > 0:
            summary[f'{diff}_mean'] = values.mean()
            summary[f'{diff}_median'] = values.median()
            summary[f'{diff}_std'] = values.std()
            summary[f'{diff}_ci_lower'] = np.percentile(values, 2.5)
            summary[f'{diff}_ci_upper'] = np.percentile(values, 97.5)
            
            # 判断显著性
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            if ci_lower > 0 or ci_upper < 0:
                summary[f'{diff}_significant'] = 'Yes'
            else:
                summary[f'{diff}_significant'] = 'No'
        else:
            summary[f'{diff}_mean'] = np.nan
            summary[f'{diff}_median'] = np.nan
            summary[f'{diff}_std'] = np.nan
            summary[f'{diff}_ci_lower'] = np.nan
            summary[f'{diff}_ci_upper'] = np.nan
            summary[f'{diff}_significant'] = 'No'
    
    return pd.DataFrame([summary])

# 定义基础路径
base_path = "/bigdat2/user/xuln/olink_disease_predict"
cohort_path = f"{base_path}/67traits_cox_analysis"
test_pred_path = f"{base_path}/test_predictions"
riskscore_path = f"{base_path}/67traits_cox_analysis/67traits_riskscore_test_predictions"

# 存储所有疾病的结果
all_results_summary = []

# 遍历所有疾病类别
for disease_category in DISEASE_CATEGORIES:
    print("\n" + "="*80)
    print(f"处理疾病类别: {disease_category}")
    print("="*80)
    
    # 获取该疾病类别下的所有疾病文件夹
    category_path = os.path.join(cohort_path, disease_category)
    
    # 检查路径是否存在
    if not os.path.exists(category_path):
        print(f"路径不存在: {category_path}")
        continue
    
    # 获取疾病文件夹列表
    try:
        disease_folders = [d for d in os.listdir(category_path) 
                          if os.path.isdir(os.path.join(category_path, d))]
    except Exception as e:
        print(f"无法读取疾病文件夹: {e}")
        continue
    
    print(f"找到 {len(disease_folders)} 个疾病文件夹")
    
    # 遍历每个疾病文件夹
    for disease_folder in disease_folders:
        disease_name = disease_folder
        print(f"\n处理疾病: {disease_name}")
        
        try:
            # 构建文件路径
            # Base模型文件
            base_file = os.path.join(category_path, disease_name, "Base_predictions.csv")
            
            # 67Traits模型文件
            traits_file = os.path.join(category_path, disease_name, "Physical_examination_predictions.csv")
            
            # MyModel文件 - 注意：疾病名称可能需要将空格替换为下划线
            disease_name_underscore = disease_name.replace(' ', '_')
            mymodel_file = os.path.join(test_pred_path, disease_category, disease_name_underscore, "test_predictions.csv")
            disease_name_underscore_lower=disease_name_underscore.lower()
            # 67Traits Risk Score文件
            riskscore_file = os.path.join(riskscore_path, disease_category, f"{disease_name_underscore_lower}_test_riskscores.csv")
            
            # 检查所有文件是否存在
            if not os.path.exists(base_file):
                print(f"  Base文件不存在: {base_file}")
                continue
            if not os.path.exists(traits_file):
                print(f"  67Traits文件不存在: {traits_file}")
                continue
            if not os.path.exists(mymodel_file):
                print(f"  MyModel文件不存在: {mymodel_file}")
                continue
            if not os.path.exists(riskscore_file):
                print(f"  67Traits Risk Score文件不存在: {riskscore_file}")
                continue
            
            print(f"  所有文件存在，开始处理...")
            
            # 读取数据
            test_prediction_base = pd.read_csv(base_file)
            test_prediction_base = test_prediction_base.rename(columns={
                'time': 'survival_time',
                'event': 'event_status'
            })
            
            test_prediction_67traits = pd.read_csv(traits_file)
            test_prediction_67traits = test_prediction_67traits.rename(columns={
                'time': 'survival_time',
                'event': 'event_status'
            })
            
            test_prediction_mymodel = pd.read_csv(mymodel_file)
            # 检查列名并重命名
            if 'patient_id' in test_prediction_mymodel.columns:
                test_prediction_mymodel = test_prediction_mymodel.rename(columns={'patient_id': 'participant.eid'})
            
            test_prediction_67traits_riskscore = pd.read_csv(riskscore_file)
            
            # 重命名风险评分列
            test_prediction_base = test_prediction_base.rename(columns={'risk_score': 'risk_score_base'})
            test_prediction_67traits = test_prediction_67traits.rename(columns={'risk_score': 'risk_score_67traits'})
            test_prediction_mymodel = test_prediction_mymodel.rename(columns={'risk_score': 'risk_score_mymodel'})
            
            # 注意：67traits_riskscore文件的列名可能是'risk_score_combined'
            if 'risk_score_combined' in test_prediction_67traits_riskscore.columns:
                test_prediction_67traits_riskscore = test_prediction_67traits_riskscore.rename(
                    columns={'risk_score_combined': 'risk_score_67traits_riskscore'})
            elif 'risk_score' in test_prediction_67traits_riskscore.columns:
                test_prediction_67traits_riskscore = test_prediction_67traits_riskscore.rename(
                    columns={'risk_score': 'risk_score_67traits_riskscore'})
            
            # 合并数据
            # 选择base_df
            base_df = test_prediction_base[['participant.eid', 'survival_time', 'event_status']].copy()
            
            # 使用merge依次合并各个数据集的risk_score列
            combined_df = base_df.merge(
                test_prediction_base[['participant.eid', 'risk_score_base']], 
                on='participant.eid'
            )
            
            combined_df = combined_df.merge(
                test_prediction_67traits[['participant.eid', 'risk_score_67traits']], 
                on='participant.eid'
            )
            
            combined_df = combined_df.merge(
                test_prediction_mymodel[['participant.eid', 'risk_score_mymodel']], 
                on='participant.eid'
            )
            
            combined_df = combined_df.merge(
                test_prediction_67traits_riskscore[['participant.eid', 'risk_score_67traits_riskscore']], 
                on='participant.eid'
            )
            
            # 确保没有缺失值
            combined_df_clean = combined_df.dropna(subset=[
                'survival_time', 'event_status', 
                'risk_score_base', 'risk_score_67traits', 
                'risk_score_mymodel', 'risk_score_67traits_riskscore'
            ])
            
            print(f"  合并后数据量: {len(combined_df)} 行，清洗后: {len(combined_df_clean)} 行")
            
            if len(combined_df_clean) > 0:
                # 进行bootstrap分析
                print(f"  开始bootstrap分析 (1000次)...")
                bootstrap_results_df = bootstrap_cindex_comparison(combined_df_clean, n_bootstrap=1000)
                
                # 计算汇总统计量
                summary_stats_df = calculate_summary_statistics(bootstrap_results_df)
                
                # 添加疾病信息
                summary_stats_df['disease_category'] = disease_category
                summary_stats_df['disease_name'] = disease_name
                summary_stats_df['sample_size'] = len(combined_df_clean)
                
                # 保存到所有结果中
                all_results_summary.append(summary_stats_df)
                
                # 创建保存结果的目录
                output_dir = os.path.join(category_path, disease_name, "bootstrap_results")
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存原始bootstrap结果
                bootstrap_results_df.to_csv(
                    os.path.join(output_dir, "bootstrap_raw_results.csv"), 
                    index=False
                )
                
                # 保存汇总统计量
                summary_stats_df.to_csv(
                    os.path.join(output_dir, "bootstrap_summary_statistics.csv"), 
                    index=False
                )
                
                print(f"  ✓ 分析完成，结果已保存到: {output_dir}")
                
                # 打印简要结果
                print(f"    样本量: {len(combined_df_clean)}")
                if 'cindex_mymodel_mean' in summary_stats_df.columns:
                    cindex_val = summary_stats_df['cindex_mymodel_mean'].values[0]
                    if not pd.isna(cindex_val):
                        print(f"    MyModel C-index: {cindex_val:.4f}")
            else:
                print(f"  ✗ 清洗后数据为空，跳过此疾病")
                
        except Exception as e:
            print(f"  处理疾病 {disease_name} 时出错: {str(e)}")
            continue

# 汇总所有疾病的结果
if all_results_summary:
    # 合并所有疾病的结果
    all_summary_df = pd.concat(all_results_summary, ignore_index=True)
    
    # 重新排列列顺序
    cols = ['disease_category', 'disease_name', 'sample_size'] + \
           [col for col in all_summary_df.columns if col not in 
            ['disease_category', 'disease_name', 'sample_size']]
    all_summary_df = all_summary_df[cols]
    
    # 保存汇总结果
    summary_output_path = os.path.join(base_path, "all_diseases_bootstrap_summary.csv")
    all_summary_df.to_csv(summary_output_path, index=False)
    
    print("\n" + "="*80)
    print("所有疾病分析完成!")
    print("="*80)
    print(f"总计分析了 {len(all_summary_df)} 个疾病")
    print(f"汇总结果已保存到: {summary_output_path}")
    
    # 打印一些统计信息
    print(f"\nC-index统计:")
    if 'cindex_mymodel_mean' in all_summary_df.columns:
        mymodel_cindex = all_summary_df['cindex_mymodel_mean'].dropna()
        if len(mymodel_cindex) > 0:
            print(f"  MyModel C-index范围: [{mymodel_cindex.min():.4f}, {mymodel_cindex.max():.4f}]")
            print(f"  MyModel平均C-index: {mymodel_cindex.mean():.4f}")
    
    print(f"\n模型比较统计:")
    if 'diff_mymodel_base_mean' in all_summary_df.columns:
        mymodel_base_diff = all_summary_df['diff_mymodel_base_mean'].dropna()
        if len(mymodel_base_diff) > 0:
            print(f"  MyModel优于Base的疾病数: {(mymodel_base_diff > 0).sum()}")
            print(f"  MyModel劣于Base的疾病数: {(mymodel_base_diff < 0).sum()}")
    
    # 显示前几个疾病的结果
    print(f"\n前5个疾病的结果:")
    print(all_summary_df[['disease_category', 'disease_name', 'sample_size', 
                          'cindex_mymodel_mean', 'diff_mymodel_base_mean']].head())
else:
    print("没有成功分析任何疾病")

print("\n分析完成!")