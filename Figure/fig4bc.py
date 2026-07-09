# %%
#model_labels = ['Clinic', 'Clinic+Lifestyles',  'Clinic+Lifestyles+Proteins']

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 全局设置与文件路径
# ==========================================
# 设置全局字体大小
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,        # 全局图例字体适当调大
    'legend.title_fontsize': 20
})

disease_name ='Primary Malignancy Prostate'

# 基线 (0-year) 数据路径

baseline_path = "/home/xuln/olink_disease_predict/ProPathNet-github/result/4models_risk_scores_Primary_Malignancy_Prostate_0.csv"
#baseline_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0527/Haematological_or_immunological/Iron deficiency anaemia/fig5_select_protein_xgb_prediction/time_gt_minus10/4models_risk_scores_Iron deficiency anaemia.csv'

#baseline_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0527/Musculoskeletal/Osteoporosis/fig5_select_protein_xgb_prediction/time_gt_minus10/4models_risk_scores_Osteoporosis.csv'
#baseline_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0527/Cardiovascular/Stable angina/fig5_select_protein_xgb_prediction/time_gt_minus10/4models_risk_scores_Stable angina.csv'
#baseline_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/67trait_with_select_protein_before_baseline_0527/Cardiovascular/Hypertension/fig5_select_protein_xgb_prediction/time_gt_minus10/4models_risk_scores_Hypertension.csv'


# 随访 (3, 5, 10-year) 数据路径
followup_path = '/home/xuln/olink_disease_predict/ProPathNet-github/result/4models_risk_scores_Primary_Malignancy_Prostate_filtered_3510.csv'
#followup_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/prognostic_results/Haematological_or_immunological/Iron_deficiency_anaemia/4models_risk_scores_Iron_deficiency_anaemia.csv'

#followup_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/prognostic_results/Musculoskeletal/Osteoporosis/4models_risk_scores_Osteoporosis.csv'
#followup_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/prognostic_results/Cardiovascular/Stable_angina/4models_risk_scores_Stable_angina.csv'
#followup_path = '/bigdat2/user/xuln/olink_disease_predict/save_models/prognostic_results/Cardiovascular/Hypertension/4models_risk_scores_Hypertension.csv'


# 读取数据
df_base = pd.read_csv(baseline_path)
df_follow = pd.read_csv(followup_path)

# 创建 3 年、5 年、10 年的 event 状态
df_follow['event_3yr'] = df_follow.apply(lambda row: row['event'] if row['time'] <= 3 else 0, axis=1)
df_follow['event_5yr'] = df_follow.apply(lambda row: row['event'] if row['time'] <= 5 else 0, axis=1)
df_follow['event_10yr'] = df_follow.apply(lambda row: row['event'] if row['time'] <= 10 else 0, axis=1)

# ==========================================
# 2. 定义模型信息与 4 个子图的配置 
# ==========================================
models = [
    'trait_risk_score', 
    'trait+lifestyle_risk_score', 
    'trait+lifestyle+protein_risk_score'
]

model_colors = ['blue', 'orange', 'red']

# 🌟 核心修改：定义两套 Labels，针对 0 年和随访年分开
labels_0yr = ['Clinic', 'Clinic+Lifestyle-opt-pred', 'Clinic+ProPathNet-opt-pred']
labels_followup = ['Clinic', 'Clinic+Lifestyle-opt', 'Clinic+ProPathNet-opt']

panels_config = [
    {'title': '0-Year',           'data': df_base,   'event_col': 'event',      'has_time': False},
    {'title': '3-Year',           'data': df_follow, 'event_col': 'event_3yr',  'has_time': True},
    {'title': '5-Year',           'data': df_follow, 'event_col': 'event_5yr',  'has_time': True},
    {'title': '10-Year',          'data': df_follow, 'event_col': 'event_10yr', 'has_time': True}
]

metrics_results = [] # 用于保存所有评价指标

# ==============================================================================
# 3. 绘制 1x4 ROC 曲线图
# ==============================================================================
fig_roc, axes_roc = plt.subplots(1, 4, figsize=(32, 8))
fig_roc.suptitle(f'{disease_name} - ROC Curves', fontsize=36, fontweight='bold', y=1.0)

for idx, config in enumerate(panels_config):
    ax = axes_roc[idx]
    data = config['data']
    events = data[config['event_col']]
    
    # 🌟 动态判断当前使用哪套标签
    current_labels = labels_0yr if idx == 0 else labels_followup
    
    if len(np.unique(events)) < 2:
        ax.text(0.5, 0.5, 'Insufficient events', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(config['title'])
        continue
        
    for model, color, label in zip(models, model_colors, current_labels):
        preds = data[model].fillna(0)
        
        auc = roc_auc_score(events, preds)
        fpr, tpr, _ = roc_curve(events, preds)
        
        if config['has_time']:
            cindex = concordance_index(data['time'], -preds, events)
            legend_label = f'{label}\n(AUC={auc:.3f}, Cindex={cindex:.3f})'
        else:
            cindex = np.nan
            legend_label = f'{label}\n(AUC={auc:.3f})'
            
        if idx == 0: 
            auprc = average_precision_score(events, preds)
            precisions, recalls, _ = precision_recall_curve(events, preds)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            metrics_results.append({
                'Time_Point': config['title'],
                'Model': label,
                'AUC': auc,
                'C-index': cindex,
                'AUPRC': auprc,
                'Best_F1': f1_scores[best_idx],
                'Precision': precisions[best_idx],
                'Recall': recalls[best_idx]
            })
            
        ax.plot(fpr, tpr, color=color, lw=3, label=legend_label)

    if config['has_time']:
        # 🌟 随访记录指标同样使用动态标签
        for model, label in zip(models, current_labels):
            preds = data[model].fillna(0)
            auprc = average_precision_score(events, preds)
            precisions, recalls, _ = precision_recall_curve(events, preds)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            metrics_results.append({
                'Time_Point': config['title'],
                'Model': label,
                'AUC': roc_auc_score(events, preds),
                'C-index': concordance_index(data['time'], -preds, events),
                'AUPRC': auprc,
                'Best_F1': f1_scores[best_idx],
                'Precision': precisions[best_idx],
                'Recall': recalls[best_idx]
            })

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    if idx == 0: ax.set_ylabel('True Positive Rate')
    ax.set_title(config['title'], fontsize=24, fontweight='bold')
    ax.legend(loc='lower right', fontsize=16)

plt.tight_layout()
fig_roc.subplots_adjust(top=0.85)

output_dir = os.path.dirname(followup_path)
plot_roc_path = os.path.join(output_dir, f"3models_ROC_1x4_{disease_name.replace(' ', '_')}.png")
plt.savefig(plot_roc_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"1x4 ROC曲线图已保存至: {plot_roc_path}")


# ==============================================================================
# 4. 绘制 1x4 PRC 曲线图
# ==============================================================================
fig_prc, axes_prc = plt.subplots(1, 4, figsize=(32, 8))
fig_prc.suptitle(f'{disease_name} - PR Curves', fontsize=36, fontweight='bold', y=1.0)

for idx, config in enumerate(panels_config):
    ax = axes_prc[idx]
    data = config['data']
    events = data[config['event_col']]
    
    # 🌟 动态判断当前使用哪套标签
    current_labels = labels_0yr if idx == 0 else labels_followup
    
    if len(np.unique(events)) < 2:
        ax.text(0.5, 0.5, 'Insufficient events', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(config['title'])
        continue
        
    for model, color, label in zip(models, model_colors, current_labels):
        preds = data[model].fillna(0)
        
        auprc = average_precision_score(events, preds)
        precisions, recalls, _ = precision_recall_curve(events, preds)
        
        ax.plot(recalls, precisions, color=color, lw=3, label=f'{label}\n(AUPRC={auprc:.3f})')
        
    baseline = events.sum() / len(events)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)')
    if idx == 0: ax.set_ylabel('Precision (PPV)')
    ax.set_title(config['title'], fontsize=24, fontweight='bold')
    ax.legend(loc='upper right', fontsize=16)

plt.tight_layout()
fig_prc.subplots_adjust(top=0.85)

plot_prc_path = os.path.join(output_dir, f"3models_PRC_1x4_{disease_name.replace(' ', '_')}.png")
plt.savefig(plot_prc_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"1x4 PRC曲线图已保存至: {plot_prc_path}")


# ==============================================================================
# 5. 导出统一的详细指标
# ==============================================================================
metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.drop_duplicates(subset=['Time_Point', 'Model']).reset_index(drop=True)

metrics_save_path = os.path.join(output_dir, f"3models_ALL_metrics_{disease_name.replace(' ', '_')}.csv")
metrics_df.to_csv(metrics_save_path, index=False)

#print("\n" + "="*85)
#print("各模型在全部时间点（0, 3, 5, 10年）的综合评估指标：")
#print("="*85)
#print(metrics_df.to_string(index=False))
#print("="*85)
#print(f"综合指标表已保存至: {metrics_save_path}")
# %%
