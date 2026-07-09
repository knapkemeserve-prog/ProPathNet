#%%

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局设置与颜色定义
# ==========================================
plt.rcParams.update({
    'font.size': 16,           
    'axes.labelsize': 20,      
    'xtick.labelsize': 16,     
    'ytick.labelsize': 16,     
    'legend.fontsize': 14,     
})

MODEL_COLORS = {
    'Base': '#4DBBD5',              # 浅蓝
    'Clinic': '#00A087',            # 绿色
    'ProPathNet': '#E64B35',        # 红色
    'Clinic+ProPathNet': '#3C5488'  # 深蓝
}

# ==========================================
# 1. 数据读取与预处理
# ==========================================
file_path = "/bigdat2/user/xuln/olink_disease_predict/67traits_cox_analysis/all_diseases_bootstrap_summary_simplified.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"找不到文件 {file_path}，请检查路径。")
    raise

data_renamed = data.rename(columns={
    'cindex_base_mean': 'Base',
    'cindex_traits_mean': 'Clinic',
    'cindex_mymodel_mean': 'ProPathNet',
    'cindex_traits_riskscore_mean': 'Clinic+ProPathNet'
})

# ==========================================
# 2. 准备画布 (单图)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# ==========================================
# 3. 绘制柱状图 - 降序排列
# ==========================================
models_to_compare = ['Clinic+ProPathNet', 'ProPathNet', 'Clinic', 'Base']
display_labels = ['Clinic+\nProPathNet', 'ProPathNet', 'Clinic', 'Base']
df_models = data_renamed[models_to_compare]

comparison_list = ['Base', 'Clinic', 'Clinic+ProPathNet']
results = {}
for other_model in comparison_list:
    valid_data = df_models[['ProPathNet', other_model]].dropna()
    if len(valid_data) > 1:
        t_stat, p_val = stats.ttest_rel(valid_data['ProPathNet'], valid_data[other_model])
        results[other_model] = {'p_val': p_val, 'n': len(valid_data)}
    else:
        results[other_model] = {'p_val': np.nan, 'n': 0}

means, cis = [], []
for model in models_to_compare:
    model_data = df_models[model].dropna()
    n = len(model_data)
    if n > 0:
        mean_val = np.mean(model_data)
        sem_val = stats.sem(model_data)
        t_score = stats.t.ppf(0.975, n - 1)
        ci = sem_val * t_score
    else:
        mean_val, ci = np.nan, np.nan
    means.append(mean_val)
    cis.append(ci)

bar_width = 0.6
x_pos = np.arange(len(models_to_compare))
bar_colors = [MODEL_COLORS[m] for m in models_to_compare]

bars = ax.bar(x_pos, means, yerr=cis, capsize=8, color=bar_colors, 
               alpha=0.9, width=bar_width, linewidth=0,
               error_kw={'elinewidth': 1.5, 'ecolor': '#444444'})

for i, bar in enumerate(bars):
    height = bar.get_height()
    if not np.isnan(height):
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', 
                 fontsize=18, fontweight='500')

max_bar_height = max([m + c for m, c in zip(means, cis) if not np.isnan(m + c)])
base_height = max_bar_height + 0.05
height_increment = 0.07

idx_my = models_to_compare.index('ProPathNet')
comparison_list_sorted = sorted(comparison_list, key=lambda x: abs(idx_my - models_to_compare.index(x)))

for i, other_model in enumerate(comparison_list_sorted):
    idx_other = models_to_compare.index(other_model)
    p = results[other_model]['p_val']
    n_samples = results[other_model]['n']

    if pd.isna(p) or n_samples <= 1 or p >= 0.05:
        continue

    sig_symbol = '***' if p < 0.001 else '**' if p < 0.01 else '*'
    y_h = base_height + i * height_increment
    
    ax.plot([idx_my, idx_my, idx_other, idx_other], 
             [y_h, y_h+0.005, y_h+0.005, y_h], 
             lw=1.5, c='#333333')
    ax.text((idx_my+idx_other)*0.5, y_h+0.005, sig_symbol, 
             ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(display_labels, fontsize=18, fontweight='bold')
ax.set_ylabel('Mean C-index (95% CI)', fontsize=20, labelpad=10, fontweight='bold')
ax.set_ylim(0, 1.15) 

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# ==========================================
# 4. 全局调整与显示
# ==========================================
plt.tight_layout()
plt.show()
# %%
