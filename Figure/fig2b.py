#%%

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import textwrap

# ==========================================
# 0. 全局设置与函数定义
# ==========================================
plt.rcParams.update({
    'font.size': 16,           
    'axes.labelsize': 20,      
    'xtick.labelsize': 16,     
    'ytick.labelsize': 16,     
    'legend.fontsize': 14,     
})

def get_significance_label(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return '' 

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

# 用于提取数据的模型列表
models_to_compare = ['Clinic+ProPathNet', 'ProPathNet', 'Clinic', 'Base']

# ==========================================
# 2. 准备画布 (极坐标图)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10), dpi=100, subplot_kw={'polar': True})

# ==========================================
# 3. 绘制雷达图
# ==========================================
if 'disease_category' in data_renamed.columns:
    counts = data_renamed.groupby('disease_category').size()
    valid_cats = counts[counts > 3].index
    
    df_radar = data_renamed[data_renamed['disease_category'].isin(valid_cats)]
    df_radar_mean = df_radar.groupby('disease_category')[models_to_compare].mean()
    
    radar_labels = []
    for cat in df_radar_mean.index:
        subset = df_radar[df_radar['disease_category'] == cat]
        
        sig_symbol = ""
        if len(subset) > 1:
            t_stat, p_val = stats.ttest_rel(subset['ProPathNet'], subset['Clinic'])
            mean_diff = subset['ProPathNet'].mean() - subset['Clinic'].mean()
            
            if mean_diff > 0 and p_val < 0.05:
                sig_symbol = get_significance_label(p_val)
        
        wrapped_name = textwrap.fill(cat.replace('_', ' '), 15)
        
        if sig_symbol:
            radar_labels.append(f"{wrapped_name}\n{sig_symbol}")
        else:
            radar_labels.append(wrapped_name)

    N = len(radar_labels)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] 
    
    ax.set_theta_offset(np.pi / 2) 
    ax.set_theta_direction(-1)     
    
    safe_angle = 0
    ax.set_rlabel_position(safe_angle)
    
    min_val = max(0.4, np.floor(df_radar_mean.min().min() * 10) / 10 - 0.05)
    max_val = min(1.0, np.ceil(df_radar_mean.max().max() * 10) / 10)
    ax.set_ylim(min_val, max_val)

    y_ticks = np.arange(0.5, 1.0, 0.1) 
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], color="grey", fontsize=12)
    
    ax.grid(color='#B3B3B3', linestyle='--', linewidth=1)
    ax.spines['polar'].set_visible(False) 

    ax.tick_params(axis='x', pad=30) 
    
    # 仅绘制指定的三个模型
    radar_models = ['ProPathNet', 'Clinic', 'Base']
    
    for idx, model in enumerate(radar_models):
        values = df_radar_mean[model].tolist()
        values += values[:1] 
        
        line_width = 2.5 if 'ProPathNet' in model else 1.5
        z_order = 10 - idx  
        
        ax.plot(angles, values, linewidth=line_width, linestyle='solid', 
                 label=model, color=MODEL_COLORS[model], zorder=z_order)
        
        # 将阴影填充限制在 ProPathNet 上
        if model == 'ProPathNet':
            ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.08, zorder=z_order-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=16, fontweight='bold')
    
    # 调整雷达图的图例位置
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), 
               fontsize=14, frameon=False, title='Models', title_fontsize=16)
else:
    ax.text(0.5, 0.5, "Column 'disease_category' not found", ha='center', va='center')

# ==========================================
# 4. 全局调整与显示
# ==========================================
plt.tight_layout()
plt.show()
# %%
