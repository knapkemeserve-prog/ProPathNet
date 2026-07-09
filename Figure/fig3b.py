#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend 
import matplotlib.lines as mlines     
import pandas as pd
import numpy as np

print("\n" + "=" * 80)
print("开始极速绘制 Cancers 疾病临床特征全景图...")

# ==========================================
# 1. 直接读取总表并切片
# ==========================================
master_file = "/home/xuln/olink_disease_predict/ProPathNet-github/result/all_diseases_top10_lifestyles.csv"
try:
    df_all = pd.read_csv(master_file)
except FileNotFoundError:
    print(f"未找到总表文件 {master_file}，请先运行数据整合脚本！")
    raise

# 极速筛选（若要画其他疾病，只需把 'Cancers' 换成比如 'Psychiatric' 即可）
category_to_plot = 'Cancers'
df_cancers = df_all[df_all['Category'] == category_to_plot].copy()

if not df_cancers.empty:
    # ==========================================
    # 2. 数据透视与频次统计 (提取前 15 个)
    # ==========================================
    pivot_df = pd.crosstab(df_cancers['Clinical_Feature'], df_cancers['Disease'])
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    
    max_frequency = pivot_df['Total'].max()
    
    top_n = min(15, len(pivot_df))
    top_features_df = pivot_df.head(top_n)
    
    pivot_df = pivot_df.drop('Total', axis=1)
    print("共有特征数量:", len(pivot_df))

    # ==========================================
    # 3. 深度清洗：图例 (Columns) 与 特征名 (Index)
    # ==========================================
    pivot_df.columns = pivot_df.columns.str.replace('_', ' ')
    pivot_df.columns = pivot_df.columns.str.replace('filtered', '', case=False)
    pivot_df.columns = pivot_df.columns.str.strip()

    # 自动化批量缩写特征名称
    new_index = []
    for idx in pivot_df.index:
        s = str(idx)
        s = s.replace('Time spent ', '').replace('Days/week of ', '')
        s = s.replace('days/week', 'd/wk').replace('/week', '/wk')
        s = s.replace('Never/rarely', 'Rarely').replace('Never or almost never', 'Rarely')
        s = s.replace('More than ', '>').replace('Less than ', '<')
        s = s.replace('About the same', 'Same')
        s = s.replace('moderate physical activity', 'MPA').replace('vigorous physical activity', 'VPA')
        s = s.replace('Body Mass Index (BMI)', 'BMI')
        if ': ' in s:
            parts = s.split(': ', 1)
            s = f"{parts[0]} ({parts[1]})"
        new_index.append(s)
    pivot_df.index = new_index

    # 硬编码特殊长难句缩写
    abbr_dict_hardcoded = {
        'outdoors in summer': 'Outdoors (Sum)',
        'outdoors in winter': 'Outdoors (Win)',
        'walk 10+ mins': 'Walk 10+m',
        'Alcohol vs 10yrs ago (<10yrs: Less now)': 'Alc <10y (Less)',
        'Mother (High blood pressure)': 'Maternal HTN',
        'Father (Heart disease)': 'Paternal CHD',
        'Age first had sexual intercourse': 'Age first sex',
        'Facial ageing (Younger than you are)': 'Facial age (Young)'
    }
    pivot_df.rename(index=abbr_dict_hardcoded, inplace=True)

    # ==========================================
    # 4. 绘图准备与主图绘制
    # ==========================================
    cmap = plt.get_cmap('Set3')
    colors = [cmap(i % 12) for i in range(len(pivot_df.columns))]
    total_features_count = len(pivot_df)

    fig1, ax1 = plt.subplots(figsize=(20, 10)) 
    
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.8, legend=False)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.05, 1, 0.2), 
               mode="expand", borderaxespad=0, ncol=2, fontsize=16, frameon=False)

    ax1.set_ylabel('Number of diseases for\nwhich it was selected as a predictor', fontsize=16)
    ax1.set_xlabel('Lifestyle Features', fontsize=16, labelpad=10) 
    ax1.set_xticks(range(len(pivot_df)))
    ax1.set_xticklabels(pivot_df.index, rotation=90, fontsize=10)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ==========================================
    # 5. 动态局部放大框 + 基于独立 Legend 对象的双列对齐
    # ==========================================
    if top_n > 0:
        rect_height = max_frequency + 0.3
        rect = Rectangle((-0.5, 0), top_n, rect_height, 
                         linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax1.add_patch(rect)

        top_list = top_features_df.index.tolist()
        labels_list = [f"{i+1}. {feat}" for i, feat in enumerate(top_list)]
        
        empty_handle = mlines.Line2D([], [], linestyle='')
        dummy_handles = [empty_handle] * len(top_list)
        
        header = f"Top Lifestyle Predictors in {category_to_plot}" 
        
        text_x = total_features_count * 0.4  
        text_y = max_frequency * 1.1          
        
        box_legend = Legend(ax1, dummy_handles, labels_list, 
                            loc='upper left', 
                            bbox_to_anchor=(text_x, text_y),
                            bbox_transform=ax1.transData, 
                            ncol=2,                       
                            fontsize=14,                  
                            title=header,
                            title_fontsize=16,
                            handlelength=0,               
                            handletextpad=0,              
                            columnspacing=2.5)            
        
        ax1.add_artist(box_legend)
        
        frame = box_legend.get_frame()
        frame.set_boxstyle("round,pad=1")
        frame.set_edgecolor("black")
        frame.set_facecolor("white")
        frame.set_linestyle("--")
        frame.set_linewidth(1.2)
        frame.set_alpha(0.95)

        half = (len(top_list) + 1) // 2
        line_box_bottom = text_y - (max_frequency * 0.05 * (half + 2.5))

        ax1.plot([top_n - 0.5, text_x], [rect_height, text_y], 
                 color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax1.plot([top_n - 0.5, text_x], [0, line_box_bottom], 
                 color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # ==========================================
    # 6. 布局调整与输出
    # ==========================================
    plt.subplots_adjust(top=0.78, bottom=0.3) 
    plt.show() 

else:
    print(f"未提取到 {category_to_plot} 类别的临床数据，请检查！")
# %%
