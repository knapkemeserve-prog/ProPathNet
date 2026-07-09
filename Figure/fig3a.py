# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend 
import matplotlib.lines as mlines     

print("\n" + "=" * 80)
print("开始绘制 Cancers 疾病 Olink 蛋白特征全景图...")

# ==========================================
# 1. 直接从总表中读取数据并筛选
# ==========================================
master_file = "/home/xuln/olink_disease_predict/ProPathNet-github/result/all_diseases_top1_proteins.csv"
try:
    df_all = pd.read_csv(master_file)
except FileNotFoundError:
    print(f"未找到总表文件 {master_file}，请先运行数据整合脚本！")
    raise

# 极速筛选：只保留 Category 为 Cancers 的行
df_cancers = df_all[df_all['Category'] == 'Cancers'].copy()

if not df_cancers.empty:
    # ==========================================
    # 2. 数据透视与排序 (核心逻辑保持不变)
    # ==========================================
    pivot_df = pd.crosstab(df_cancers['Protein'], df_cancers['Disease'])
    
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)

    total_proteins_on_x = len(pivot_df)
    print(f"📊 统计报告: 横轴共展示了 {total_proteins_on_x} 个蛋白。")
    print(f"📊 报告: 其中前 {min(15, total_proteins_on_x)} 个蛋白被包含在虚线框(top_n)中。")
    
    max_frequency = pivot_df['Total'].max()
    
    top_n = min(15, len(pivot_df))
    top_proteins_df = pivot_df.head(top_n)
    
    pivot_df = pivot_df.drop('Total', axis=1)
    pivot_df.columns = pivot_df.columns.str.replace('_', ' ').str.replace('filtered', '', case=False).str.strip()

    cmap = plt.get_cmap('Set3')
    colors = [cmap(i % 12) for i in range(len(pivot_df.columns))]
    total_proteins_count = len(pivot_df)

    # ==========================================
    # 3. 绘制主图 & 正常保留主图例
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    
    pivot_df.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.8, legend=False)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.05, 1, 0.2), 
               mode="expand", borderaxespad=0, ncol=2, fontsize=16, frameon=False)

    ax1.set_ylabel('Number of diseases for\nwhich it was selected as a predictor', fontsize=16)
    ax1.set_xlabel('Proteins', fontsize=16, labelpad=10) 
    ax1.set_xticks(range(len(pivot_df)))
    ax1.set_xticklabels(pivot_df.index, rotation=90, fontsize=10)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ==========================================
    # 4. 局部放大框 + 基于独立 Legend 对象的双列对齐
    # ==========================================
    if top_n > 0:
        rect_height = max_frequency + 0.3
        rect = Rectangle((-0.5, 0), top_n, rect_height, 
                         linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax1.add_patch(rect)

        top_list = top_proteins_df.index.tolist()
        labels_list = [f"{i+1}. {prot}" for i, prot in enumerate(top_list)]
        
        empty_handle = mlines.Line2D([], [], linestyle='')
        dummy_handles = [empty_handle] * len(top_list)
        
        header = "Top Protein Predictors in Cancers\n" 
        
        text_x = total_proteins_count * 0.45  
        text_y = max_frequency * 1.1          
        
        box_legend = Legend(ax1, dummy_handles, labels_list, 
                            loc='upper left', 
                            bbox_to_anchor=(text_x, text_y),
                            bbox_transform=ax1.transData, 
                            ncol=2,                       
                            fontsize=16, 
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
    # 5. 布局微调与输出
    # ==========================================
    plt.subplots_adjust(top=0.78, bottom=0.2)
    plt.show() 

else:
    print("未提取到 Cancers 类别的数据！请检查整合表内容。")
# %%
