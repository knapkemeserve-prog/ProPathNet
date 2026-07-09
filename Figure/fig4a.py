# %%

#按照疾病类型画的的柱状图
#分别是Trait, Trait+lifestyle, Trait+lifestyle+protein的结果
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap

# 1. 读取数据 (替换为你实际的文件路径)
file_path = "/home/xuln/olink_disease_predict/ProPathNet-github/result/step_wise_select_features_results.csv"
result_df = pd.read_csv(file_path)

# 2. 定义需要求均值的三个 C-index 列 (已移除 Trait_Protein_Best_Test_Cindex)
cindex_cols = [
    'Trait_Only_Test_Cindex', 
    'Trait_LS_Best_Test_Cindex', 
    'Combined_Test_Cindex'
]

# 3. 按 Disease_Type 分组，并对三个指标求均值
grouped_df = result_df.groupby('Disease_Type')[cindex_cols].mean().reset_index()

# 为了图表美观，你可以根据某个均值（比如Combined列）排个序，不需要的话可以注释掉下一行
grouped_df = grouped_df.sort_values('Combined_Test_Cindex', ascending=False)

# 4. 生成 X 轴的标签 (对 Disease_Type 进行换行处理)
labels = []
for _, row in grouped_df.iterrows():
    disease_type = str(row['Disease_Type']).replace('_', ' ')
    # 将过长的 Disease_Type 名称换行，宽度可根据实际显示效果调整
    wrapped_name = textwrap.fill(disease_type, width=25) 
    labels.append(wrapped_name)

# ==========================================
# 在这里定义你想要在图表中展示的新模型名称
# ==========================================
model1_name = 'Clinic'           
model2_name = 'Clinic + Lifestyle-opt'      
model3_name = 'Clinic + ProPathNet-opt' 

# 5. 开始绘图
n_types = len(grouped_df)
# 设置图形大小，根据分类的数量动态调整宽度
fig, ax = plt.subplots(figsize=(max(10, n_types * 1.5), 8))

x = np.arange(n_types)
width = 0.25  # 3根柱子，稍微把宽度调大一点点到0.25比较饱满

# 绘制三个柱状组 (保留了你原来分配给这三个模型的专属颜色)
bars1 = ax.bar(x - width, grouped_df['Trait_Only_Test_Cindex'], width, label=model1_name, color='#ffffb2')
bars2 = ax.bar(x, grouped_df['Trait_LS_Best_Test_Cindex'], width, label=model2_name, color='#a1dab4')
bars3 = ax.bar(x + width, grouped_df['Combined_Test_Cindex'], width, label=model3_name, color='#225ea8')

# 添加标签和标题
ax.set_xlabel('Disease Type', fontsize=24, labelpad=15)
ax.set_ylabel('Mean Test C-index', fontsize=24, labelpad=15)
#ax.set_title('Mean Test C-index across Models by Disease Type', fontsize=28, fontweight='bold', pad=20)

# 设置 X 轴刻度
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=22) 
ax.tick_params(axis='y', labelsize=20)

# ==========================================
# 将图例移回图内
# loc='best' 会让 matplotlib 自动寻找一个不遮挡柱子的最优位置
# ==========================================
ax.legend(fontsize=22, loc='best')

# 统一设定 Y 轴范围
ax.set_ylim(0.6, 0.85) 

# ==========================================
# 调整边距
# ==========================================
plt.tight_layout()
# 仅保留 bottom=0.2 防止 X 轴的长标签被截断
plt.subplots_adjust(bottom=0.2) 
plt.show()
# %%
