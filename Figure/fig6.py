#%%
###############################fig6a###################################################

#蛋白重要性的图
#比如要画Primary Malignancy Prostate
#画的是选出来的蛋白
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义汇总文件路径
summary_csv_path = "/home/xuln/olink_disease_predict/ProPathNet-github/result/step_wise_select_features_results.csv"

# 2. 定义目标疾病相关变量
TARGET_DISEASE = "Primary Malignancy Prostate"          
TARGET_DISEASE_PATH = "Primary_Malignancy_Prostate"     # 用于输出子目录的文件夹名
TARGET_DISEASE_TYPE = "Cancers"  
BASE_IG_RESULTS_PATH = "/bigdat2/user/xuln/olink_disease_predict/ig_results_GAT/"

def extract_target_proteins():
    # 读取汇总文件
    try:
        summary_data = pd.read_csv(summary_csv_path)
    except FileNotFoundError:
        print(f"Error: 找不到汇总文件 {summary_csv_path}")
        return

    # 定位到目标疾病所在的行
    disease_row = summary_data[summary_data['Disease'] == TARGET_DISEASE]
    
    if disease_row.empty:
        print(f"Error: 在汇总文件中未找到疾病 '{TARGET_DISEASE}'")
        return
    
    # 获取对应的最佳蛋白数量 (Trait_Protein_Best_Count)
    protein_count = int(disease_row['Trait_Protein_Best_Count'].values[0])
    print(f"已在汇总表中找到 '{TARGET_DISEASE}'，使用的最佳蛋白数量为: {protein_count}")

    # 拼接对应疾病的蛋白重要性文件路径
    importance_file_path = os.path.join(
        BASE_IG_RESULTS_PATH, 
        TARGET_DISEASE_TYPE, 
        TARGET_DISEASE_PATH, 
        "joint_protein_importance.csv"
    )
    
    # 读取蛋白重要性文件并提取 Top N 蛋白
    if os.path.exists(importance_file_path):
        importance_data = pd.read_csv(importance_file_path)
        
        # 【修改提示】请将这里的 'Protein_Name' 和 'Importance_Score' 
        # 替换为你 joint_protein_importance.csv 文件中真实的列名
        protein_col = 'protein'      
        importance_col = 'importance' 
        
        # 确保数据按重要性从大到小排序
        importance_data = importance_data.sort_values(by=importance_col, ascending=False)
        
        # 截取前 N 个蛋白
        top_proteins = importance_data.head(protein_count)
        
        print(f"\n成功提取出前 {protein_count} 个靶蛋白，数据预览：")
        print(top_proteins.head())
        
        # --- 开始绘图 ---
        # 为了让最重要的在图表最上方，我们需要将数据反转（升序排序），因为 barh 是从下往上画的
        plot_data = top_proteins.sort_values(by=importance_col, ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.grid(False)
        
        # 生成渐变蓝色 (数值越大，颜色越深)
        # np.linspace(0.2, 0.9, ...) 控制颜色的深浅范围，可以自行微调
        colors = plt.cm.Blues(np.linspace(0.2, 0.9, len(plot_data)))
        
        # 绘制水平柱状图
        bars = plt.barh(plot_data[protein_col], plot_data[importance_col], color=colors, edgecolor='none')
        
        # 设置标签和标题
        plt.xlabel('Feature Importance', fontsize=16)
        plt.ylabel('Protein', fontsize=16)
        plt.title(TARGET_DISEASE, fontsize=20, fontweight='bold')
        # ====== 新增这两行来调整刻度字号 ======
        plt.xticks(fontsize=14)  # 调整 X 轴刻度数值的字号
        plt.yticks(fontsize=14)  # 调整 Y 轴刻度文本（蛋白质名）的字号

        # 调整布局以防止标签被截断
        plt.tight_layout()
        
        # 保存图片到指定目录
        plot_filename = f"top_{protein_count}_proteins_importance_{TARGET_DISEASE_PATH}.png"
        plot_file_path = os.path.join(
            BASE_IG_RESULTS_PATH, 
            TARGET_DISEASE_TYPE, 
            TARGET_DISEASE_PATH, 
            plot_filename
        )
        #plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
        #print(f"\n图表绘制完成！已保存至: {plot_file_path}")
        
        # 如果你想在跑代码时直接弹出窗口看图，可以取消下面这行的注释
        plt.show()
        # --- 绘图结束 ---

        # 保存提取出的蛋白CSV
        output_filename = f"top_{protein_count}_proteins_{TARGET_DISEASE_PATH}.csv"
        output_file_path = os.path.join(
            BASE_IG_RESULTS_PATH, 
            TARGET_DISEASE_TYPE, 
            TARGET_DISEASE_PATH, 
            output_filename)
        #top_proteins.to_csv(output_file_path, index=False)
        #print(f"CSV 数据已保存至: {output_file_path}")
        
    else:
        print(f"Error: 找不到蛋白重要性文件 {importance_file_path}")

if __name__ == "__main__":
    extract_target_proteins()


#%%
###############################fig6b###################################################

#来自https://genemania.org/网站

#%%
###############################fig6c###################################################

#绘图思路见fig5