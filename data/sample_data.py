#%%
import os
import anndata as ad
from sklearn.model_selection import train_test_split

def create_sample_toy_h5ad(h5ad_path, output_dir='/home/xuln/olink_disease_predict/ProPathNet-github/data', toy_size=50, random_state=42):
    """
    按照 obs['event'] 等比例分层抽取 50 个样本，并直接保存为 .h5ad 格式
    """
    print(f"正在读取原始 H5AD 数据: {h5ad_path}")
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"未找到 H5AD 文件，请确认路径是否正确。")
        
    adata = ad.read_h5ad(h5ad_path)
    print(f"原始数据加载成功，总样本数: {adata.n_obs}, 特征数: {adata.n_vars}")
    
    # 检查 event 列
    if 'event' not in adata.obs.columns:
        raise KeyError("在 adata.obs 中未找到 'event' 列。")
        
    # 执行等比例分层抽样
    print(f"正在进行分层抽样，抽取样本量: {toy_size}")
    _, toy_indices = train_test_split(
        adata.obs.index,
        test_size=toy_size,
        stratify=adata.obs['event'],
        random_state=random_state
    )
    
    # 切片提取子集 (此时 adata_toy 依然是标准的 AnnData 对象)
    adata_toy = adata[toy_indices, :].copy()
    
    print("\n=== Toy 数据集中的 Event 分布 ===")
    counts = adata_toy.obs['event'].value_counts()
    for val, cnt in counts.items():
        print(f"Event = {val}: {cnt} 个样本")
    print("===============================\n")
    
    # 创建输出目录并保存为 H5AD
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'sample_data.h5ad')  # <--- 后缀改为 h5ad
    
    # 核心修改：直接调用 anndata 的保存方法
    adata_toy.write_h5ad(output_file)
    
    print(f"🎉 成功！Toy 蛋白数据已完美保存至: {output_file}")
    print(f"数据维度: {adata_toy.n_obs} 行 (样本) x {adata_toy.n_vars} 列 (特征)")

if __name__ == "__main__":
    ORIGINAL_H5AD = '/bigdat2/user/xuln/olink_disease_predict/data/Cardiovascular/Hypertension.h5ad'
    create_sample_toy_h5ad(h5ad_path=ORIGINAL_H5AD, toy_size=50)



# %%
##/bigdat2/user/xuln/olink_disease_predict/data/clinic_drop20_10_9.csv

import os
import pandas as pd
import anndata as ad

def filter_clinic_by_h5ad(clinic_csv_path, toy_h5ad_path, output_csv_path=None):
    """
    根据刚才抽样的 sample_data.h5ad 中的样本 ID，过滤原始临床 CSV 数据
    """
    print("正在读取数据，请稍候...")
    
    if not os.path.exists(clinic_csv_path):
        raise FileNotFoundError(f"未找到临床数据文件: {clinic_csv_path}")
    if not os.path.exists(toy_h5ad_path):
        raise FileNotFoundError(f"未找到样本H5AD文件: {toy_h5ad_path}")
        
    df_clinic = pd.read_csv(clinic_csv_path)
    
    # 核心修改：读取 H5AD 提取 ID
    adata_toy = ad.read_h5ad(toy_h5ad_path)
    
    print(f"📊 原始临床数据维度: {df_clinic.shape}")
    
    # 从 h5ad.obs.index 提取独立的样本 ID
    target_ids = adata_toy.obs.index.astype(str).unique()
    print(f"🔍 从 H5AD 提取到目标独立样本 ID 数量: {len(target_ids)}")
    
    # 确定 clinic 数据中的样本 ID 列名
    clinic_id_col = 'participant.eid'
    if clinic_id_col not in df_clinic.columns:
        print(f"⚠️ 警告：在临床数据中未找到精确匹配的 '{clinic_id_col}' 列。")
        possible_cols = [col for col in df_clinic.columns if 'id' in col.lower() or 'eid' in col.lower()]
        if possible_cols:
            clinic_id_col = possible_cols[0]
            print(f"💡 自动选择可能的主键列: '{clinic_id_col}' 进行匹配。")
        else:
            raise KeyError("无法在临床数据中定位样本 ID 列。")
            
    # 执行过滤提取
    df_clinic[clinic_id_col] = df_clinic[clinic_id_col].astype(str)
    df_filtered = df_clinic[df_clinic[clinic_id_col].isin(target_ids)].copy()
    
    print(f"✨ 过滤提取完成！")
    print(f"📈 提取后的临床数据维度: {df_filtered.shape}")
    
    matched_count = df_filtered[clinic_id_col].nunique()
    if matched_count < len(target_ids):
        print(f"⚠️ 提示：目标样本中有 {len(target_ids) - matched_count} 个 ID 在临床数据中未找到匹配项。")
        
    # 保存结果为 CSV
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_filtered.to_csv(output_csv_path, index=False)
        print(f"💾 临床表型数据已成功保存为 CSV 至: {output_csv_path}")
        
    return df_filtered

if __name__ == "__main__":
    CLINIC_FILE = '/bigdat2/user/xuln/olink_disease_predict/data/clinic_drop20_10_9.csv'
    # 指向刚才生成的新 h5ad 文件
    TOY_H5AD_FILE = '/home/xuln/olink_disease_predict/ProPathNet-github/data/sample_data.h5ad'
    
    OUTPUT_FILE = '/home/xuln/olink_disease_predict/ProPathNet-github/data/sample_lifestyle_data.csv'
    
    filtered_df = filter_clinic_by_h5ad(
        clinic_csv_path=CLINIC_FILE, 
        toy_h5ad_path=TOY_H5AD_FILE,
        output_csv_path=OUTPUT_FILE
    )
# %%
