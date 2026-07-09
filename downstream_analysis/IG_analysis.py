import os
import sys
import json
import pickle
import random
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from captum.attr import IntegratedGradients

# --- 强制将项目根目录加入系统路径 ---
# 这一步极其关键，确保能找到 models 和 preprocess 文件夹
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入我们拆分好的各个模块 ---
from preprocess.data_loader import preprocess_data
from models.propathnet import ProPathNet

# ----------------------------
# 1. 联合特征重要性分析包装器
# ----------------------------
class JointFeatureWrapper(nn.Module):
    """
    为了适配 Captum IG 计算，将双模态输入包装成单一张量输入的 Wrapper
    """
    def __init__(self, full_model, edge_index, num_nodes, clinical_dim):
        super().__init__()
        self.model = full_model
        self.register_buffer('edge_index', edge_index)
        self.num_nodes = num_nodes
        self.clinical_dim = clinical_dim

    def forward(self, joint_features):
        # joint_features: [B, num_nodes + clinical_dim]
        # 前 num_nodes 个特征是基因表达，后 clinical_dim 个特征是临床特征
        B = joint_features.shape[0]
        
        # 分割特征
        gene_expr = joint_features[:, :self.num_nodes]  # [B, num_nodes]
        clinical = joint_features[:, self.num_nodes:]   # [B, clinical_dim]
        
        # 构建图数据
        batch = torch.arange(B, device=joint_features.device).repeat_interleave(self.num_nodes)
        x = gene_expr.reshape(-1, 1)  # [B * num_nodes, 1]
        
        # 调用 ProPathNet (注意这里的输入顺序需与 forward 保持一致)
        return self.model(x, self.edge_index, batch, gene_expr, clinical)

# ----------------------------
# 2. 核心分析功能函数
# ----------------------------
def sample_balanced_data(train_data, n_samples=1000):
    """从训练集中按照事件发生率采样样本。如果样本不足，直接返回全部样本。"""
    total_samples = len(train_data)
    
    # === 新增逻辑：如果总样本数不足，直接使用全部样本 ===
    if total_samples <= n_samples:
        print(f"⚠️ 提示: 训练集总样本数 ({total_samples}) 小于或等于请求的采样数 ({n_samples})。")
        print(f"将直接使用全部现有 {total_samples} 个样本进行分析！")
        balanced_samples = list(train_data) # 复制一份列表
        random.shuffle(balanced_samples)    # 打乱顺序
        return balanced_samples
        
    # 如果总样本数充足，则按发生率提取
    positive_samples = [data for data in train_data if data.e.item() == 1]
    negative_samples = [data for data in train_data if data.e.item() == 0]
    
    positive_count, negative_count = len(positive_samples), len(negative_samples)
    
    print(f"训练集总样本数: {total_samples}")
    print(f"训练集中正样本数: {positive_count} (占比: {positive_count/total_samples:.2%})")
    print(f"训练集中负样本数: {negative_count} (占比: {negative_count/total_samples:.2%})")
    
    n_positive = int(n_samples * (positive_count / total_samples))
    n_negative = n_samples - n_positive
    n_positive = max(1, min(n_positive, positive_count))
    n_negative = max(1, min(n_negative, negative_count))
    
    sampled_positive = random.sample(positive_samples, n_positive) if positive_count >= n_positive else positive_samples
    sampled_negative = random.sample(negative_samples, n_negative) if negative_count >= n_negative else negative_samples
    
    balanced_samples = sampled_positive + sampled_negative
    random.shuffle(balanced_samples)
    
    print(f"最终采样样本数: {len(balanced_samples)}")
    return balanced_samples

def compute_joint_importance_batch(model, samples, gene_names, clinical_columns, output_dir, steps=30, device=None):
    """批量计算联合特征重要性"""
    model.eval()
    
    edge_index = samples[0].edge_index.to(device)
    num_nodes = samples[0].x.shape[0]
    clinical_dim = samples[0].clinical.shape[1]
    
    # 计算均值作为 Baseline
    print("计算特征均值作为基线...")
    all_gene_expr = [s.x.view(1, -1) for s in samples]
    all_clinical = [s.clinical for s in samples]
    
    mean_gene_expr = torch.mean(torch.cat(all_gene_expr, dim=0), dim=0, keepdim=True)
    mean_clinical = torch.mean(torch.cat(all_clinical, dim=0), dim=0, keepdim=True)
    mean_baseline = torch.cat([mean_gene_expr, mean_clinical], dim=1).to(device)
    
    all_gene_importances = []
    all_clinical_importances = []
    
    print(f"开始计算 {len(samples)} 个样本的 IG 归因...")
    for i, sample in enumerate(tqdm(samples)):
        try:
            sample = sample.to(device)
            gene_expr_single = sample.x.view(1, -1).to(device)
            clinical_single = sample.clinical.to(device)
            joint_input = torch.cat([gene_expr_single, clinical_single], dim=1)
            
            joint_wrapper = JointFeatureWrapper(model, edge_index, num_nodes, clinical_dim).to(device)
            ig_joint = IntegratedGradients(joint_wrapper)
            joint_attributions = ig_joint.attribute(
                joint_input, baselines=mean_baseline, n_steps=steps
            )
            
            gene_imp = joint_attributions[:, :num_nodes].abs().cpu().detach().numpy().flatten()
            clinical_imp = joint_attributions[:, num_nodes:].abs().cpu().detach().numpy().flatten()
            
            all_gene_importances.append(gene_imp)
            all_clinical_importances.append(clinical_imp)
        except Exception as e:
            print(f"计算样本 {i} 时出错: {e}")
            all_gene_importances.append(np.zeros(len(gene_names)))
            all_clinical_importances.append(np.zeros(len(clinical_columns)))
    
    # 汇总并保存
    avg_gene_imp = np.mean(all_gene_importances, axis=0)
    avg_clinical_imp = np.mean(all_clinical_importances, axis=0)
    
    gene_df = pd.DataFrame({'protein': gene_names, 'importance': avg_gene_imp}).sort_values('importance', ascending=False)
    clinical_df = pd.DataFrame({'feature': clinical_columns, 'importance': avg_clinical_imp}).sort_values('importance', ascending=False)
    
    gene_df.to_csv(os.path.join(output_dir, 'joint_protein_importance.csv'), index=False)
    clinical_df.to_csv(os.path.join(output_dir, 'joint_clinical_importance.csv'), index=False)
    pd.DataFrame(all_gene_importances, columns=gene_names).to_csv(os.path.join(output_dir, 'protein_importance_all_samples.csv'), index=False)
    pd.DataFrame(all_clinical_importances, columns=clinical_columns).to_csv(os.path.join(output_dir, 'clinical_importance_all_samples.csv'), index=False)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    sns.barplot(data=gene_df.head(20), x='importance', y='protein')
    plt.title('Top 20 Important Proteins')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_protein_importance.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, max(6, len(clinical_columns) * 0.3)))
    sns.barplot(data=clinical_df, x='importance', y='feature')
    plt.title('Clinical Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_clinical_importance.png'), dpi=300)
    plt.close()

# ----------------------------
# 3. 主干流程
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Integrated Gradients 可解释性分析")
    parser.add_argument('--model_path', type=str, required=True, help='模型.pth文件路径')
    parser.add_argument('--config_path', type=str, default=None, help='config.json路径（获取最优超参数）')
    parser.add_argument('--output_dir', type=str, default='result/ig_analysis', help='结果保存目录')
    parser.add_argument('--n_samples', type=int, default=200, help='采样样本数量')
    parser.add_argument('--ig_steps', type=int, default=30, help='IG计算步数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA设备编号')
    
    # 数据文件参数
    parser.add_argument('--adata_path', type=str, required=True)
    parser.add_argument('--ppi_path', type=str, required=True)
    parser.add_argument('--clinical_path', type=str, required=True)
    parser.add_argument('--biovnn_path', type=str, required=True)
    
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 随机种子固定
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 自动提取 clinical_columns 名称 (配合原特征列逻辑)
    clinical_raw = pd.read_csv(args.clinical_path)
    continuous_vars = ['874','884','904','864','1090','1080','1070','1160','1269','1279',
                       '1289','1299','1309','1319','1438','1458','1488','1498','1528','1050',
                       '1060','2277','2139','2149','age','bmi']
    available_continuous = [v for v in continuous_vars if v in clinical_raw.columns]
    categorical_cols = [col for col in clinical_raw.columns if col not in available_continuous and col != 'participant.eid']
    clinical_columns = available_continuous + categorical_cols

    # 2. 读取配置和预处理数据 (调用我们拆分好的 preprocess_data)
    config = {}
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_data = json.load(f)
            if "long_training_results" in config_data and len(config_data["long_training_results"]) > 0:
                config = config_data["long_training_results"][0]["params"]

    print(">>> 第一步: 调用模块加载数据...")
    train_data, val_data, test_data, protein_to_idx, gene_names, clinical_dim = preprocess_data(
        adata_path=args.adata_path, ppi_path=args.ppi_path, clinical_path=args.clinical_path
    )

    with open(args.biovnn_path, "rb") as f:
        biovnn_dict = pickle.load(f)

    # 3. 初始化 ProPathNet 模块
    print(">>> 第二步: 初始化 ProPathNet 模型...")
    num_genes = len(gene_names)
    num_nodes = train_data[0].x.shape[0]
    
    # 兼容缺失的超参数默认值
    config = {
        'hidden_dim': 64, 'num_layers': 4, 'fusion_type': 'bottleneck',
        'fusion_heads': 4, 'fusion_dropout_prob': 0.5, 'vnn_dropout_p': 0.5,
        'neuron_min': 64, 'neuron_ratio': 0.2, 'batch_size': 256,
        'lr': 1e-4, 'weight_decay': 1e-5, 'l2_reg': 1e-4,
        'epochs': 100, 'patience': 10, 'gat_heads': 4, 'gat_dropout': 0.5
    }

    model = ProPathNet(
        num_nodes=num_nodes, num_genes=num_genes, hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'], biovnn_dict=biovnn_dict,
        config=config, clinical_dim=clinical_dim
    ).to(device)

    # 4. 加载权重
    print(f">>> 第三步: 加载最优权重 ({args.model_path})...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # strict=False 保证封装后的名字略有差异时也能兼容加载
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    # 5. 执行 IG 分析
    print(">>> 第四步: 启动 Integrated Gradients 计算...")
    sampled_data = sample_balanced_data(train_data, n_samples=args.n_samples)
    compute_joint_importance_batch(
        model=model, samples=sampled_data, gene_names=gene_names,
        clinical_columns=clinical_columns, output_dir=args.output_dir,
        steps=args.ig_steps, device=device
    )

if __name__ == "__main__":
    main()