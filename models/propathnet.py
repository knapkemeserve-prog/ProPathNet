import torch
import torch.nn as nn
from .gat import GATNet
from .vnn import VNNWrapper
from .fnn import ClinicalFNN
from .fusion import AttentionFusion

class ProPathNet(nn.Module):
    
    def __init__(self, num_nodes, num_genes, hidden_dim, num_layers, biovnn_dict, config, clinical_dim):
        super(ProPathNet, self).__init__()
        self.clinical_dim = clinical_dim
        
        
        self.gat_net = GATNet(num_nodes, hidden_dim, num_layers, config.get('gat_heads', 8), config.get('dropout', 0.6))
        self.vnn_net = VNNWrapper(num_genes, hidden_dim, biovnn_dict, config)
        self.fnn_net = ClinicalFNN(clinical_dim, hidden_dim)
        self.fusion_net = AttentionFusion(hidden_dim, config['fusion_type'], config['fusion_heads'], config['dropout'])

    def forward(self, x, edge_index, batch, gene_expression, clinical, return_attention=False):
        device = x.device
        
        
        clinical = clinical.view(-1, self.clinical_dim).to(device)
        clinical_feat = self.fnn_net(clinical)

        vnn_feat = self.vnn_net(gene_expression, device)
      
        if return_attention:
            gat_feat, _, layer_attentions = self.gat_net(x, edge_index, batch, return_attention=True)
        else:
            gat_feat, _ = self.gat_net(x, edge_index, batch)
        
        features_to_fuse = [gat_feat, vnn_feat, clinical_feat]
        risk_score = self.fusion_net(features_to_fuse)
        
        if return_attention:
            return risk_score, layer_attentions
        return risk_score