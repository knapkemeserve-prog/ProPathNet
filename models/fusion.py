import torch.nn as nn
import sys
from models.fusion_models import fusion_transformer

class FeaturePassThrough(nn.Module):

    def __init__(self, in_features, out_features):
        super(FeaturePassThrough, self).__init__()
        self.out = nn.Linear(in_features, out_features)
        nn.init.eye_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        
    def forward(self, x):
        return self.out(x)

class AttentionFusion(nn.Module):
    
    def __init__(self, hidden_dim, fusion_type, fusion_heads, dropout_prob):
        super(AttentionFusion, self).__init__()
        
        sub_models = [
            FeaturePassThrough(hidden_dim, hidden_dim), # GAT
            FeaturePassThrough(hidden_dim, hidden_dim), # VNN
            FeaturePassThrough(hidden_dim, hidden_dim)  # Clinical
        ]
        resize_latent_space = [hidden_dim, hidden_dim, hidden_dim]
        
        self.fusion = fusion_transformer(
            sub_models=sub_models,
            num_fusion_layers=1,
            fusion_type=fusion_type,
            heads=fusion_heads,
            fusion_dim=hidden_dim,
            dropout_prob=dropout_prob,
            n_classes=1,
            freeze_submodels=False,
            resize_latent_space=resize_latent_space
        )

    def forward(self, features_list):
        return self.fusion(features_list)