import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool

class GATNet(nn.Module):

    def __init__(self, num_nodes, hidden_dim, num_layers, gat_heads=8, gat_dropout=0.6):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        self.raw_feature_fc = nn.Linear(num_nodes, hidden_dim)
        self.raw_bn = BatchNorm(hidden_dim)
        self.relu = nn.ReLU()
        
        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
       
        self.gat_layers.append(GATConv(
            in_channels=1, out_channels=hidden_dim // gat_heads, 
            heads=gat_heads, dropout=gat_dropout, add_self_loops=True, concat=True
        ))
        self.bn_layers.append(BatchNorm(hidden_dim))
        
        
        for _ in range(1, num_layers):
            self.gat_layers.append(GATConv(
                in_channels=hidden_dim, out_channels=hidden_dim // gat_heads,
                heads=gat_heads, dropout=gat_dropout, add_self_loops=True, concat=True
            ))
            self.bn_layers.append(BatchNorm(hidden_dim))

    def forward(self, x, edge_index, batch, return_attention=False):
        
        batch_size = torch.max(batch).item() + 1 if batch is not None else 1
        x_flat = x.view(batch_size, self.num_nodes)
        raw_x = self.relu(self.raw_bn(self.raw_feature_fc(x_flat)))
        
        current_x = x
        layer_attentions = []
        
        for i in range(self.num_layers):
            if return_attention:
                current_x, (edge_index_i, att_weights) = self.gat_layers[i](
                    current_x, edge_index, return_attention_weights=True
                )
                layer_attentions.append(att_weights)
            else:
                current_x = self.gat_layers[i](current_x, edge_index)
            
            current_x = self.bn_layers[i](current_x)
            current_x = self.relu(current_x)
            
        gat_feat = global_mean_pool(current_x, batch)
        
        if return_attention:
            return gat_feat, raw_x, layer_attentions
        return gat_feat, raw_x