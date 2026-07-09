import torch
import torch.nn as nn
import sys
from models.VNN_cell import VNN_cell

class VNNWrapper(nn.Module):
    
    def __init__(self, num_genes, hidden_dim, biovnn_dict, config):
        super(VNNWrapper, self).__init__()
        self.vnn_model = VNN_cell(
            omic_dim=1,
            input_dim=num_genes,
            output_dim=hidden_dim,
            biovnn_dict=biovnn_dict,
            run_mode="ref",
            act_func='Mish',
            use_sigmoid_output=False,
            dropout_p=config['dropout'],
            only_combine_child_gene_group=True,
            neuron_min=config['neuron_min'],
            neuron_ratio=config['neuron_ratio'],
            use_classification=False,
            child_map_fully=None,
            use_average_neuron_n=True,
            for_lr_finder=False
        )
        self.relu = nn.ReLU()

    def forward(self, gene_expression, device):
        
        vnn_input = list(torch.split(gene_expression, 1, dim=1))
        vnn_input = [tensor.to(device) for tensor in vnn_input]
        vnn_output = self.vnn_model(vnn_input)
        
        if torch.isnan(vnn_output).any():
            vnn_output = torch.nan_to_num(vnn_output, nan=0.0)
            
        return self.relu(vnn_output)