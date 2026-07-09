import torch.nn as nn
from torch_geometric.nn import BatchNorm

class ClinicalFNN(nn.Module):

    def __init__(self, clinical_dim, hidden_dim):
        super(ClinicalFNN, self).__init__()
        self.fc1 = nn.Linear(clinical_dim, hidden_dim * 2)
        self.bn1 = BatchNorm(hidden_dim * 2)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, clinical_features):
        x = self.relu(self.bn1(self.fc1(clinical_features)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.fc3(x)))
        return x