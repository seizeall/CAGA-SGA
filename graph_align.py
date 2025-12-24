import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class SemanticAligner(nn.Module):
    def __init__(self, in_features, num_classes, gcn_out_features=128):
        super().__init__()
        self.edge_network = nn.Linear(in_features, in_features)
        self.gcn = DenseGCNConv(in_features, gcn_out_features)
        self.node_classifier = nn.Linear(gcn_out_features, num_classes)

    def forward(self, features):
        x_attn = torch.tanh(self.edge_network(features))
        affinity_hat = torch.matmul(x_attn, x_attn.transpose(1, 0))
        adj_normalized_for_gcn = F.softmax(affinity_hat, dim=1)
        gcn_input = features.unsqueeze(0)
        adj_for_gcn = adj_normalized_for_gcn.unsqueeze(0)
        updated_features = F.relu(self.gcn(gcn_input, adj_for_gcn)).squeeze(0)
        node_logits = self.node_classifier(updated_features)
        return node_logits, affinity_hat