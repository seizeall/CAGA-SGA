import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
import math

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        reversed_grad = grad_output.neg() * ctx.alpha
        return reversed_grad, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class AttentiveGraphLayer(nn.Module):
    def __init__(self, in_features, out_features, topk):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.topk = topk
        self.gconv = DenseGCNConv(in_features, out_features)
        self.attn_lin = nn.Linear(in_features, in_features)

    def forward(self, x):
        x_attn = torch.tanh(self.attn_lin(x))
        adj = torch.matmul(x_attn, x_attn.transpose(2, 1))
        adj = F.softmax(adj, dim=2)
        mask = torch.zeros_like(adj, device=x.device)
        _, top_indices = torch.topk(adj, self.topk, dim=2)
        mask.scatter_(2, top_indices, 1)
        adj = adj * mask
        x_gcn = F.relu(self.gconv(x, adj))
        return x_gcn

class SpatialStream(nn.Module):
    def __init__(self, in_dim, layer_dims, topk):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims)):
            self.layers.append(
                AttentiveGraphLayer(in_dim if i == 0 else layer_dims[i - 1], layer_dims[i], topk)
            )
    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        return layer_outputs

class CoAttnEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.co_attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.norm3     = nn.LayerNorm(d_model)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.dropout3  = nn.Dropout(dropout)

    def forward(self, temporal_q, spatial_kv):
        q_self, _ = self.self_attn(temporal_q, temporal_q, temporal_q)
        temporal_q = temporal_q + self.dropout1(q_self)
        temporal_q = self.norm1(temporal_q)

        q_co, _ = self.co_attn(query=temporal_q, key=spatial_kv, value=spatial_kv)
        temporal_q = temporal_q + self.dropout2(q_co)
        temporal_q = self.norm2(temporal_q)

        ff_out = self.linear2(self.dropout(F.relu(self.linear1(temporal_q))))
        temporal_q = temporal_q + self.dropout3(ff_out)
        temporal_q = self.norm3(temporal_q)
        return temporal_q