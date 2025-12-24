import torch
import torch.nn as nn

def ch_stats(x, eps=1e-6):
    """计算通道均值和标准差，供 StyleAlignLayer 和 Loss 使用"""
    mu = x.mean(dim=1)
    var = x.var(dim=1, unbiased=False)
    std = torch.sqrt(var + eps)
    return mu, std

def gram(x):
    """计算 Gram 矩阵，供 Style Loss 使用"""
    B, N, D = x.shape
    G = torch.matmul(x.transpose(1, 2), x) / (N + 1e-6)
    return G

class GoldenStyleBank(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.num_layers = num_layers
        self.mu_g     = nn.Parameter(torch.zeros(num_layers, d_model))
        self.logsig_g = nn.Parameter(torch.zeros(num_layers, d_model))
        self._bootstrapped = [False]*num_layers

    def maybe_bootstrap(self, layer_idx, x_src, x_tgt):
        if not self._bootstrapped[layer_idx]:
            with torch.no_grad():
                mu_s, std_s = ch_stats(x_src)
                mu_t, std_t = ch_stats(x_tgt)
                mu = torch.cat([mu_s, mu_t], dim=0).mean(dim=0)
                std = torch.cat([std_s, std_t], dim=0).mean(dim=0)
                self.mu_g[layer_idx].data.copy_(mu)
                self.logsig_g[layer_idx].data.copy_(torch.log(std + 1e-6))
            self._bootstrapped[layer_idx] = True

    def stats(self, layer_idx):
        mu = self.mu_g[layer_idx]
        sig = torch.exp(self.logsig_g[layer_idx])
        return mu, sig

class StyleAlignLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, mu_g, sig_g, eps=1e-6):
        mu_x, std_x = ch_stats(x, eps=eps)
        x_norm = (x - mu_x.unsqueeze(1)) / (std_x.unsqueeze(1) + eps)
        x_hat  = x_norm * sig_g.view(1,1,-1) + mu_g.view(1,1,-1)
        alpha = torch.sigmoid(self.gate)
        y = (1 - alpha) * x + alpha * x_hat
        return y, alpha