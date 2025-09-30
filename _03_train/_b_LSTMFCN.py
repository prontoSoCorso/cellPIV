import sys
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
import multiprocessing as mp

mp.set_start_method("forkserver", force=True)

# Configurazione dei percorsi
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from config import Config_03_train_with_optimization as conf

device = conf.device

# Positional Encoding
def get_positional_encoding(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.orig_d = d_model
        self.n_heads = n_heads
        pad_to = math.ceil(d_model / n_heads) * n_heads
        self.pad_extra = pad_to - d_model
        self.pre_proj = nn.Linear(d_model, pad_to) if pad_to != d_model else nn.Identity()
        self.d_model = pad_to
        self.d_k = pad_to // n_heads
        self.w_q = nn.Linear(pad_to, pad_to, bias=False)
        self.w_k = nn.Linear(pad_to, pad_to, bias=False)
        self.w_v = nn.Linear(pad_to, pad_to, bias=False)
        self.w_o = nn.Linear(pad_to, pad_to)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pre_proj(x)  # [B, L, pad_to]
        B, L, _ = x.size()
        Q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, V)  # [B, heads, L, d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.w_o(ctx)
        if self.pad_extra:
            out = out[..., : self.orig_d]
        return out

# Residual Convolutional Block
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dropout, se_ratio=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        se_ch = max(1, int(out_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, se_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(se_ch, out_ch, 1),
            nn.Sigmoid()
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.res_norm = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        if hasattr(self.res_norm, "weight"):
            res = self.res_norm(res)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out * self.se(out)
        out = self.act(out + res)
        return out

# Modality Specific Encoder 
class ModalitySpecificEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encoder(x)

# Main Model
class TimeSeriesClassifier(nn.Module):
    """
    Classificatore binario per serie 1D. Accetta x come:
    - [B, 1, L] oppure [B, L, 1]
    """
    def __init__(
        self,
        input_channels: int = 1,
        enc_hidden_dim: int = 64,
        dropout_enc: float = 0.1,
        num_features: int = 128,
        n_heads: int = 4,
        dropout_attn: float = 0.1,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout_lstm: float = 0.2,
        cnn_filter_sizes: tuple = (64, 128, 256),
        cnn_kernel_sizes: tuple = (3, 3, 3),
        dropout_cnn: float = 0.2,
        se_ratio: float = 0.25,
        dropout_classifier: float = 0.3,
        use_positional: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.use_positional = use_positional
        self.input_channels = input_channels

        self.encoder = ModalitySpecificEncoder(input_channels, enc_hidden_dim, dropout_enc)
        fusion_dim = enc_hidden_dim
        self.fusion_proj = nn.Linear(fusion_dim, num_features)
        self.fusion_norm = nn.LayerNorm(num_features)

        self.attention = MultiHeadAttention(num_features, n_heads=n_heads, dropout=dropout_attn)
        self.attn_dropout = nn.Dropout(dropout_attn)
        self.attn_norm = nn.LayerNorm(num_features)

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        assert len(cnn_filter_sizes) == 3 and len(cnn_kernel_sizes) == 3, "Need 3 filter sizes and 3 kernel sizes"
        fs = list(cnn_filter_sizes)
        ks = list(cnn_kernel_sizes)
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(num_features, fs[0], ks[0], dropout_cnn, se_ratio),
            ResidualConvBlock(fs[0], fs[1], ks[1], dropout_cnn, se_ratio),
            ResidualConvBlock(fs[1], fs[2], ks[2], dropout_cnn, se_ratio),
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        conv_feat_dim = fs[-1] * 2
        final_dim = lstm_out_dim + conv_feat_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(final_dim, max(16, final_dim // 2)),
            nn.LayerNorm(max(16, final_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout_classifier * 0.5),
            nn.Linear(max(16, final_dim // 2), num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(m, "weight"):
                nn.init.ones_(m.weight)
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # sanitize and reshape to [B, L, C]
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        if x.dim() == 3 and x.shape[1] == 1:   # [B, 1, L]
            x = x.transpose(1, 2)              # -> [B, L, 1]
        # encode per-timestep channels
        enc = self.encoder(x)                  # [B, L, enc_hidden_dim]
        fused = self.fusion_proj(enc)
        fused = self.fusion_norm(fused)

        if self.use_positional:
            L = fused.size(1)
            pos = get_positional_encoding(L, fused.size(-1), device=fused.device)  # [L, D]
            fused = fused + pos.unsqueeze(0)

        attn_out = self.attention(fused)
        attn_out = self.attn_dropout(attn_out)
        fused = self.attn_norm(fused + attn_out)

        lstm_out, (h_n, _) = self.lstm(fused)
        if self.lstm.bidirectional:
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]

        x_conv = fused.permute(0, 2, 1)  # [B, D, L]
        for block in self.conv_blocks:
            x_conv = block(x_conv)

        conv_avg = self.global_avg_pool(x_conv).squeeze(-1)
        conv_max = self.global_max_pool(x_conv).squeeze(-1)
        conv_feat = torch.cat([conv_avg, conv_max], dim=1)
        final_feat = torch.cat([h_final, conv_feat], dim=1)
        logits = self.classifier(final_feat)   # [B, 2]
        return logits


if __name__ == "__main__":
    pass