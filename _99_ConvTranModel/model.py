import torch
import torch.nn as nn
import sys
import os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _99_ConvTranModel.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from _99_ConvTranModel.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_factory(config):
    return ConvTran(config, num_classes=config.num_labels)

class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        channel_size, seq_len = config.Data_shape
        emb_size = config.emb_size
        num_heads = config.num_heads
        dim_ff = config.dim_ff
        self.Fix_pos_encode = config.Fix_pos_encode
        self.Rel_pos_encode = config.Rel_pos_encode

        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size*4, kernel_size=(1, 8), padding='same'),
            nn.BatchNorm2d(emb_size*4),
            nn.GELU(),
            nn.Conv2d(emb_size*4, emb_size, kernel_size=(channel_size, 1), padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config.dropout, max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config.dropout, max_len=seq_len)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config.dropout, max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config.dropout)
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config.dropout)
        else:
            self.attention_layer = Attention(emb_size, num_heads, config.dropout)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config.dropout)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.float()  # Converti in float32 se necessario
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x).squeeze(2).permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = self.gap(out.permute(0, 2, 1)).flatten(1)
        out = self.out(out)
        return out
