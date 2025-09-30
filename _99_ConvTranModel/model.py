import torch.nn as nn
import torch
import sys
import os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "cellPIV":
    parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from _99_ConvTranModel.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from _99_ConvTranModel.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec

def model_factory(config):
    return ConvTran(config, num_classes=config.num_labels)

class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        channel_size, seq_len = config.Data_shape
        emb_size = config.emb_size_convtran
        num_heads = config.num_heads_convtran
        dim_ff = config.dim_ff
        self.Fix_pos_encode = config.Fix_pos_encode
        self.Rel_pos_encode = config.Rel_pos_encode

        k = getattr(config, "kernel_len_emb", 9)
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size*4, kernel_size=(1, k), padding='same', bias=False),
            nn.BatchNorm2d(emb_size*4),
            nn.GELU(),
            nn.Conv2d(emb_size*4, emb_size, kernel_size=(channel_size, 1), padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU(),
            nn.Dropout(config.dropout_convtran)
            )

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config.dropout_convtran, max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config.dropout_convtran, max_len=seq_len)
        elif self.Fix_pos_encode == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config.dropout_convtran, max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config.dropout_convtran)
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config.dropout_convtran)
        else:
            self.attention_layer = Attention(emb_size, num_heads, config.dropout_convtran)

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        self.attn_dropout = nn.Dropout(config.dropout_convtran)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout_convtran),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config.dropout_convtran)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)  # sanitize
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x).squeeze(2).permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)

        # Pre-LN Transformer block
        h = self.LayerNorm(x_src)
        att_out = self.attention_layer(h)
        att = x_src + self.attn_dropout(att_out)
        
        h2 = self.LayerNorm2(att)
        ff = self.FeedForward(h2)
        out = att + ff

        out = self.gap(out.permute(0, 2, 1)).flatten(1)
        out = self.out(out)
        return out
