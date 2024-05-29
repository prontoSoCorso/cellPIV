"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

import sys
#sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/Tesi Magistrale/cellPIV")
sys.path.append("/home/giovanna/Desktop/Lorenzo/Tesi Magistrale/cellPIV")
from networksTemporalSeries.blastoFormer.blocks.encoder_layer import EncoderLayer
from networksTemporalSeries.blastoFormer.embedding.positional_encoding import PostionalEncoding


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob,details, device):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x ): 
        for layer in self.layers:
            x = layer(x ) 
        return x
