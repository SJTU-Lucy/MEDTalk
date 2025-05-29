import torch
import torch.nn as nn


class AudioTextFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super(AudioTextFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.predictor = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        attn_output, _ = self.cross_attn(Q, K, V)
        Q = self.norm1(Q + attn_output)

        linear_out = self.dropout(self.linear(Q))
        output = self.norm2(Q + linear_out)
        output = self.predictor(output)

        return output
