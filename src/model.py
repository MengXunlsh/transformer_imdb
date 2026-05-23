"""
Transformer 情感分类模型
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码，让 Transformer 感知词序"""

    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Embedding -> PositionalEncoding -> TransformerEncoder -> MeanPool -> FC"""

    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=2, dropout=0.3, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        logits, _ = self.forward_with_attention(x)
        return logits

    def forward_with_attention(self, x):
        """返回 (logits, [attn_layer1, attn_layer2, ...])
        每个 attn_layer: (batch, tgt_len, src_len)
        """
        pad_mask = (x == 0)
        x = self.embedding(x)
        x = self.pos_encoder(x)

        attn_weights = []
        for layer in self.encoder.layers:
            # Self-Attention
            x2, w = layer.self_attn(
                x, x, x,
                key_padding_mask=pad_mask,
                need_weights=True,
                average_attn_weights=True
            )
            x = layer.norm1(x + layer.dropout1(x2))
            # Feed-Forward
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(x2))
            attn_weights.append(w)

        mask = ~pad_mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x = self.fc(self.dropout(x))
        return x.squeeze(-1), attn_weights
