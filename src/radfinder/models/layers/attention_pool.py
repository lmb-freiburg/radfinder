import torch
import torch.nn as nn


class AttentionPoolMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # query is (B, 1, D), key/value are (B, S, D)
        q = self.query_token.expand(x.size(0), 1, x.size(-1))
        y, _ = self.mha(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return y[:, 0, :]
