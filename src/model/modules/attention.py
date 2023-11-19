import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        attn_output, attn_output_weights = self.multihead_attn(q, k, v, attn_mask=mask)
        output = self.layer_norm(attn_output + q)
        return output, attn_output_weights
