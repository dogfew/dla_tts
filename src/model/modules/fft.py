import torch
import torch.nn as nn
from src.model.modules.attention import MultiHeadAttention
from src.model.modules.utils import Transpose


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(
        self,
        d_in,
        d_hid,
        fft_conv1d_kernel=(9, 1),
        fft_conv1d_padding=(4, 0),
        dropout=0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(
                d_in,
                d_hid,
                kernel_size=fft_conv1d_kernel[0],
                padding=fft_conv1d_padding[0],
            ),
            nn.ReLU(),
            nn.Conv1d(
                d_hid,
                d_in,
                kernel_size=fft_conv1d_kernel[1],
                padding=fft_conv1d_padding[1],
            ),
            Transpose(1, 2),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x):
        return self.net(x) + x


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=1, slf_attn_mask=1):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output * non_pad_mask) * non_pad_mask
        return enc_output, enc_slf_attn
