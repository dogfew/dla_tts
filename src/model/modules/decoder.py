import torch.nn as nn
from .fft import FFTBlock
from src.model.modules.utils import (
    get_non_pad_mask,
    get_attn_key_pad_mask_heads,
)


class Decoder(nn.Module):
    """Decoder"""

    def __init__(
        self,
        max_seq_len=3000,
        decoder_n_layer=4,
        encoder_dim=256,
        PAD=0,
        encoder_head=2,
        encoder_conv1d_filter_size=1024,
        dropout=0.1,
    ):
        super(Decoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer
        self.PAD = PAD
        self.n_heads = encoder_head
        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    encoder_dim,
                    encoder_conv1d_filter_size,
                    encoder_head,
                    encoder_dim // encoder_head,
                    encoder_dim // encoder_head,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, enc_pos, return_attns=False):
        dec_slf_attn_list = []
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask_heads(
            seq_k=enc_pos, seq_q=enc_pos, n_heads=self.n_heads, PAD=self.PAD
        )
        non_pad_mask = get_non_pad_mask(enc_pos, PAD=self.PAD)
        # -- Forward
        encoded = self.position_enc(enc_pos)
        dec_output = enc_seq + encoded
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
        return dec_output
