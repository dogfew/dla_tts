import torch.nn as nn
from .fft import FFTBlock
from src.model.modules.utils import (
    get_non_pad_mask,
    get_attn_key_pad_mask_heads,
)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size=300,
        max_seq_len=3000,
        encoder_dim=256,
        encoder_n_layer=4,
        PAD=0,
        encoder_head=2,
        encoder_conv1d_filter_size=1024,
        dropout=0.1,
    ):
        super(Encoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer
        self.PAD = PAD
        self.n_heads = encoder_head
        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD,
        )

        self.position_enc = nn.Embedding(n_position, encoder_dim, padding_idx=PAD)

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

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []
        slf_attn_mask = get_attn_key_pad_mask_heads(
            seq_k=src_seq, seq_q=src_seq, n_heads=self.n_heads, PAD=self.PAD
        )
        non_pad_mask = get_non_pad_mask(src_seq, PAD=self.PAD)
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask
