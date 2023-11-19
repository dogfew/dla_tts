import torch
from torch import nn

from src.model.modules.decoder import Decoder
from src.model.modules.encoder import Encoder
from src.model.modules.length_regulator import LengthRegulator
from src.model.modules.utils import get_mask_from_lengths


class FastSpeech(nn.Module):
    """FastSpeech"""

    def __init__(
        self,
        max_seq_len=3000,
        decoder_n_layer=4,
        PAD=0,
        encoder_head=2,
        encoder_dim=256,
        encoder_conv1d_filter_size=1024,
        dropout=0.1,
        vocab_size=300,
        encoder_n_layer=4,
        decoder_dim=256,
        duration_predictor_filter_size=256,
        duration_predictor_kernel_size=3,
        num_mels=80,
        **kwargs
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            encoder_dim=encoder_dim,
            encoder_n_layer=encoder_n_layer,
            PAD=PAD,
            encoder_head=encoder_head,
            encoder_conv1d_filter_size=encoder_conv1d_filter_size,
            dropout=dropout,
        )
        self.length_regulator = LengthRegulator(
            encoder_dim=encoder_dim,
            duration_predictor_filter_size=duration_predictor_filter_size,
            duration_predictor_kernel_size=duration_predictor_kernel_size,
            dropout=dropout,
        )
        self.decoder = Decoder(
            max_seq_len=max_seq_len,
            decoder_n_layer=decoder_n_layer,
            encoder_dim=encoder_dim,
            PAD=PAD,
            encoder_head=encoder_head,
            encoder_conv1d_filter_size=encoder_conv1d_filter_size,
            dropout=dropout,
        )

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = (
            ~get_mask_from_lengths(lengths, max_len=mel_max_length)
            .unsqueeze(-1)
            .expand(-1, -1, mel_output.size(-1))
        )
        return mel_output.masked_fill(mask, 0.0)

    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        duration=None,
        alpha=1.0,
        **kwargs
    ):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predictor_output = self.length_regulator(
                x, alpha, duration, mel_max_length
            )
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {"mel": output, "duration_predicted": duration_predictor_output}
        output, mel_pos = self.length_regulator(x, alpha)
        output = self.decoder(output, mel_pos)
        output = self.mel_linear(output)
        return output
