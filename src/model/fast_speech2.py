import torch
from torch import nn

from src.model.modules.decoder import Decoder
from src.model.modules.encoder import Encoder
from src.model.modules.energy_regulator import EnergyRegulator
from src.model.modules.length_regulator import LengthRegulator
from src.model.modules.pitch_regulator import PitchRegulator
from src.model.modules.utils import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """FastSpeech2"""

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
            pitch_min=0,
            pitch_max=6.76,
            energy_min=0.0177,
            energy_max=5.756,
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
        # Variance Adaptor
        self.length_regulator = LengthRegulator(
            encoder_dim=encoder_dim,
            duration_predictor_filter_size=duration_predictor_filter_size,
            duration_predictor_kernel_size=duration_predictor_kernel_size,
            dropout=dropout,
        )
        self.energy_regulator = EnergyRegulator(
            energy_min=energy_min, energy_max=energy_max,
            encoder_dim=encoder_dim,
            duration_predictor_filter_size=duration_predictor_filter_size,
            duration_predictor_kernel_size=duration_predictor_kernel_size,
            dropout=dropout,
        )
        self.pitch_regulator = PitchRegulator(
            pitch_min=pitch_min, pitch_max=pitch_max,
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
            pitch_target=None,
            energy_target=None,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
            **kwargs
    ):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            assert pitch_target is not None
            assert energy_target is not None
            x, log_duration_predicted = self.length_regulator(
                x, alpha, duration, mel_max_length
            )
            mask = mel_pos == 0
            pitch_emb, pitch_predictor_output = self.pitch_regulator(
                x, beta, pitch_target, mask=mask
            )
            energy_emb, energy_predictor_output = self.energy_regulator(
                x, gamma, energy_target, mask=mask
            )
            x = x + pitch_emb + energy_emb
            x = self.decoder(x, mel_pos)
            x = self.mask_tensor(x, mel_pos, mel_max_length)
            x = self.mel_linear(x)
            return {
                "mel": x,
                "log_duration_predicted": log_duration_predicted,
                "pitch_predicted": pitch_predictor_output,
                "energy_predicted": energy_predictor_output,
            }
        x, mel_pos = self.length_regulator(x, alpha)
        pitch_emb, pitch_predictor_output = self.pitch_regulator(x, beta)
        energy_emb, energy_predictor_output = self.energy_regulator(x, gamma)
        x = x + pitch_emb + energy_emb
        x = self.decoder(x, mel_pos)
        x = self.mel_linear(x)
        return x
