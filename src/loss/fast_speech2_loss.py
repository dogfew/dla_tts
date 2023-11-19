import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        mel,
        log_duration_predicted,
        mel_target,
        duration,
        pitch_predicted,
        energy_predicted,
        energy_target,
        pitch_target,
        **kwargs
    ):
        mel_loss = self.l1_loss(mel, mel_target)
        duration_predictor_loss = self.mse_loss(
            log_duration_predicted, duration.float().log1p()
        )
        pitch_loss = self.mse_loss(pitch_predicted, pitch_target.log1p())
        energy_loss = self.mse_loss(energy_predicted, energy_target.log1p())
        loss = mel_loss + duration_predictor_loss + pitch_loss + energy_loss
        return {
            "loss": loss,
            "mel_loss": mel_loss,
            "dp_loss": duration_predictor_loss,
            "energy_loss": energy_loss,
            "pitch_loss": pitch_loss,
        }
