import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, mel_target, duration, **kwargs):
        mel_loss = self.mse_loss(mel, mel_target)
        duration_predictor_loss = self.l1_loss(duration_predicted, duration.float())
        loss = mel_loss + duration_predictor_loss
        return {"loss": loss, "mel_loss": mel_loss, "dp_loss": duration_predictor_loss}
