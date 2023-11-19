import torch
import torch.nn as nn
from src.model.modules.utils import Transpose


class VariancePredictor(nn.Module):
    """Duration/Pitch/Energy Predictor"""

    def __init__(
        self,
        encoder_dim,
        duration_predictor_filter_size,
        duration_predictor_kernel_size,
        dropout,
    ):
        super().__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size, kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.conv_output_size, 1),
        )

    def forward(self, encoder_output, mask=None):
        out = self.net(encoder_output).squeeze()
        if self.training:
            if mask is not None:
                out.masked_fill(mask, 0.0)
            return out
        return out.unsqueeze(0)
