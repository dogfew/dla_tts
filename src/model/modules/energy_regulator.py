import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.modules.utils import QuantizationEmbedding
from src.model.modules.variance_predictor import VariancePredictor


class EnergyRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, **kwargs):
        super().__init__()
        self.energy_predictor = VariancePredictor(**kwargs)
        self.energy_embedding = QuantizationEmbedding(values_range=(0.0177, 5.756))

    def forward(self, x, gamma=1.0, target=None, mask=None, **kwargs):
        energy_pred = self.energy_predictor(x, mask)
        embed = self.energy_embedding(
            target if target is not None else energy_pred.expm1() * gamma
        )
        return embed, energy_pred