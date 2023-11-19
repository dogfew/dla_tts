import torch.nn as nn
from src.model.modules.utils import QuantizationEmbedding
from src.model.modules.variance_predictor import VariancePredictor


class PitchRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, **kwargs):
        super().__init__()
        self.pitch_predictor = VariancePredictor(**kwargs)
        # self.pitch_embedding = QuantizationEmbedding(values_range=(4.12, 6.76), large_zero=True)
        self.pitch_embedding = QuantizationEmbedding(
            values_range=(0, 6.76), large_zero=False
        )

    def forward(self, x, beta=1.0, target=None, mask=None, **kwargs):
        pitch_pred = self.pitch_predictor(x, mask)
        embed = self.pitch_embedding(
            target if target is not None else pitch_pred.expm1() * beta
        )
        return embed, pitch_pred
