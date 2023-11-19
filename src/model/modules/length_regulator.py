import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.modules.variance_predictor import VariancePredictor


def create_alignment(duration_predictor_output, dtype):
    N, L = duration_predictor_output.shape
    M = torch.sum(duration_predictor_output, dim=1).max()
    base_mat = torch.zeros(
        N, M, L, dtype=dtype, device=duration_predictor_output.device
    )
    end_indices = duration_predictor_output.cumsum(dim=1)
    start_indices = end_indices - duration_predictor_output
    idx = (
        torch.arange(M, device=duration_predictor_output.device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    mask = (idx >= start_indices.unsqueeze(1)) & (idx < end_indices.unsqueeze(1))
    base_mat[mask] = 1
    return base_mat


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, **kwargs):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(**kwargs)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        with torch.no_grad():
            alignment = create_alignment(duration_predictor_output, x.dtype)
        output = alignment @ x
        if mel_max_length:
            output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        log_duration_pred = self.duration_predictor(x)
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, log_duration_pred
        else:
            output = self.LR(x, (log_duration_pred.expm1() * alpha).int())
            mel_pos = torch.arange(
                1, output.size(1) + 1, dtype=torch.long, device=x.device
            ).unsqueeze(0)
            return output, mel_pos
