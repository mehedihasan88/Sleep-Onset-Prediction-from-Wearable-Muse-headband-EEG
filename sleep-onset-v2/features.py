"""Deterministic causal physiological features for one 5-second EEG window."""

from __future__ import annotations

import torch
from torch import Tensor, nn

BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))
N_FEATURES = 48


class SpectralFeatureExtractor(nn.Module):
    """Extract 20 log powers, 20 relative powers, 4 log-RMS and 4 log-MAD."""

    def __init__(self, sfreq: float = 128.0, n_times: int = 640) -> None:
        super().__init__()
        self.sfreq = float(sfreq)
        self.n_times = int(n_times)
        self.register_buffer("hann", torch.hann_window(n_times, periodic=True))
        self.register_buffer("feature_mean", torch.zeros(N_FEATURES))
        self.register_buffer("feature_std", torch.ones(N_FEATURES))
        self.register_buffer("scaler_fitted", torch.tensor(False))

    def raw_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.n_times:
            raise ValueError(f"Expected (B, C, {self.n_times}), got {tuple(x.shape)}")
        x = torch.nan_to_num(
            x.float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6
        ).clamp(-1.0e6, 1.0e6)
        centered = x - x.mean(dim=-1, keepdim=True)
        tapered = centered * self.hann
        spectrum = torch.fft.rfft(tapered, dim=-1)
        psd = spectrum.abs().square() / (
            self.sfreq * self.hann.square().sum().clamp_min(1e-12)
        )
        scale = torch.ones_like(psd)
        if self.n_times % 2 == 0:
            scale[..., 1:-1] = 2.0
        else:
            scale[..., 1:] = 2.0
        psd = psd * scale
        frequencies = torch.fft.rfftfreq(
            self.n_times, d=1.0 / self.sfreq, device=x.device
        )
        df = self.sfreq / self.n_times
        powers = []
        for low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            powers.append(psd[..., mask].sum(dim=-1) * df)
        bandpower = torch.stack(powers, dim=-1).clamp_min(0.0)  # (B,C,5)
        log_power = torch.log(bandpower + 1e-12)
        relative = bandpower / bandpower.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rms = centered.square().mean(dim=-1).sqrt()
        median = x.median(dim=-1, keepdim=True).values
        mad = (x - median).abs().median(dim=-1).values
        features = torch.cat(
            [
                log_power.flatten(1),
                relative.flatten(1),
                torch.log(rms + 1e-6),
                torch.log(mad + 1e-6),
            ],
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=30.0, neginf=-30.0)

    def set_scaler(self, mean: Tensor, std: Tensor) -> None:
        if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
            raise ValueError("Spectral scaler must contain exactly 48 values.")
        self.feature_mean.copy_(mean.detach().to(self.feature_mean))
        self.feature_std.copy_(std.detach().to(self.feature_std).clamp_min(1e-5))
        self.scaler_fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.raw_features(x)
        return ((raw - self.feature_mean) / self.feature_std).clamp(-8.0, 8.0)


@torch.inference_mode()
def fit_feature_scaler(
    extractor: SpectralFeatureExtractor,
    batches,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    """Fit mean/std on valid training windows only using stable running moments."""
    count = 0
    total = torch.zeros(N_FEATURES, dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    extractor = extractor.to(device)
    for windows, _targets, valid in batches:
        windows = windows.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True).reshape(-1)
        flat = windows.flatten(0, 1)[valid]
        values = extractor.raw_features(flat).double()
        total += values.sum(dim=0)
        total_sq += values.square().sum(dim=0)
        count += len(values)
    if count < 2:
        raise RuntimeError("At least two training windows are needed for scaling.")
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(1e-10)
    std = variance.sqrt()
    extractor.set_scaler(mean.float(), std.float())
    return mean.float().cpu(), std.float().cpu(), count
