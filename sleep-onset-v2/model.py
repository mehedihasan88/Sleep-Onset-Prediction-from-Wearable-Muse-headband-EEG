"""Dual-path spectral-temporal CNN-GRU for causal sleep-onset prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from features import SpectralFeatureExtractor

CAP_SECONDS = 600.0


class WindowNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        median = x.median(dim=-1, keepdim=True).values
        mad = (x - median).abs().median(dim=-1, keepdim=True).values
        return ((x - median) / (1.4826 * mad + 1e-3)).clamp(-10.0, 10.0)


class ConvBranch(nn.Module):
    def __init__(self, n_chans: int, kernel: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(32, 32, 7, padding=3, groups=32, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.AvgPool1d(4),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DualPathSleepNet(nn.Module):
    def __init__(
        self,
        n_chans: int = 4,
        cnn_embedding: int = 128,
        spectral_embedding: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.25,
        sampling_rate: float = 128.0,
        n_times: int = 640,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.wave_norm = WindowNorm()
        self.cnn_branches = nn.ModuleList(
            [ConvBranch(n_chans, kernel) for kernel in (7, 15, 31)]
        )
        self.cnn_fusion = nn.Sequential(
            nn.Conv1d(96, cnn_embedding, 1, bias=False),
            nn.GroupNorm(8, cnn_embedding),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.features = SpectralFeatureExtractor(sampling_rate, n_times)
        self.spectral_mlp = nn.Sequential(
            nn.Linear(48, spectral_embedding),
            nn.LayerNorm(spectral_embedding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spectral_embedding, spectral_embedding),
        )
        self.spectral_projection = nn.Linear(spectral_embedding, cnn_embedding)
        self.gru = nn.GRU(
            cnn_embedding,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._state: Tensor | None = None
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start Path B useful but conservative; it is not a zero/V1 initialization.
        nn.init.xavier_normal_(self.spectral_projection.weight, gain=0.25)

    def reset_state(self) -> None:
        self._state = None
        self.gru.flatten_parameters()

    def window_embeddings(
        self, x: Tensor, return_diagnostics: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3 or x.shape[1:] != (self.n_chans, self.n_times):
            raise ValueError(
                f"Expected (B,{self.n_chans},{self.n_times}), got {tuple(x.shape)}"
            )
        normalized = self.wave_norm(x)
        cnn_map = torch.cat([branch(normalized) for branch in self.cnn_branches], 1)
        cnn = self.cnn_fusion(cnn_map).squeeze(-1)
        physiological = self.features(x)
        spectral = self.spectral_mlp(physiological)
        correction = self.spectral_projection(spectral)
        fused = cnn + correction
        if not return_diagnostics:
            return fused
        return fused, {
            "cnn": cnn,
            "physiological": physiological,
            "spectral": spectral,
            "correction": correction,
            "fused": fused,
        }

    def forward(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        single = x.ndim == 3
        if single:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected (B,S,C,T), got {tuple(x.shape)}")
        batch, steps, chans, times = x.shape
        fused = self.window_embeddings(x.reshape(batch * steps, chans, times))
        sequence = fused.reshape(batch, steps, -1)
        output, state = self.gru(sequence, state)
        prediction = torch.sigmoid(self.head(output).squeeze(-1)) * CAP_SECONDS
        return (prediction[:, 0] if single else prediction), state

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        self.eval()
        prediction, state = self.forward(x, self._state)
        self._state = state.detach()
        return torch.nan_to_num(
            prediction, nan=300.0, posinf=CAP_SECONDS, neginf=0.0
        ).clamp(0.0, CAP_SECONDS)
