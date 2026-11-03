"""Lightweight causal CNN-GRU for Track 3 sleep-onset prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

CAP_SECONDS = 600.0


class WindowNorm(nn.Module):
    """Per-window robust scaling using no past or future windows."""

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        median = x.median(dim=-1, keepdim=True).values
        mad = (x - median).abs().median(dim=-1, keepdim=True).values
        return ((x - median) / (1.4826 * mad + 1e-3)).clamp_(-10.0, 10.0)


class ConvBranch(nn.Module):
    def __init__(self, n_chans: int, width: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_chans, width, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(
                width,
                width,
                kernel_size=7,
                padding=3,
                groups=width,
                bias=False,
            ),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.AvgPool1d(4),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SleepOnsetNet(nn.Module):
    """Multi-scale window encoder followed by a unidirectional GRU."""

    def __init__(
        self,
        n_chans: int = 4,
        embedding_size: int = 128,
        gru_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.embedding_size = embedding_size
        self.gru_hidden_size = gru_hidden_size
        self.preprocess = WindowNorm()
        self.branches = nn.ModuleList(
            [ConvBranch(n_chans, 32, kernel) for kernel in (7, 15, 31)]
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(96, embedding_size, kernel_size=1, bias=False),
            nn.GroupNorm(8, embedding_size),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden_size),
            nn.Linear(gru_hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self._state: Tensor | None = None

    def reset_state(self) -> None:
        self._state = None
        # deepcopy (used by the official stream evaluator) can invalidate
        # cuDNN's packed RNN-weight layout.
        self.gru.flatten_parameters()

    def encode_windows(self, x: Tensor) -> Tensor:
        """Encode ``(N, C, T)`` windows into ``(N, embedding_size)``."""
        if x.ndim != 3 or x.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected (N, {self.n_chans}, T), got {tuple(x.shape)}"
            )
        x = self.preprocess(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.encoder(x).squeeze(-1)

    def forward(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Predict a sequence.

        ``x`` may be ``(B, S, C, T)`` or one sequence ``(B, C, T)``.
        State is explicit here so training never mutates streaming state.
        """
        single_step = x.ndim == 3
        if single_step:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected (B, S, C, T), got {tuple(x.shape)}")
        batch, steps, chans, times = x.shape
        embeddings = self.encode_windows(x.reshape(batch * steps, chans, times))
        embeddings = embeddings.reshape(batch, steps, -1)
        output, state = self.gru(embeddings, state)
        seconds = torch.sigmoid(self.head(output).squeeze(-1)) * CAP_SECONDS
        return (seconds[:, 0] if single_step else seconds), state

    @torch.inference_mode()
    def predict(self, x: Tensor) -> Tensor:
        """Official streamed inference API; preserve state within a recording."""
        self.eval()
        prediction, state = self.forward(x, self._state)
        self._state = state.detach()
        return torch.nan_to_num(
            prediction, nan=300.0, posinf=CAP_SECONDS, neginf=0.0
        ).clamp_(0.0, CAP_SECONDS)
