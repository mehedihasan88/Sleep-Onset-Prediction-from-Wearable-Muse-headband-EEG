"""Self-contained Codabench submission for Track 3."""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

CAP_SECONDS = 600.0


class WindowNorm(nn.Module):
    def forward(self, x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        median = x.median(dim=-1, keepdim=True).values
        mad = (x - median).abs().median(dim=-1, keepdim=True).values
        return ((x - median) / (1.4826 * mad + 1e-3)).clamp_(-10.0, 10.0)


class ConvBranch(nn.Module):
    def __init__(self, n_chans, width, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                n_chans, width, kernel_size, padding=kernel_size // 2, bias=False
            ),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(
                width, width, kernel_size=7, padding=3, groups=width, bias=False
            ),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.AvgPool1d(4),
        )

    def forward(self, x):
        return self.net(x)


class SleepOnsetNet(nn.Module):
    def __init__(self, n_chans=4, embedding_size=128, gru_hidden_size=128):
        super().__init__()
        self.n_chans = n_chans
        self.preprocess = WindowNorm()
        self.branches = nn.ModuleList(
            [ConvBranch(n_chans, 32, kernel) for kernel in (7, 15, 31)]
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(96, embedding_size, 1, bias=False),
            nn.GroupNorm(8, embedding_size),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gru = nn.GRU(
            embedding_size, gru_hidden_size, batch_first=True, num_layers=1
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden_size),
            nn.Linear(gru_hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self._state = None

    def reset_state(self):
        self._state = None
        self.gru.flatten_parameters()

    def encode_windows(self, x):
        x = self.preprocess(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.encoder(x).squeeze(-1)

    def forward(self, x, state=None):
        single_step = x.ndim == 3
        if single_step:
            x = x.unsqueeze(1)
        batch, steps, chans, times = x.shape
        embeddings = self.encode_windows(x.reshape(batch * steps, chans, times))
        output, state = self.gru(embeddings.reshape(batch, steps, -1), state)
        seconds = torch.sigmoid(self.head(output).squeeze(-1)) * CAP_SECONDS
        return (seconds[:, 0] if single_step else seconds), state

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        prediction, state = self.forward(x, self._state)
        self._state = state.detach()
        return torch.nan_to_num(
            prediction, nan=300.0, posinf=CAP_SECONDS, neginf=0.0
        ).clamp_(0.0, CAP_SECONDS)


class Solver(CompetSolver):
    name = "MultiScale-Causal-GRU"

    def load_model(self, meta):
        checkpoint_path = meta["submission_dir"] / "best_model.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=meta["device"], weights_only=True
        )
        config = checkpoint["model_config"]
        if int(config["n_chans"]) != int(meta["n_chans"]):
            raise ValueError(
                f"Checkpoint has {config['n_chans']} channels, "
                f"but evaluation provides {meta['n_chans']}."
            )
        model = SleepOnsetNet(**config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return model.to(meta["device"]).eval()
