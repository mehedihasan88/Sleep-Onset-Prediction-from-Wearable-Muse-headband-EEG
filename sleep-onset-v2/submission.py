"""Self-contained Codabench inference for the V2 dual-path model."""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

CAP_SECONDS = 600.0
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, sfreq=128.0, n_times=640):
        super().__init__()
        self.sfreq, self.n_times = float(sfreq), int(n_times)
        self.register_buffer("hann", torch.hann_window(n_times, periodic=True))
        self.register_buffer("feature_mean", torch.zeros(48))
        self.register_buffer("feature_std", torch.ones(48))
        self.register_buffer("scaler_fitted", torch.tensor(False))

    def raw_features(self, x):
        x = torch.nan_to_num(
            x.float(), nan=0.0, posinf=1e6, neginf=-1e6
        ).clamp(-1e6, 1e6)
        centered = x - x.mean(-1, keepdim=True)
        spectrum = torch.fft.rfft(centered * self.hann, dim=-1)
        psd = spectrum.abs().square() / (
            self.sfreq * self.hann.square().sum().clamp_min(1e-12)
        )
        scale = torch.ones_like(psd)
        scale[..., 1:-1] = 2.0
        psd = psd * scale
        frequencies = torch.fft.rfftfreq(
            self.n_times, d=1.0 / self.sfreq, device=x.device
        )
        powers = []
        for low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            powers.append(psd[..., mask].sum(-1) * (self.sfreq / self.n_times))
        bandpower = torch.stack(powers, -1).clamp_min(0.0)
        relative = bandpower / bandpower.sum(-1, keepdim=True).clamp_min(1e-12)
        rms = centered.square().mean(-1).sqrt()
        median = x.median(-1, keepdim=True).values
        mad = (x - median).abs().median(-1).values
        values = torch.cat(
            [
                torch.log(bandpower + 1e-12).flatten(1),
                relative.flatten(1),
                torch.log(rms + 1e-6),
                torch.log(mad + 1e-6),
            ],
            1,
        )
        return torch.nan_to_num(values, nan=0.0, posinf=30.0, neginf=-30.0)

    def forward(self, x):
        return ((self.raw_features(x) - self.feature_mean) / self.feature_std).clamp(
            -8.0, 8.0
        )


class WindowNorm(nn.Module):
    def forward(self, x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        median = x.median(-1, keepdim=True).values
        mad = (x - median).abs().median(-1, keepdim=True).values
        return ((x - median) / (1.4826 * mad + 1e-3)).clamp(-10.0, 10.0)


class ConvBranch(nn.Module):
    def __init__(self, n_chans, kernel):
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

    def forward(self, x):
        return self.net(x)


class DualPathSleepNet(nn.Module):
    def __init__(
        self,
        n_chans=4,
        cnn_embedding=128,
        spectral_embedding=64,
        gru_hidden=128,
        gru_layers=1,
        dropout=0.25,
        sampling_rate=128.0,
        n_times=640,
    ):
        super().__init__()
        self.n_chans, self.n_times = n_chans, n_times
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
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._state = None

    def reset_state(self):
        self._state = None
        self.gru.flatten_parameters()

    def window_embeddings(self, x):
        normalized = self.wave_norm(x)
        cnn = self.cnn_fusion(
            torch.cat([branch(normalized) for branch in self.cnn_branches], 1)
        ).squeeze(-1)
        spectral = self.spectral_mlp(self.features(x))
        return cnn + self.spectral_projection(spectral)

    def forward(self, x, state=None):
        single = x.ndim == 3
        if single:
            x = x.unsqueeze(1)
        batch, steps, chans, times = x.shape
        embeddings = self.window_embeddings(
            x.reshape(batch * steps, chans, times)
        ).reshape(batch, steps, -1)
        output, state = self.gru(embeddings, state)
        prediction = torch.sigmoid(self.head(output).squeeze(-1)) * CAP_SECONDS
        return (prediction[:, 0] if single else prediction), state

    @torch.inference_mode()
    def predict(self, x):
        self.eval()
        prediction, state = self.forward(x, self._state)
        self._state = state.detach()
        return torch.nan_to_num(
            prediction, nan=300.0, posinf=600.0, neginf=0.0
        ).clamp(0.0, 600.0)


class Solver(CompetSolver):
    name = "DualPath-Spectral-CNN-GRU-V2"

    def load_model(self, meta):
        checkpoint = torch.load(
            meta["submission_dir"] / "model_weights.pt",
            map_location=meta["device"],
            weights_only=True,
        )
        config = checkpoint["model_config"]
        if config["n_chans"] != meta["n_chans"]:
            raise ValueError("Evaluation channel count does not match V2 checkpoint.")
        model = DualPathSleepNet(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        scaler = checkpoint["spectral_feature_scaler"]
        expected_mean = scaler["mean"].to(model.features.feature_mean)
        if not torch.allclose(model.features.feature_mean, expected_mean):
            raise RuntimeError("Spectral feature scaler was not restored.")
        return model.to(meta["device"]).eval()
