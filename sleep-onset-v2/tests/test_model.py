import torch

from model import DualPathSleepNet


def test_shapes_range_and_gradient_flow():
    model = DualPathSleepNet(dropout=0.0)
    x = torch.randn(2, 3, 4, 640)
    fused, diagnostics = model.window_embeddings(
        x[:, 0], return_diagnostics=True
    )
    prediction, state = model(x)
    assert fused.shape == (2, 128)
    assert diagnostics["cnn"].shape == (2, 128)
    assert diagnostics["spectral"].shape == (2, 64)
    assert prediction.shape == (2, 3)
    assert state.shape == (1, 2, 128)
    assert torch.all((prediction >= 0) & (prediction <= 600))
    prediction.mean().backward()
    for component in (
        model.cnn_branches,
        model.spectral_mlp,
        model.spectral_projection,
        model.gru,
        model.head,
    ):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())


def test_sequence_and_streaming_agree_and_reset():
    torch.manual_seed(7)
    model = DualPathSleepNet(dropout=0.0).eval()
    x = torch.randn(1, 5, 4, 640)
    sequence, _ = model(x)
    model.reset_state()
    streamed = torch.stack([model.predict(x[:, i]) for i in range(5)], dim=1)
    assert torch.allclose(sequence, streamed, atol=2e-5, rtol=2e-5)
    model.reset_state()
    first = model.predict(x[:, 0])
    model.predict(x[:, 1])
    model.reset_state()
    assert torch.allclose(first, model.predict(x[:, 0]), atol=1e-6)


def test_future_window_cannot_change_past_predictions():
    model = DualPathSleepNet(dropout=0.0).eval()
    x = torch.randn(1, 4, 4, 640)
    changed = x.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 100
    original, _ = model(x)
    modified, _ = model(changed)
    assert torch.allclose(original[:, :2], modified[:, :2], atol=1e-6)
