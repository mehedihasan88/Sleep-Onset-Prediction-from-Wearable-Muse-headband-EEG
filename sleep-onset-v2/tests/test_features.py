import torch

from features import SpectralFeatureExtractor


def test_feature_shape_and_finiteness():
    extractor = SpectralFeatureExtractor()
    for x in (torch.zeros(3, 4, 640), torch.full((3, 4, 640), 1.0e9)):
        features = extractor.raw_features(x)
        assert features.shape == (3, 48)
        assert torch.isfinite(features).all()


def test_delta_sinusoid_has_largest_absolute_bandpower():
    t = torch.arange(640) / 128.0
    signal = torch.sin(2 * torch.pi * 2.0 * t).repeat(2, 4, 1)
    features = SpectralFeatureExtractor().raw_features(signal)
    log_power = features[:, :20].reshape(2, 4, 5)
    assert torch.all(log_power.argmax(-1) == 0)


def test_cpu_gpu_consistency():
    if not torch.cuda.is_available():
        return
    x = torch.randn(2, 4, 640)
    cpu = SpectralFeatureExtractor().raw_features(x)
    gpu = SpectralFeatureExtractor().cuda().raw_features(x.cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=2e-4, rtol=2e-4)
