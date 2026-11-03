# Sleep Onset V2

Freshly initialized dual-path spectral-temporal CNN-GRU for Neural Interfaces
2026 Track 3. No V1 weights are loaded or referenced during training.

## Architecture

- Path A: causal-safe per-window median/MAD normalization and kernels 7/15/31
  multi-scale CNN, producing 128 features.
- Path B: 48 deterministic measurements (absolute and relative power in five
  bands per channel, log-RMS and log-MAD), standardized with training-split
  statistics and mapped to 64 features.
- Fusion: `CNN + Linear(spectral)` followed by a unidirectional 128-unit GRU.
- Output: sigmoid regression in `[0, 600]` seconds.

## Kaggle/Colab

Upload this directory as a private Kaggle Dataset and import
`notebooks/train_v2_kaggle_colab.ipynb`. Enable a GPU and Internet, attach the
source dataset, and run all cells. The notebook reuses
`/kaggle/working/neural_compet` when available.

Command-line equivalent:

```bash
python train.py \
  --data-dir /kaggle/working/neural_compet \
  --output-dir /kaggle/working/sleep-v2 \
  --download
```

Use `--resume` to continue from `checkpoints/last_model.pt`. Training stops
with a 15-minute reserve before the configured 4.75-hour budget.

Outputs:

```text
sleep-v2/
├── checkpoints/best_model.pt
├── checkpoints/last_model.pt
├── logs/
├── plots/
├── submission.zip
└── v2_diagnostics.zip
```

Only `submission.zip` goes to Codabench. It contains `submission.py` and
`model_weights.pt` at the ZIP root.

## Tests

```bash
pytest -q
```

The synthetic suite checks features, FFT bands, finite extreme inputs,
gradient flow, causality, state reset, sequence/stream equivalence, exact
metric boundaries, deep-copy support, checkpoint restoration and ZIP layout.
Real-data shape, channel and sampling-rate assertions run when training loads
the official dataset.

## V1 reference

V1 validation W-bMAE was 43.0692 seconds and Codabench warm-up W-bMAE was
55.0472 seconds. V2 results must come from a fresh run and are not known yet.
