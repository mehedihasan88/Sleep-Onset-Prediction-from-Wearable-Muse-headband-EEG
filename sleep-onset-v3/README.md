# Sleep Onset V3 — Full-Recording Training

Standalone controlled experiment for Neural Interfaces 2026 Track 3.

V3 preserves V2's 143,777-parameter dual-path CNN-GRU, preprocessing,
spectral features, weighted L1, optimizer, scheduler, split, and seed. The
only intended experimental change is training the GRU on each complete
chronological recording instead of independent 24-window chunks.

No V2 checkpoint or saved scaler is required. The scaler is recomputed from
the same 61,303 eligible training windows through the unchanged V2
24-window scaler-fitting view.

The CNN uses mixed precision. The GRU, regression head, and loss stay in
fp32: backpropagation through a complete recording overflowed fp16 gradients.
An isolated non-finite gradient skips that optimizer update and lowers the
loss scale instead of stopping the run.

## Kaggle

Upload this entire folder as a private Kaggle Dataset and import:

```text
notebooks/train_v3_kaggle_colab.ipynb
```

Enable a GPU and Internet, attach the source Dataset, then choose **Run All**.
The script runs correctness tests, fits the scaler, benchmarks median, p90,
and longest recordings, and then trains.

Command-line equivalent:

```bash
python train.py \
  --data-dir /kaggle/working/neural_compet \
  --output-dir /kaggle/working/sleep-onset-v3 \
  --download
```

Use `--benchmark-only` to stop after scaler fitting and benchmarking.
Full-recording OOM raises an error; it never silently switches to truncated
BPTT.

## Outputs

```text
sleep-onset-v3/
├── checkpoints/
│   ├── best_model.pt
│   └── last_model.pt
├── logs/
│   ├── epoch_metrics.csv
│   ├── runtime_benchmark.json
│   ├── optimization_diagnostics.csv
│   ├── validation_predictions.csv
│   └── per_recording_metrics.csv
├── plots/
├── experiment_summary.md
├── submission.zip
└── v3_diagnostics.zip
```

Upload only `submission.zip` to Codabench.

## Local verification

```bash
python -m pytest -q
```
