# Sleep Onset Prediction

A sleeper wearing a Muse headband produces a continuous EEG recording while falling asleep. At the end of every 5-second window, the task is to estimate how many seconds remain before that person first reaches stage N2. The estimate has to be causal: it may use the current window and earlier windows only, since a streaming system has access only to EEG observed so far. Predictions are in seconds, limited to 0–600, for [Neural Interfaces 2026, Track 3](https://www.codabench.org/).

Each 5-second EEG window must be scored using only that window and earlier windows. The model emits seconds remaining until N2, clipped to 0–600. The competition score is the recording-level weighted binned MAE (W-bMAE): bins `[0, 40)`, `[40, 90)`, `[90, 300)`, and `[300, 600]` are weighted `10, 5, 3, 1` inside each recording, then averaged equally across recordings. Lower is better.

## Dataset

Official Interaxon 2026 Muse recordings, loaded with NeuralBench 0.4.0 (`Interaxon2026Muse`, task `_sleep_onset_stream`).

- Four channels: TP9, AF7, AF8, TP10.
- 128 Hz raw EEG, cut into non-overlapping 5-second windows of shape `(4, 640)`.
- Target: `clip(first N2 onset − window end, 0, 600)` seconds.
- Official subject-disjoint split: 412 train, 88 validation, and 40 public-test recordings.
- 61,303 eligible training windows. Recordings vary in length; the longest training recording has 300 windows and the median has 138.

## Tried solutions

Every model normalizes each 5-second window on its own, encodes it with a multi-scale 1D CNN using kernels 7, 15, and 31, and passes that sequence through a unidirectional 128-unit GRU whose sigmoid output is scaled to 0–600 seconds. V2 and V3 also standardize 48 spectral and amplitude features on the training windows and add them to the CNN representation through a residual projection. All three models train from scratch with weighted L1 and AdamW at seed 42: V1 and V2 use shuffled independent chunks of 12 and 24 windows, while V3 backpropagates through each complete recording.

| Version | Training setup | Parameters | Where |
|---|---|---:|---|
| V1 | CNN only, independent 12-window chunks | 128,033 | repository root |
| V2 | Adds a 48-feature spectral path, independent 24-window chunks | 143,777 | `sleep-onset-v2/` |
| V3 | Same network as V2, backpropagation through each complete recording | 143,777 | `sleep-onset-v3/` |

## Findings

Scores are seconds. Validation is the 88-recording development split. Codabench is the 40-recording public test.

| Model | Validation W-bMAE | Codabench W-bMAE | Codabench bMAE | Codabench MAE |
|---|---:|---:|---:|---:|
| V1 | 43.07 | 55.05 | — | — |
| V2 | 41.52 | **47.74** | 96.91 | 166.65 |
| V3 | 41.83 | 51.26 | 92.41 | 151.02 |

## Layout

```text
.
├── model.py, train.py          V1 CNN-GRU
├── train_kaggle_colab.ipynb
├── sleep-onset-v2/             spectral CNN-GRU, 24-window training
└── sleep-onset-v3/             same model, full-recording training
```
