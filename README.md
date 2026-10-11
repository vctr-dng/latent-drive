# latent-drive

`latent-drive` builds and audits frozen video representations from passive dashcam footage. The
pipeline inventories source videos, creates leakage-safe one-second clips, decodes frames by
timestamp, extracts V-JEPA 2 embeddings, and generates a representation audit. It does not infer
or apply driving controls.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- `ffprobe` and `ffmpeg` on `PATH`
- An NVIDIA GPU supported by PyTorch CUDA 13.0
- Source videos named `DJI_<session>_<segment>.MP4` under `data/dashcam/`

Install the locked environment:

```bash
uv sync
```

Raw videos and generated artifacts are ignored by Git.

## Pipeline

Run commands from the repository root:

```bash
uv run latent-drive inventory
uv run latent-drive index
uv run latent-drive extract --limit 2
uv run latent-drive extract
uv run latent-drive validate
uv run latent-drive analyze
uv run marimo run notebooks/representation_audit.py
```

The smoke run writes an isolated `smoke.npz` and does not replace full shards. Full extraction
writes one atomic shard per video and reuses valid shards on subsequent runs. Use
`uv run latent-drive extract --force` to regenerate them.

All commands default to `configs/representation_audit.toml`. Select another configuration by
placing the global option before the subcommand:

```bash
uv run latent-drive --config configs/representation_audit.toml validate
```

## Data Contract

- Splits are assigned by complete source video so clips from one video cannot cross splits.
- Clips are non-overlapping and one second long.
- Four frames are requested at clip-relative times `0.125`, `0.375`, `0.625`, and `0.875`
  seconds.
- Frames are resized with aspect-preserving 256 x 256 letterboxing and ImageNet normalization.
- The frozen `facebook/vjepa2-vitl-fpc64-256` encoder runs at a pinned Hub revision using BF16
  and SDPA.
- Patch tokens are spatially and temporally mean-pooled into one 1024-dimensional vector per
  clip. Both raw and L2-normalized vectors are stored.
- Decoded timestamps must remain within one source frame plus 5 ms of requested timestamps.

The configured dataset produces 2,067 clips: 816 train, 141 validation, and 1,110 test.

## Artifacts

```text
artifacts/
  inventory/
    inventory.json
    videos.parquet
  manifests/
    clips.parquet
  features/facebook__vjepa2-vitl-fpc64-256/
    DJI_*.npz
    metadata.parquet
    run.json
    smoke.npz
    smoke-run.json
  analysis/facebook__vjepa2-vitl-fpc64-256/representation_audit/
    summary.json
    pca_coordinates.parquet
    pca_variance.parquet
    pair_similarities.parquet
    centroid_distances.parquet
    retrievals.parquet
    outliers.parquet
notebooks/
  representation_audit.py
```

Each `.npz` shard contains `clip_ids`, `embeddings`, `normalized_embeddings`,
`requested_timestamps`, and `actual_timestamps`. `run.json` records the model revision,
configuration, manifest hash, package/CUDA versions, runtime, decoder devices, GPU, and peak GPU
memory.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run latent-drive validate
uv run marimo check notebooks/representation_audit.py
uv run notebooks/representation_audit.py
```

`validate` checks clip coverage and ordering, array shapes, finite values, unit norms, requested
timestamps, source-frame timestamp tolerance, consistent embedding dimensions, and metadata shard
references without loading model weights.

## Representation Audit

Run the deterministic embedding analysis after full extraction has passed validation:

```bash
uv run latent-drive analyze
uv run marimo run notebooks/representation_audit.py
```

The marimo notebook includes:

- PCA geometry and explained-variance diagnostics
- Top-1 nearest-neighbor concentration by video, session, split, and temporal locality
- Adjacent, nonlocal same-video, same-session, and cross-session similarity baselines
- Split, session, and video centroid distances
- Interactive nearest-neighbor retrievals with midpoint source frames
- Local-density outliers for targeted visual inspection

The audit is descriptive. It does not establish semantic understanding, causal structure, driving
intent, control quality, or downstream task performance. Edit the notebook interactively with
`uv run marimo edit notebooks/representation_audit.py` or execute it non-interactively with
`uv run notebooks/representation_audit.py`.
