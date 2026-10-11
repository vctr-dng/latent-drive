from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from latent_drive.analysis import analyze_embeddings
from latent_drive.config import PathsConfig, ProjectConfig, load_config


def _make_artifacts(tmp_path: Path) -> ProjectConfig:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs" / "representation_audit.toml")
    config = replace(
        config,
        paths=PathsConfig(
            root=tmp_path,
            data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
        ),
        analysis=replace(
            config.analysis,
            retrieval_queries=6,
            retrieval_neighbors=3,
            pca_components=5,
            pair_samples=20,
            outlier_neighbors=3,
            outlier_count=4,
        ),
    )
    config.manifest_dir.mkdir(parents=True)
    config.feature_dir.mkdir(parents=True)
    rng = np.random.default_rng(4)
    rows: list[dict[str, object]] = []
    videos = [
        ("DJI_0001_001", "0001", "train"),
        ("DJI_0001_002", "0001", "validation"),
        ("DJI_0002_001", "0002", "test"),
        ("DJI_0002_002", "0002", "test"),
    ]
    for video_number, (video_id, session_id, split) in enumerate(videos):
        clip_ids: list[str] = []
        embeddings: list[np.ndarray] = []
        requested_timestamps: list[list[float]] = []
        for clip_number in range(6):
            clip_id = f"{video_id}__{clip_number * 1000:09d}"
            requested = [clip_number + offset for offset in (0.125, 0.375, 0.625, 0.875)]
            vector = rng.normal(scale=0.1, size=8)
            vector[video_number] += 1.0
            vector /= np.linalg.vector_norm(vector)
            clip_ids.append(clip_id)
            embeddings.append(vector.astype(np.float32))
            requested_timestamps.append(requested)
            rows.append(
                {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "session_id": session_id,
                    "split": split,
                    "path": str(tmp_path / f"{video_id}.MP4"),
                    "fps_numerator": 50,
                    "fps_denominator": 1,
                    "start_seconds": float(clip_number),
                    "end_seconds": float(clip_number + 1),
                    "requested_timestamps": requested,
                    "actual_timestamps": requested,
                    "shard": f"{video_id}.npz",
                }
            )
        values = np.stack(embeddings)
        timestamps = np.asarray(requested_timestamps)
        np.savez_compressed(
            config.feature_dir / f"{video_id}.npz",
            clip_ids=np.asarray(clip_ids),
            embeddings=values,
            normalized_embeddings=values,
            requested_timestamps=timestamps,
            actual_timestamps=timestamps,
        )
    metadata = pd.DataFrame(rows)
    manifest_columns = [
        column for column in metadata.columns if column not in {"actual_timestamps", "shard"}
    ]
    metadata[manifest_columns].to_parquet(config.manifest_dir / "clips.parquet", index=False)
    metadata.to_parquet(config.feature_dir / "metadata.parquet", index=False)
    return config


def test_embedding_analysis(tmp_path: Path) -> None:
    config = _make_artifacts(tmp_path)
    analysis_dir = analyze_embeddings(config)
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["audit_name"] == "representation_audit"
    assert summary["clip_count"] == 24
    assert summary["embedding_dimension"] == 8
    assert (analysis_dir / "retrievals.parquet").exists()
    assert (analysis_dir / "pair_similarities.parquet").exists()
