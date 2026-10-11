from __future__ import annotations

from pathlib import Path

import pandas as pd

from latent_drive.config import (
    AnalysisConfig,
    ClipsConfig,
    EncoderConfig,
    PathsConfig,
    PreprocessConfig,
    ProjectConfig,
    SplitsConfig,
)
from latent_drive.data import make_clip_manifest, sample_timestamps


def make_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        paths=PathsConfig(
            root=tmp_path,
            data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
        ),
        clips=ClipsConfig(duration_seconds=1.0, sampling_fps=4.0),
        splits=SplitsConfig(train=("DJI_0001_001",), validation=(), test=()),
        encoder=EncoderConfig(
            model_id="example/model",
            revision="revision",
            device="cpu",
            decode_device="cpu",
            dtype="float32",
            attention="sdpa",
            batch_size=2,
        ),
        preprocess=PreprocessConfig(mode="letterbox", size=256),
        analysis=AnalysisConfig(seed=1, retrieval_queries=1, retrieval_neighbors=1),
    )


def test_sample_timestamps_use_frame_centers() -> None:
    assert sample_timestamps(10.0, 1.0, 4.0) == [10.125, 10.375, 10.625, 10.875]


def test_manifest_only_contains_complete_clips(tmp_path: Path) -> None:
    inventory = pd.DataFrame(
        [
            {
                "video_id": "DJI_0001_001",
                "session_id": "0001",
                "path": "/video.mp4",
                "duration_seconds": 2.9,
                "fps_numerator": 50,
                "fps_denominator": 1,
            }
        ]
    )
    manifest = make_clip_manifest(inventory, make_config(tmp_path))
    assert manifest["clip_id"].tolist() == ["DJI_0001_001__000000000", "DJI_0001_001__000001000"]
    assert manifest["split"].tolist() == ["train", "train"]
    assert manifest["end_seconds"].max() == 2.0
