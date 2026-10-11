from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from latent_drive.config import PathsConfig, load_config
from latent_drive.features import (
    letterbox_frames,
    pool_vjepa_tokens,
    validate_feature_artifacts,
)


def test_letterbox_preserves_shape_and_zero_normalizes_padding() -> None:
    frames = torch.full((2, 3, 90, 160), 127, dtype=torch.uint8)
    output = letterbox_frames(frames, 256)
    assert output.shape == (2, 3, 256, 256)
    padding = output[:, :, 0, :]
    assert torch.allclose(padding, torch.zeros_like(padding), atol=1e-6)


def test_pool_vjepa_tokens() -> None:
    tokens = torch.arange(2 * 512 * 8, dtype=torch.float32).reshape(2, 512, 8)
    embeddings, normalized = pool_vjepa_tokens(tokens, 4, 256, 16, 2)
    assert embeddings.shape == (2, 8)
    assert normalized.shape == (2, 8)
    np.testing.assert_allclose(torch.linalg.vector_norm(normalized, dim=1), [1.0, 1.0])


def test_validate_feature_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "configs" / "representation_audit.toml")
    config = replace(
        config,
        paths=PathsConfig(
            root=tmp_path,
            data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
        ),
    )
    config.manifest_dir.mkdir(parents=True)
    config.feature_dir.mkdir(parents=True)
    clip_id = "DJI_0477_001__000000000"
    requested = np.array([[0.125, 0.375, 0.625, 0.875]])
    manifest = pd.DataFrame(
        {
            "clip_id": [clip_id],
            "video_id": ["DJI_0477_001"],
            "fps_numerator": [50],
            "fps_denominator": [1],
            "requested_timestamps": requested.tolist(),
        }
    )
    manifest.to_parquet(config.manifest_dir / "clips.parquet", index=False)
    embedding = np.array([[3.0, 4.0]], dtype=np.float32)
    np.savez_compressed(
        config.feature_dir / "DJI_0477_001.npz",
        clip_ids=np.array([clip_id]),
        embeddings=embedding,
        normalized_embeddings=embedding / 5.0,
        requested_timestamps=requested,
        actual_timestamps=requested + 0.01,
    )
    manifest.assign(
        actual_timestamps=(requested + 0.01).tolist(),
        shard="DJI_0477_001.npz",
    ).to_parquet(config.feature_dir / "metadata.parquet", index=False)

    metrics = validate_feature_artifacts(config)
    assert metrics["clip_count"] == 1
    assert metrics["embedding_dimension"] == 2
    assert metrics["maximum_timestamp_error_seconds"] == pytest.approx(0.01)

    np.savez_compressed(
        config.feature_dir / "DJI_0477_001.npz",
        clip_ids=np.array([clip_id]),
        embeddings=embedding,
        normalized_embeddings=embedding,
        requested_timestamps=requested,
        actual_timestamps=requested + 0.01,
    )
    with pytest.raises(ValueError, match="not unit normalized"):
        validate_feature_artifacts(config)
