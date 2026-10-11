from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from latent_drive.config import ProjectConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_frames(frames: Any, size: int) -> Any:
    """Resize TCHW or BTCHW frames to a mean-padded square and normalize them."""
    import torch
    import torch.nn.functional as functional

    if frames.ndim not in {4, 5}:
        raise ValueError(f"Expected TCHW or BTCHW frames, got shape {tuple(frames.shape)}")
    original_shape = frames.shape
    flat = frames.reshape(-1, *frames.shape[-3:]).to(dtype=torch.float32) / 255.0
    _, _, height, width = flat.shape
    scale = min(size / height, size / width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    resized = functional.interpolate(
        flat,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    mean = torch.tensor(IMAGENET_MEAN, device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
    canvas = mean.expand(flat.shape[0], 3, size, size).clone()
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[:, :, top : top + resized_height, left : left + resized_width] = resized
    normalized = (canvas - mean) / std
    return normalized.reshape(*original_shape[:-3], 3, size, size)


def pool_vjepa_tokens(
    tokens: Any,
    frame_count: int,
    image_size: int,
    patch_size: int,
    tubelet_size: int,
) -> tuple[Any, Any]:
    import torch.nn.functional as functional

    temporal_steps = frame_count // tubelet_size
    spatial_tokens = (image_size // patch_size) ** 2
    expected_tokens = temporal_steps * spatial_tokens
    if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"Expected [B, {expected_tokens}, D] V-JEPA tokens, got {tuple(tokens.shape)}"
        )
    temporal = tokens.reshape(
        tokens.shape[0], temporal_steps, spatial_tokens, tokens.shape[2]
    ).mean(2)
    global_embeddings = temporal.mean(1)
    normalized_embeddings = functional.normalize(global_embeddings, dim=-1)
    return global_embeddings, normalized_embeddings


class VJepaEncoder:
    def __init__(self, config: ProjectConfig) -> None:
        import torch
        from transformers import AutoModel

        if config.encoder.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("The configured CUDA device is unavailable")
        self.config = config
        self.device = torch.device(config.encoder.device)
        self.dtype = getattr(torch, config.encoder.dtype)
        self.model = AutoModel.from_pretrained(
            config.encoder.model_id,
            revision=config.encoder.revision,
            dtype=self.dtype,
            attn_implementation=config.encoder.attention,
        ).to(self.device)
        self.model.eval()

    def encode(self, frames: Any) -> tuple[np.ndarray, np.ndarray]:
        import torch

        frame_count = frames.shape[1]
        frames = letterbox_frames(
            frames.to(self.device, non_blocking=True), self.config.preprocess.size
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            tokens = self.model.get_vision_features(pixel_values_videos=frames)
            embeddings, normalized = pool_vjepa_tokens(
                tokens,
                frame_count=frame_count,
                image_size=self.config.preprocess.size,
                patch_size=int(self.model.config.patch_size),
                tubelet_size=int(self.model.config.tubelet_size),
            )
        return (
            embeddings.float().cpu().numpy(),
            normalized.float().cpu().numpy(),
        )


def _open_decoder(path: str, preferred_device: str) -> tuple[Any, str]:
    from torchcodec.decoders import VideoDecoder

    try:
        return VideoDecoder(path, device=preferred_device), preferred_device
    except (RuntimeError, ValueError) as error:
        if preferred_device == "cpu":
            raise
        print(f"Decoder could not use {preferred_device} for {Path(path).name}: {error}")
        print("Falling back to CPU video decoding")
        return VideoDecoder(path, device="cpu"), "cpu"


def _decode_frames(
    decoder: Any,
    seconds: np.ndarray,
    decoder_device: str,
    path: str,
) -> tuple[Any, np.ndarray, Any, str]:
    try:
        decoded = decoder.get_frames_played_at(seconds=seconds.tolist())
        return decoded.data, decoded.pts_seconds.cpu().numpy(), decoder, decoder_device
    except (RuntimeError, ValueError) as error:
        if decoder_device == "cpu":
            raise
        print(f"CUDA decode failed for {Path(path).name}: {error}")
        decoder, decoder_device = _open_decoder(path, "cpu")
        decoded = decoder.get_frames_played_at(seconds=seconds.tolist())
        return decoded.data, decoded.pts_seconds.cpu().numpy(), decoder, decoder_device


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_existing_shard(path: Path, rows: pd.DataFrame) -> dict[str, float | int]:
    required = {
        "clip_ids",
        "embeddings",
        "normalized_embeddings",
        "requested_timestamps",
        "actual_timestamps",
    }
    with np.load(path, allow_pickle=False) as shard:
        missing = required.difference(shard.files)
        if missing:
            raise ValueError(f"Shard {path} is missing arrays: {sorted(missing)}")
        clip_ids = shard["clip_ids"].astype(str).tolist()
        embeddings = shard["embeddings"]
        normalized = shard["normalized_embeddings"]
        requested = shard["requested_timestamps"]
        actual = shard["actual_timestamps"]

    expected_clip_ids = rows["clip_id"].astype(str).tolist()
    expected_requested = np.stack(rows["requested_timestamps"].map(np.asarray))
    expected_shape = expected_requested.shape
    if clip_ids != expected_clip_ids:
        raise ValueError(f"Existing shard does not match the current manifest: {path}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows) or embeddings.shape[1] == 0:
        raise ValueError(f"Invalid embedding shape {embeddings.shape} in {path}")
    if normalized.shape != embeddings.shape:
        raise ValueError(f"Normalized embedding shape does not match in {path}")
    if requested.shape != expected_shape or actual.shape != expected_shape:
        raise ValueError(f"Timestamp shape does not match the current manifest: {path}")
    if not np.allclose(requested, expected_requested, rtol=0.0, atol=1e-9):
        raise ValueError(f"Existing shard does not match the current manifest: {path}")
    arrays = (embeddings, normalized, requested, actual)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Non-finite values found in {path}")

    norms = np.linalg.vector_norm(normalized, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Embeddings are not unit normalized in {path}")
    fps = (rows["fps_numerator"] / rows["fps_denominator"]).to_numpy()
    errors = np.abs(actual - requested).max(axis=1)
    if np.any(errors > 1.0 / fps + 0.005):
        raise ValueError(f"Decoded timestamps exceed one source frame in {path}")
    return {
        "clip_count": len(rows),
        "embedding_dimension": embeddings.shape[1],
        "maximum_timestamp_error_seconds": float(errors.max()),
        "minimum_embedding_norm": float(norms.min()),
        "maximum_embedding_norm": float(norms.max()),
    }


def _extract_rows(
    rows: pd.DataFrame,
    encoder: VJepaEncoder,
    config: ProjectConfig,
    output_path: Path,
    description: str,
) -> dict[str, Any]:
    import torch

    requested = np.stack(rows["requested_timestamps"].map(np.asarray))
    frames_per_clip = requested.shape[1]
    decoder, decoder_device = _open_decoder(rows.iloc[0]["path"], config.encoder.decode_device)
    all_embeddings: list[np.ndarray] = []
    all_normalized: list[np.ndarray] = []
    all_actual_timestamps: list[np.ndarray] = []

    progress = tqdm(range(0, len(rows), config.encoder.batch_size), desc=description)
    for start in progress:
        stop = min(start + config.encoder.batch_size, len(rows))
        target_times = requested[start:stop]
        flat_times = target_times.reshape(-1)
        frames, actual_times, decoder, decoder_device = _decode_frames(
            decoder,
            flat_times,
            decoder_device,
            rows.iloc[0]["path"],
        )
        frames = frames.reshape(stop - start, frames_per_clip, *frames.shape[1:])
        actual_times = actual_times.reshape(stop - start, frames_per_clip)
        fps = rows.iloc[start]["fps_numerator"] / rows.iloc[start]["fps_denominator"]
        maximum_error = float(np.abs(actual_times - target_times).max())
        if maximum_error > (1.0 / fps + 0.005):
            raise ValueError(
                f"Decoded timestamp error {maximum_error:.4f}s exceeds one source frame"
            )
        embeddings, normalized = encoder.encode(frames)
        all_embeddings.append(embeddings)
        all_normalized.append(normalized)
        all_actual_timestamps.append(actual_times)
        if torch.cuda.is_available():
            progress.set_postfix(memory_gb=f"{torch.cuda.max_memory_allocated() / 2**30:.2f}")

    arrays = {
        "clip_ids": rows["clip_id"].to_numpy(dtype=str),
        "embeddings": np.concatenate(all_embeddings).astype(np.float32),
        "normalized_embeddings": np.concatenate(all_normalized).astype(np.float32),
        "requested_timestamps": requested.astype(np.float64),
        "actual_timestamps": np.concatenate(all_actual_timestamps).astype(np.float64),
    }
    if not np.isfinite(arrays["embeddings"]).all():
        raise ValueError(f"Non-finite embeddings produced for {description}")
    _write_npz_atomic(output_path, **arrays)
    return {"decoder_device": decoder_device, "clip_count": len(rows)}


def extract_features(
    config: ProjectConfig,
    *,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    import torch

    manifest_path = config.manifest_dir / "clips.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError("Run `latent-drive index` before extracting features")
    manifest = pd.read_parquet(manifest_path)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")

    config.feature_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    encoder = VJepaEncoder(config)
    decoder_devices: set[str] = set()

    if limit is not None:
        rows = manifest.iloc[:limit].reset_index(drop=True)
        output_path = config.feature_dir / "smoke.npz"
        result = _extract_rows(rows, encoder, config, output_path, "smoke")
        decoder_devices.add(result["decoder_device"])
    else:
        for video_id, rows in manifest.groupby("video_id", sort=False):
            rows = rows.reset_index(drop=True)
            output_path = config.feature_dir / f"{video_id}.npz"
            if output_path.exists() and not force:
                _validate_existing_shard(output_path, rows)
                print(f"Reusing {output_path.name}")
                continue
            result = _extract_rows(rows, encoder, config, output_path, video_id)
            decoder_devices.add(result["decoder_device"])
        _write_feature_metadata(manifest, config)
        output_path = config.feature_dir

    run_metadata = {
        "model_id": config.encoder.model_id,
        "revision": config.encoder.revision,
        "encoder": asdict(config.encoder),
        "preprocess": asdict(config.preprocess),
        "clips": asdict(config.clips),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "limited_clip_count": limit,
        "elapsed_seconds": time.monotonic() - started,
        "decoder_devices": sorted(decoder_devices),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "torchcodec_version": importlib.metadata.version("torchcodec"),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    }
    metadata_name = "smoke-run.json" if limit is not None else "run.json"
    (config.feature_dir / metadata_name).write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def _write_feature_metadata(manifest: pd.DataFrame, config: ProjectConfig) -> None:
    actual_by_clip: dict[str, list[float]] = {}
    shard_by_clip: dict[str, str] = {}
    for video_id in manifest["video_id"].drop_duplicates():
        shard_path = config.feature_dir / f"{video_id}.npz"
        with np.load(shard_path, allow_pickle=False) as shard:
            clip_ids = shard["clip_ids"].astype(str)
            actual_timestamps = shard["actual_timestamps"]
        for clip_id, timestamps in zip(clip_ids, actual_timestamps, strict=True):
            actual_by_clip[clip_id] = timestamps.tolist()
            shard_by_clip[clip_id] = shard_path.name
    metadata = manifest.copy()
    metadata["actual_timestamps"] = metadata["clip_id"].map(actual_by_clip)
    metadata["shard"] = metadata["clip_id"].map(shard_by_clip)
    if metadata["actual_timestamps"].isna().any():
        raise ValueError("Feature metadata is missing extracted clips")
    metadata.to_parquet(config.feature_dir / "metadata.parquet", index=False)


def validate_feature_artifacts(config: ProjectConfig) -> dict[str, float | int]:
    manifest_path = config.manifest_dir / "clips.parquet"
    metadata_path = config.feature_dir / "metadata.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError("Run `latent-drive index` before validating features")
    if not metadata_path.exists():
        raise FileNotFoundError("Run `latent-drive extract` before validating features")

    manifest = pd.read_parquet(manifest_path)
    dimensions: set[int] = set()
    metrics: list[dict[str, float | int]] = []
    for video_id, rows in manifest.groupby("video_id", sort=False):
        shard_path = config.feature_dir / f"{video_id}.npz"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing feature shard: {shard_path}")
        shard_metrics = _validate_existing_shard(shard_path, rows.reset_index(drop=True))
        dimensions.add(int(shard_metrics["embedding_dimension"]))
        metrics.append(shard_metrics)
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions across shards: {sorted(dimensions)}")

    metadata = pd.read_parquet(metadata_path)
    if metadata["clip_id"].astype(str).tolist() != manifest["clip_id"].astype(str).tolist():
        raise ValueError("Feature metadata does not match the clip manifest")
    expected_shards = manifest["video_id"].map(lambda video_id: f"{video_id}.npz")
    if metadata["shard"].astype(str).tolist() != expected_shards.tolist():
        raise ValueError("Feature metadata points to unexpected shards")

    return {
        "video_count": len(metrics),
        "clip_count": sum(int(item["clip_count"]) for item in metrics),
        "embedding_dimension": dimensions.pop(),
        "maximum_timestamp_error_seconds": max(
            float(item["maximum_timestamp_error_seconds"]) for item in metrics
        ),
        "minimum_embedding_norm": min(float(item["minimum_embedding_norm"]) for item in metrics),
        "maximum_embedding_norm": max(float(item["maximum_embedding_norm"]) for item in metrics),
    }
