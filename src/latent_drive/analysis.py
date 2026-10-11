from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from latent_drive.config import ProjectConfig
from latent_drive.features import validate_feature_artifacts


def _load_embeddings(config: ProjectConfig) -> tuple[pd.DataFrame, np.ndarray]:
    validate_feature_artifacts(config)
    metadata = pd.read_parquet(config.feature_dir / "metadata.parquet")
    clip_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for shard_name in metadata["shard"].drop_duplicates():
        with np.load(config.feature_dir / shard_name, allow_pickle=False) as shard:
            clip_ids.extend(shard["clip_ids"].astype(str).tolist())
            embeddings.append(shard["normalized_embeddings"].astype(np.float32))
    if clip_ids != metadata["clip_id"].astype(str).tolist():
        raise ValueError("Feature shard ordering does not match feature metadata")
    return metadata, np.concatenate(embeddings)


def _select_queries(metadata: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    count = min(count, len(metadata))
    rng = np.random.default_rng(seed)
    groups = [group.index.to_numpy() for _, group in metadata.groupby("split", sort=True)]
    selected: list[int] = []
    base, remainder = divmod(count, len(groups))
    for position, indices in enumerate(groups):
        group_count = min(len(indices), base + (position < remainder))
        if group_count:
            selected.extend(rng.choice(indices, size=group_count, replace=False).tolist())
    if len(selected) < count:
        available = np.setdiff1d(
            np.arange(len(metadata)), np.asarray(selected), assume_unique=False
        )
        selected.extend(rng.choice(available, size=count - len(selected), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _neighbor_indices(
    embeddings: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(count, len(embeddings) - 1)
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=count + 1)
    model.fit(embeddings)
    raw_distances, raw_indices = model.kneighbors(embeddings)
    indices = np.empty((len(embeddings), count), dtype=np.int64)
    distances = np.empty((len(embeddings), count), dtype=np.float32)
    for row in range(len(embeddings)):
        keep = raw_indices[row] != row
        indices[row] = raw_indices[row][keep][:count]
        distances[row] = raw_distances[row][keep][:count]
    return indices, distances


def _sample_pair_similarities(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    sample_count: int,
    seed: int,
    clip_duration: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, float]] = []
    video_groups = {
        str(name): group.sort_values("start_seconds").index.to_numpy()
        for name, group in metadata.groupby("video_id", sort=True)
    }
    session_groups = {
        str(name): group.index.to_numpy()
        for name, group in metadata.groupby("session_id", sort=True)
    }
    session_videos = {
        str(name): [indices for _, indices in group.groupby("video_id", sort=True).groups.items()]
        for name, group in metadata.groupby("session_id", sort=True)
    }

    for indices in video_groups.values():
        if len(indices) < 2:
            continue
        for left, right in zip(indices, indices[1:], strict=False):
            similarity = float(np.dot(embeddings[left], embeddings[right]))
            samples.append(("adjacent", similarity))

    nonlocal_gap = max(5.0, 5.0 * clip_duration)
    eligible_videos = [indices for indices in video_groups.values() if len(indices) >= 2]
    attempts = 0
    collected = 0
    while eligible_videos and collected < sample_count and attempts < sample_count * 50:
        attempts += 1
        indices = eligible_videos[int(rng.integers(len(eligible_videos)))]
        left, right = rng.choice(indices, size=2, replace=False)
        starts = metadata.loc[[left, right], "start_seconds"].to_numpy()
        if abs(float(starts[0] - starts[1])) < nonlocal_gap:
            continue
        samples.append(("same_video_nonlocal", float(np.dot(embeddings[left], embeddings[right]))))
        collected += 1

    eligible_sessions = [videos for videos in session_videos.values() if len(videos) >= 2]
    for _ in range(sample_count if eligible_sessions else 0):
        videos = eligible_sessions[int(rng.integers(len(eligible_sessions)))]
        first_video, second_video = rng.choice(len(videos), size=2, replace=False)
        left = int(rng.choice(np.asarray(videos[first_video])))
        right = int(rng.choice(np.asarray(videos[second_video])))
        samples.append(
            ("same_session_cross_video", float(np.dot(embeddings[left], embeddings[right])))
        )

    sessions = list(session_groups.values())
    for _ in range(sample_count if len(sessions) >= 2 else 0):
        first_session, second_session = rng.choice(len(sessions), size=2, replace=False)
        left = int(rng.choice(sessions[first_session]))
        right = int(rng.choice(sessions[second_session]))
        samples.append(("cross_session", float(np.dot(embeddings[left], embeddings[right]))))

    return pd.DataFrame(samples, columns=["category", "cosine_similarity"])


def _centroid_distances(metadata: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_type in ("split", "session_id", "video_id"):
        centroids: dict[str, np.ndarray] = {}
        for name, group in metadata.groupby(group_type, sort=True):
            centroid = embeddings[group.index].mean(axis=0)
            centroids[str(name)] = centroid / np.linalg.vector_norm(centroid)
        for (left_name, left), (right_name, right) in combinations(centroids.items(), 2):
            similarity = float(np.dot(left, right))
            rows.append(
                {
                    "group_type": group_type,
                    "left": left_name,
                    "right": right_name,
                    "cosine_similarity": similarity,
                    "cosine_distance": 1.0 - similarity,
                }
            )
    return pd.DataFrame(rows)


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.1)),
        "p90": float(values.quantile(0.9)),
    }


def analyze_embeddings(config: ProjectConfig) -> Path:
    metadata, embeddings = _load_embeddings(config)
    if len(metadata) < 3:
        raise ValueError("Representation analysis requires at least three clips")
    config.analysis_dir.mkdir(parents=True, exist_ok=True)

    component_count = min(
        config.analysis.pca_components,
        len(embeddings) - 1,
        embeddings.shape[1],
    )
    solver = "randomized" if component_count < min(embeddings.shape) else "full"
    pca = PCA(
        n_components=component_count,
        svd_solver=solver,
        random_state=config.analysis.seed,
    )
    coordinates = pca.fit_transform(embeddings)
    pca_table = metadata[["clip_id", "video_id", "session_id", "split", "start_seconds"]].copy()
    pca_table["pc1"] = coordinates[:, 0]
    pca_table["pc2"] = coordinates[:, 1]
    pca_table.to_parquet(config.analysis_dir / "pca_coordinates.parquet", index=False)

    maximum_neighbors = max(config.analysis.retrieval_neighbors, config.analysis.outlier_neighbors)
    neighbors, distances = _neighbor_indices(embeddings, maximum_neighbors)
    nearest = neighbors[:, 0]
    same_video = metadata.iloc[nearest]["video_id"].to_numpy() == metadata["video_id"].to_numpy()
    same_session = (
        metadata.iloc[nearest]["session_id"].astype(str).to_numpy()
        == metadata["session_id"].astype(str).to_numpy()
    )
    same_split = metadata.iloc[nearest]["split"].to_numpy() == metadata["split"].to_numpy()
    start_delta = np.abs(
        metadata.iloc[nearest]["start_seconds"].to_numpy() - metadata["start_seconds"].to_numpy()
    )
    local_temporal = same_video & (start_delta <= 10.0 * config.clips.duration_seconds)

    query_indices = _select_queries(
        metadata,
        config.analysis.retrieval_queries,
        config.analysis.seed,
    )
    retrieval_rows: list[dict[str, Any]] = []
    for query_index in query_indices:
        query = metadata.iloc[query_index]
        for rank, (neighbor_index, distance) in enumerate(
            zip(
                neighbors[query_index, : config.analysis.retrieval_neighbors],
                distances[query_index, : config.analysis.retrieval_neighbors],
                strict=True,
            ),
            start=1,
        ):
            neighbor = metadata.iloc[neighbor_index]
            retrieval_rows.append(
                {
                    "query_clip_id": query["clip_id"],
                    "query_video_id": query["video_id"],
                    "query_session_id": query["session_id"],
                    "query_split": query["split"],
                    "query_start_seconds": float(query["start_seconds"]),
                    "rank": rank,
                    "neighbor_clip_id": neighbor["clip_id"],
                    "neighbor_video_id": neighbor["video_id"],
                    "neighbor_session_id": neighbor["session_id"],
                    "neighbor_split": neighbor["split"],
                    "neighbor_start_seconds": float(neighbor["start_seconds"]),
                    "cosine_similarity": 1.0 - float(distance),
                    "same_video": bool(query["video_id"] == neighbor["video_id"]),
                    "same_session": bool(str(query["session_id"]) == str(neighbor["session_id"])),
                    "same_split": bool(query["split"] == neighbor["split"]),
                }
            )
    pd.DataFrame(retrieval_rows).to_parquet(config.analysis_dir / "retrievals.parquet", index=False)

    outlier_k = min(config.analysis.outlier_neighbors, distances.shape[1])
    outlier_scores = distances[:, :outlier_k].mean(axis=1)
    outlier_indices = np.argsort(outlier_scores)[::-1][: config.analysis.outlier_count]
    outliers = metadata.iloc[outlier_indices][
        ["clip_id", "video_id", "session_id", "split", "start_seconds"]
    ].copy()
    outliers["mean_neighbor_cosine_distance"] = outlier_scores[outlier_indices]
    outliers.to_parquet(config.analysis_dir / "outliers.parquet", index=False)

    similarities = _sample_pair_similarities(
        metadata,
        embeddings,
        config.analysis.pair_samples,
        config.analysis.seed,
        config.clips.duration_seconds,
    )
    similarities.to_parquet(config.analysis_dir / "pair_similarities.parquet", index=False)
    centroid_distances = _centroid_distances(metadata, embeddings)
    centroid_distances.to_parquet(config.analysis_dir / "centroid_distances.parquet", index=False)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pca_variance = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )
    pca_variance.to_parquet(config.analysis_dir / "pca_variance.parquet", index=False)

    similarity_summary = {
        category: _distribution_summary(group["cosine_similarity"])
        for category, group in similarities.groupby("category", sort=True)
    }
    components_for_90 = int(np.searchsorted(cumulative, 0.9) + 1) if cumulative[-1] >= 0.9 else None
    feature_run = config.feature_dir / "run.json"
    summary = {
        "audit_name": config.analysis.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": config.encoder.model_id,
        "model_revision": config.encoder.revision,
        "clip_count": len(metadata),
        "embedding_dimension": embeddings.shape[1],
        "split_counts": {
            str(name): int(count) for name, count in metadata["split"].value_counts().items()
        },
        "video_count": int(metadata["video_id"].nunique()),
        "session_count": int(metadata["session_id"].nunique()),
        "pca": {
            "component_count": component_count,
            "first_two_explained_variance": float(explained[:2].sum()),
            "cumulative_explained_variance": float(cumulative[-1]),
            "components_for_90_percent": components_for_90,
        },
        "nearest_neighbor": {
            "same_video_fraction": float(same_video.mean()),
            "same_session_fraction": float(same_session.mean()),
            "same_split_fraction": float(same_split.mean()),
            "local_temporal_fraction": float(local_temporal.mean()),
            "mean_top1_cosine_similarity": float((1.0 - distances[:, 0]).mean()),
        },
        "pair_similarity": similarity_summary,
        "outlier_score": {
            "neighbor_count": outlier_k,
            "minimum": float(outlier_scores.min()),
            "median": float(np.median(outlier_scores)),
            "maximum": float(outlier_scores.max()),
        },
        "analysis_config": {
            "seed": config.analysis.seed,
            "retrieval_queries": config.analysis.retrieval_queries,
            "retrieval_neighbors": config.analysis.retrieval_neighbors,
            "pca_components": config.analysis.pca_components,
            "pair_samples": config.analysis.pair_samples,
            "outlier_neighbors": config.analysis.outlier_neighbors,
            "outlier_count": config.analysis.outlier_count,
        },
        "feature_run_sha256": (
            hashlib.sha256(feature_run.read_bytes()).hexdigest() if feature_run.exists() else None
        ),
    }
    (config.analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return config.analysis_dir
