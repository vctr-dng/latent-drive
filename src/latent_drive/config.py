from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    data_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class ClipsConfig:
    duration_seconds: float
    sampling_fps: float


@dataclass(frozen=True)
class SplitsConfig:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def split_for(self, video_id: str) -> str:
        matches = [
            name
            for name, values in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
            if video_id in values
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one split for {video_id!r}, found {matches or 'none'}"
            )
        return matches[0]


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str
    revision: str
    device: str
    decode_device: str
    dtype: str
    attention: str
    batch_size: int


@dataclass(frozen=True)
class PreprocessConfig:
    mode: str
    size: int


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int
    retrieval_queries: int
    retrieval_neighbors: int
    name: str = "representation_audit"
    pca_components: int = 50
    pair_samples: int = 5000
    outlier_neighbors: int = 10
    outlier_count: int = 20


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    clips: ClipsConfig
    splits: SplitsConfig
    encoder: EncoderConfig
    preprocess: PreprocessConfig
    analysis: AnalysisConfig

    @property
    def inventory_dir(self) -> Path:
        return self.paths.artifacts_dir / "inventory"

    @property
    def manifest_dir(self) -> Path:
        return self.paths.artifacts_dir / "manifests"

    @property
    def model_slug(self) -> str:
        return self.encoder.model_id.replace("/", "__")

    @property
    def feature_dir(self) -> Path:
        return self.paths.artifacts_dir / "features" / self.model_slug

    @property
    def analysis_dir(self) -> Path:
        return self.paths.artifacts_dir / "analysis" / self.model_slug / self.analysis.name


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    root = config_path.parent.parent
    paths = raw["paths"]
    clips = raw["clips"]
    splits = raw["splits"]
    encoder = raw["encoder"]
    preprocess = raw["preprocess"]
    analysis = raw["analysis"]

    config = ProjectConfig(
        paths=PathsConfig(
            root=root,
            data_dir=(root / paths["data_dir"]).resolve(),
            artifacts_dir=(root / paths["artifacts_dir"]).resolve(),
        ),
        clips=ClipsConfig(
            duration_seconds=float(clips["duration_seconds"]),
            sampling_fps=float(clips["sampling_fps"]),
        ),
        splits=SplitsConfig(
            train=tuple(splits["train"]),
            validation=tuple(splits["validation"]),
            test=tuple(splits["test"]),
        ),
        encoder=EncoderConfig(
            model_id=encoder["model_id"],
            revision=encoder["revision"],
            device=encoder["device"],
            decode_device=encoder["decode_device"],
            dtype=encoder["dtype"],
            attention=encoder["attention"],
            batch_size=int(encoder["batch_size"]),
        ),
        preprocess=PreprocessConfig(
            mode=preprocess["mode"],
            size=int(preprocess["size"]),
        ),
        analysis=AnalysisConfig(
            seed=int(analysis["seed"]),
            retrieval_queries=int(analysis["retrieval_queries"]),
            retrieval_neighbors=int(analysis["retrieval_neighbors"]),
            name=analysis.get("name", "representation_audit"),
            pca_components=int(analysis.get("pca_components", 50)),
            pair_samples=int(analysis.get("pair_samples", 5000)),
            outlier_neighbors=int(analysis.get("outlier_neighbors", 10)),
            outlier_count=int(analysis.get("outlier_count", 20)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: ProjectConfig) -> None:
    if config.clips.duration_seconds <= 0 or config.clips.sampling_fps <= 0:
        raise ValueError("Clip duration and sampling FPS must be positive")
    frame_count = config.clips.duration_seconds * config.clips.sampling_fps
    if not frame_count.is_integer():
        raise ValueError("duration_seconds * sampling_fps must be an integer")
    if int(frame_count) < 2 or int(frame_count) % 2:
        raise ValueError("V-JEPA clips need a positive even number of frames")
    if config.encoder.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported encoder dtype: {config.encoder.dtype}")
    if config.preprocess.mode != "letterbox":
        raise ValueError("Only aspect-preserving letterbox preprocessing is supported")
    if config.preprocess.size % 16:
        raise ValueError("Preprocessing size must be divisible by the V-JEPA patch size (16)")
    if not config.analysis.name or not config.analysis.name.replace("_", "").isalnum():
        raise ValueError("Analysis name must contain only letters, numbers, and underscores")
    analysis_counts = (
        config.analysis.retrieval_queries,
        config.analysis.retrieval_neighbors,
        config.analysis.pca_components,
        config.analysis.pair_samples,
        config.analysis.outlier_neighbors,
        config.analysis.outlier_count,
    )
    if any(value <= 0 for value in analysis_counts):
        raise ValueError("Analysis counts must be positive")

    assigned = config.splits.train + config.splits.validation + config.splits.test
    if len(assigned) != len(set(assigned)):
        raise ValueError("A video ID appears in more than one split")
