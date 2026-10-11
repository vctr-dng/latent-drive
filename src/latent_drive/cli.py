from __future__ import annotations

import argparse
from pathlib import Path

from latent_drive.config import load_config
from latent_drive.data import build_clip_index, inventory_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent-drive",
        description="Build frozen V-JEPA representations from passive dashcam footage.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/representation_audit.toml"),
        help="Project configuration TOML file",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory", help="Probe source videos and write an inventory")
    commands.add_parser("index", help="Build the leakage-safe one-second clip manifest")
    extract = commands.add_parser("extract", help="Extract frozen V-JEPA clip embeddings")
    extract.add_argument("--limit", type=int, help="Extract an isolated smoke shard")
    extract.add_argument("--force", action="store_true", help="Replace completed video shards")
    commands.add_parser("validate", help="Validate feature shards against the clip manifest")
    commands.add_parser("analyze", help="Run the deterministic representation audit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "inventory":
        inventory = inventory_videos(config)
        print(
            f"Indexed {len(inventory)} videos, "
            f"{inventory['duration_seconds'].sum() / 60:.2f} minutes"
        )
    elif args.command == "index":
        manifest = build_clip_index(config)
        counts = manifest.groupby("split").size().to_dict()
        print(f"Indexed {len(manifest)} clips: {counts}")
    elif args.command == "extract":
        from latent_drive.features import extract_features

        output = extract_features(config, limit=args.limit, force=args.force)
        print(f"Features written to {output}")
    elif args.command == "validate":
        from latent_drive.features import validate_feature_artifacts

        metrics = validate_feature_artifacts(config)
        print(
            f"Validated {metrics['clip_count']} clips across {metrics['video_count']} videos; "
            f"dimension={metrics['embedding_dimension']}, "
            f"maximum timestamp error={metrics['maximum_timestamp_error_seconds']:.6f}s"
        )
    elif args.command == "analyze":
        from latent_drive.analysis import analyze_embeddings

        output = analyze_embeddings(config)
        print(f"Representation audit written to {output}")
