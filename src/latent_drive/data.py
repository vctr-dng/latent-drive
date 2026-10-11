from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from latent_drive.config import ProjectConfig

VIDEO_PATTERN = re.compile(r"DJI_(?P<session>\d{4})_(?P<segment>\d{3})$")


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:"
            "format=duration,size:format_tags=creation_time"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    media_format = payload["format"]
    match = VIDEO_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected video filename: {path.name}")

    fps = Fraction(stream["avg_frame_rate"])
    return {
        "video_id": path.stem,
        "session_id": match.group("session"),
        "segment_index": int(match.group("segment")),
        "path": str(path.resolve()),
        "creation_time": media_format.get("tags", {}).get("creation_time"),
        "duration_seconds": float(media_format["duration"]),
        "fps_numerator": fps.numerator,
        "fps_denominator": fps.denominator,
        "frame_count": int(stream.get("nb_frames") or round(float(media_format["duration"]) * fps)),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "size_bytes": int(media_format["size"]),
    }


def inventory_videos(config: ProjectConfig) -> pd.DataFrame:
    paths = sorted(config.paths.data_dir.glob("*.MP4"))
    if not paths:
        raise FileNotFoundError(f"No MP4 videos found under {config.paths.data_dir}")

    inventory = pd.DataFrame(probe_video(path) for path in paths).sort_values(
        ["session_id", "segment_index"]
    )
    inventory = inventory.reset_index(drop=True)
    _validate_inventory(inventory)

    config.inventory_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(config.inventory_dir / "videos.parquet", index=False)
    summary = {
        "video_count": len(inventory),
        "duration_seconds": float(inventory["duration_seconds"].sum()),
        "size_bytes": int(inventory["size_bytes"].sum()),
        "sessions": sorted(inventory["session_id"].unique().tolist()),
        "segment_gaps": _segment_gaps(inventory),
    }
    (config.inventory_dir / "inventory.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return inventory


def _validate_inventory(inventory: pd.DataFrame) -> None:
    if inventory["video_id"].duplicated().any():
        raise ValueError("Duplicate video IDs found")
    if set(inventory["codec"]) != {"h264"}:
        raise ValueError(f"Expected H.264 footage, found {sorted(set(inventory['codec']))}")
    resolutions = set(zip(inventory["width"], inventory["height"], strict=True))
    if resolutions != {(1920, 1080)}:
        raise ValueError(f"Unexpected video resolutions: {sorted(resolutions)}")
    for session_id, session in inventory.groupby("session_id", sort=True):
        indices = session["segment_index"].sort_values().tolist()
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            raise ValueError(f"Missing segment in session {session_id}: found {indices}")


def _segment_gaps(inventory: pd.DataFrame) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for session_id, session in inventory.groupby("session_id", sort=True):
        ordered = session.sort_values("segment_index").to_dict("records")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if not previous["creation_time"] or not current["creation_time"]:
                continue
            previous_start = datetime.fromisoformat(
                previous["creation_time"].replace("Z", "+00:00")
            )
            expected_next = previous_start + timedelta(seconds=previous["duration_seconds"])
            current_start = datetime.fromisoformat(current["creation_time"].replace("Z", "+00:00"))
            gaps.append(
                {
                    "session_id": session_id,
                    "from_video": previous["video_id"],
                    "to_video": current["video_id"],
                    "gap_seconds": (current_start - expected_next).total_seconds(),
                }
            )
    return gaps


def sample_timestamps(
    start_seconds: float,
    duration_seconds: float,
    sampling_fps: float,
) -> list[float]:
    frame_count = int(round(duration_seconds * sampling_fps))
    return [start_seconds + (index + 0.5) / sampling_fps for index in range(frame_count)]


def make_clip_manifest(inventory: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video in inventory.to_dict("records"):
        split = config.splits.split_for(video["video_id"])
        clip_count = math.floor(video["duration_seconds"] / config.clips.duration_seconds + 1e-9)
        for index in range(clip_count):
            start = index * config.clips.duration_seconds
            end = start + config.clips.duration_seconds
            rows.append(
                {
                    "clip_id": f"{video['video_id']}__{round(start * 1000):09d}",
                    "video_id": video["video_id"],
                    "session_id": video["session_id"],
                    "split": split,
                    "path": video["path"],
                    "fps_numerator": int(video["fps_numerator"]),
                    "fps_denominator": int(video["fps_denominator"]),
                    "start_seconds": start,
                    "end_seconds": end,
                    "requested_timestamps": sample_timestamps(
                        start,
                        config.clips.duration_seconds,
                        config.clips.sampling_fps,
                    ),
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No complete clips could be indexed")
    if manifest["clip_id"].duplicated().any():
        raise ValueError("Duplicate clip IDs generated")
    if (
        manifest["end_seconds"]
        > manifest["path"].map(inventory.set_index("path")["duration_seconds"])
    ).any():
        raise ValueError("A clip extends beyond its source video")
    return manifest


def build_clip_index(config: ProjectConfig) -> pd.DataFrame:
    inventory_path = config.inventory_dir / "videos.parquet"
    if not inventory_path.exists():
        raise FileNotFoundError("Run `latent-drive inventory` before building the clip index")
    inventory = pd.read_parquet(inventory_path)
    manifest = make_clip_manifest(inventory, config)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(config.manifest_dir / "clips.parquet", index=False)
    return manifest
