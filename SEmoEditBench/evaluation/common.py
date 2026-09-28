from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar


ProgressItem = TypeVar("ProgressItem")


INTENSITY_COMMON_EMOTIONS = {"angry", "happy", "sad", "surprise"}


def records_for_system(
    system: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply each baseline's supported tasks and emotion-label coverage."""
    if system.startswith("emosteer_") and system.endswith(
        "_activation_steering_600cases"
    ):
        return [
            record
            for record in records
            if (
                record["task"] == "replacement"
                and record["target_emotion"] in INTENSITY_COMMON_EMOTIONS
            )
            or (
                record["task"] == "intensity"
                and record["source_emotion"] == "neutral"
                and record["target_emotion"] in INTENSITY_COMMON_EMOTIONS
            )
            or (
                record["task"] == "erasure"
                and record["target_emotion"] == "neutral"
                and record["source_emotion"] in INTENSITY_COMMON_EMOTIONS
            )
        ]
    if system not in {
        "cocoemo_cosyvoice2_baseline_activation_steering_600cases",
        "cocoemo_indextts2_baseline_activation_steering_600cases",
    }:
        return records
    return [
        record
        for record in records
        if (
            record["task"] == "replacement"
            and record["target_emotion"] in INTENSITY_COMMON_EMOTIONS
        )
        or (
            record["task"] == "intensity"
            and record["source_emotion"] == "neutral"
            and record["target_emotion"] in INTENSITY_COMMON_EMOTIONS
        )
    ]


def common_intensity_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Intensity cases shared by every system: neutral source and the four activation-steering emotions."""
    return [
        record
        for record in records
        if record.get("source_emotion") == "neutral"
        and record.get("target_emotion") in INTENSITY_COMMON_EMOTIONS
    ]


def load_config(path: Path) -> dict[str, Any]:
    """Load configuration and resolve relative dataset paths."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config_dir = path.resolve().parent
    for section in ("dataset_roots",):
        for name, value in config.get(section, {}).items():
            root = Path(value).expanduser()
            if not root.is_absolute():
                root = config_dir / root
            config[section][name] = str(root.resolve())
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON Lines file."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def track_progress(
    iterable: Iterable[ProgressItem], *, total: int, desc: str, enabled: bool = True
) -> Iterable[ProgressItem]:
    """Wrap case iteration with an optional progress bar."""
    from tqdm.auto import tqdm

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="case",
        dynamic_ncols=True,
        disable=None if enabled and total else True,
    )


def manifest_path(manifest: dict[str, Any], config: dict[str, Any], field: str) -> Path:
    """Resolve a manifest audio field using the dataset config."""
    value = manifest.get(field)
    if value is None:
        raise ValueError(f"{manifest['id']}: {field} is null")
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        try:
            root = Path(config["dataset_roots"][manifest["dataset"]])
        except KeyError as error:
            raise KeyError(
                f"Missing dataset root for {manifest.get('dataset')!r}"
            ) from error
        resolved = (root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{manifest['id']}: audio file does not exist: {resolved}"
        )
    return resolved


def language_for(manifest: dict[str, Any]) -> str:
    """Choose the ASR evaluation language for a case."""
    if manifest.get("language") == "zh":
        return "chinese"
    return "english"


def edit_path(output_root: Path, split: str, case_id: str) -> Path:
    """Locate a system's edited waveform for one case."""
    path = (output_root / split / f"{case_id}.wav").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing edited audio for {case_id}: {path}")
    return path


INTENSITY_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def intensity_edit_path(
    output_root: Path, split: str, case_id: str, strength: float
) -> Path:
    """Locate one strength-level waveform for an intensity case."""
    path = (output_root / split / case_id / f"{strength:g}.wav").resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing intensity edit for {case_id} at strength {strength}: {path}"
        )
    return path


def normalize_text(text: str, language: str) -> str:
    """Normalize English words or Mandarin characters for ASR scoring."""
    text = unicodedata.normalize("NFKC", text).strip()
    if language == "chinese":
        characters = []
        for character in text:
            category = unicodedata.category(character)
            if character.isspace() or category.startswith(("P", "S")):
                continue
            characters.append(character)
        return " ".join(characters)
    characters = [
        character.lower() if not character.isspace() else " " for character in text
    ]
    normalized = "".join(characters)
    return " ".join(normalized.split())


def cosine_similarity(first: Any, second: Any) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import numpy as np

    first_array = np.asarray(first, dtype=np.float32).reshape(-1)
    second_array = np.asarray(second, dtype=np.float32).reshape(-1)
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"Embedding dimensions differ: {first_array.shape} vs {second_array.shape}"
        )
    first_norm = float(np.linalg.norm(first_array))
    second_norm = float(np.linalg.norm(second_array))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Cannot calculate cosine similarity for a zero vector")
    return float(np.dot(first_array, second_array) / (first_norm * second_norm))


def macro_mean(values: list[float]) -> float | None:
    """Average available per-case metric values."""
    return sum(values) / len(values) if values else None
