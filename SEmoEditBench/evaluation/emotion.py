from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    INTENSITY_STRENGTHS,
    cosine_similarity,
    edit_path,
    intensity_edit_path,
    load_config,
    manifest_path,
    read_jsonl,
    track_progress,
    write_jsonl,
)


DEFAULT_MODEL = "emotion2vec/emotion2vec_plus_large"
LABEL_ALIASES = {
    "生气": "angry",
    "angry": "angry",
    "anger": "angry",
    "厌恶": "disgust",
    "disgusted": "disgust",
    "disgust": "disgust",
    "恐惧": "fear",
    "fearful": "fear",
    "fear": "fear",
    "开心": "happy",
    "happy": "happy",
    "joy": "happy",
    "中立": "neutral",
    "neutral": "neutral",
    "其他": "other",
    "other": "other",
    "难过": "sad",
    "sad": "sad",
    "吃惊": "surprise",
    "surprised": "surprise",
    "surprise": "surprise",
    "<unk>": "unknown",
    "unknown": "unknown",
}


def build_model(model: str, device: str, hub: str):
    """Load the Emotion2Vec classifier and embedding model."""
    from funasr import AutoModel

    return AutoModel(model=model, device=device, hub=hub, disable_update=True)


def canonical_label(label: str) -> str:
    """Map model emotion labels to benchmark labels."""
    normalized = label.strip().lower().replace("_", " ")
    for candidate in (normalized, *(part.strip() for part in normalized.split("/"))):
        if candidate in LABEL_ALIASES:
            return LABEL_ALIASES[candidate]
    return normalized


def parse_result(result: Mapping[str, Any]) -> tuple[str, dict[str, float], np.ndarray]:
    """Extract canonical labels, posteriors, and embeddings."""
    labels = result.get("labels")
    scores = result.get("scores")
    features = result.get("feats")
    if not labels or scores is None or features is None:
        raise ValueError("emotion2vec result must contain labels, scores, and feats")
    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.ndim > 1:
        score_array = score_array.mean(axis=0)
    if len(labels) != score_array.size:
        raise ValueError("emotion2vec returned different numbers of labels and scores")
    probabilities: dict[str, float] = {}
    for label, score in zip(labels, score_array, strict=True):
        normalized = canonical_label(str(label))
        probabilities[normalized] = max(
            probabilities.get(normalized, 0.0), float(score)
        )
    embedding = np.asarray(features, dtype=np.float32)
    if embedding.ndim > 1:
        embedding = embedding.mean(axis=0)
    predicted = max(probabilities, key=probabilities.get)
    return predicted, probabilities, embedding


def infer_audio(generate, audio_path: Path):
    """Run emotion recognition on one audio file."""
    generated = generate(
        str(audio_path), granularity="utterance", extract_embedding=True
    )
    if not generated:
        raise ValueError(f"emotion2vec returned no result for: {audio_path}")
    result = generated[0] if isinstance(generated, list) else generated
    return parse_result(result)


def cache_path(directory: Path, audio_path: Path, model_name: str, hub: str) -> Path:
    """Choose a cache filename for an audio emotion prediction."""
    digest = hashlib.sha256(f"{model_name}:{hub}:{audio_path}".encode()).hexdigest()[
        :12
    ]
    return directory / f"{audio_path.stem}-{digest}.npz"


def load_prediction(path: Path) -> tuple[str, dict[str, float], np.ndarray]:
    """Read a cached emotion posterior and embedding."""
    with np.load(path, allow_pickle=False) as cached:
        labels = cached["labels"].tolist()
        probabilities = cached["probabilities"].tolist()
        return (
            str(cached["predicted"].item()),
            dict(zip(labels, probabilities, strict=True)),
            cached["embedding"].copy(),
        )


def save_prediction(
    path: Path, prediction: tuple[str, dict[str, float], np.ndarray]
) -> None:
    """Cache an emotion posterior and embedding on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        predicted=prediction[0],
        labels=list(prediction[1]),
        probabilities=list(prediction[1].values()),
        embedding=prediction[2],
    )


def sign(value: float) -> int:
    """Return the sign of a scalar value."""
    return 1 if value > 0 else -1 if value < 0 else 0


def require_probability(
    prediction: tuple[str, dict[str, float], np.ndarray],
    emotion: str,
    role: str,
    case_id: str,
) -> float:
    """Get a required emotion probability or raise a case error."""
    try:
        return prediction[1][emotion]
    except KeyError as error:
        available = ", ".join(sorted(prediction[1])) or "(none)"
        raise ValueError(
            f"{case_id}: Emotion2Vec {role} result has no probability for {emotion!r}; available labels: {available}"
        ) from error


def calculate_record(record: dict[str, Any], predictions: dict[str, Any]):
    """Compute emotion metrics from one case's model predictions."""
    source = predictions["source"]
    edited = predictions["edited"]
    source_emotion = canonical_label(record.get("source_emotion") or record["emotion"])
    requested_emotion = canonical_label(
        record.get("target_emotion") or record.get("emotion")
    )
    if record["task"] == "intensity":
        source_probability = require_probability(
            source, source_emotion, "source", record["id"]
        )
        probabilities = {
            f"{strength:g}": require_probability(
                predictions[f"{strength:g}"],
                requested_emotion,
                f"{strength:g}",
                record["id"],
            )
            for strength in INTENSITY_STRENGTHS
        }
        edited = predictions["1"]
        edited_probability = probabilities["1"]
        target = predictions.get("target")
    else:
        source_probability = require_probability(
            source, source_emotion, "source", record["id"]
        )
        edited_probability = require_probability(
            edited, requested_emotion, "edited", record["id"]
        )
    result: dict[str, Any] = {
        "id": record["id"],
        "predicted_source_emotion": source[0],
        "predicted_edit_emotion": edited[0],
        "source_emotion_probability": source_probability,
        "target_emotion_probability": edited_probability,
        "source_emotion_suppression": source_probability
        - require_probability(edited, source_emotion, "edited", record["id"]),
    }
    if record["task"] == "erasure":
        result["neutral_probability"] = require_probability(
            edited, "neutral", "edited", record["id"]
        )
    if record["task"] == "intensity":
        result["source_intensity_probability"] = require_probability(
            source, requested_emotion, "source", record["id"]
        )
        result["edited_intensity_probability"] = edited_probability
        result["intensity_probabilities"] = probabilities
        values = list(probabilities.values())
        inversions = sum(
            first > second
            for index, first in enumerate(values)
            for second in values[index + 1 :]
        )
        result["intensity_monotonicity"] = 1.0 - inversions / (
            len(values) * (len(values) - 1) / 2
        )
        if target is not None:
            source_embedding = source[2]
            target_delta = target[2] - source_embedding
            denominator = float(np.dot(target_delta, target_delta))
            # Project each edit onto the source-to-target emotion direction.
            result["reference_progress"] = {
                f"{strength:g}": float(
                    np.dot(
                        predictions[f"{strength:g}"][2] - source_embedding, target_delta
                    )
                    / denominator
                )
                for strength in INTENSITY_STRENGTHS
            }
            progress = list(result["reference_progress"].values())
            result["reference_progress_range"] = progress[-1] - progress[0]
    target = predictions.get("target")
    if target is not None and record["task"] != "intensity":
        result["emotion_similarity"] = cosine_similarity(edited[2], target[2])
        edited_delta = edited[2] - source[2]
        target_delta = target[2] - source[2]
        if float(np.linalg.norm(target_delta)) == 0.0:
            raise ValueError(f"{record['id']}: target emotion delta is zero")
        result["directional_editing_score"] = (
            0.0
            if float(np.linalg.norm(edited_delta)) == 0.0
            else cosine_similarity(edited_delta, target_delta)
        )
    else:
        result["emotion_similarity"] = None
        result["directional_editing_score"] = None
    return result


def evaluate(
    manifest_path_value: Path,
    config_path: Path,
    output_root: Path,
    model_name: str,
    device: str,
    hub: str,
    embedding_dir: Path,
    model: Any | None = None,
    show_progress: bool = True,
    reuse_predictions: bool = True,
    audio_cache: dict | None = None,
):
    """Compute emotion evaluation metrics for one benchmark split."""
    config = load_config(config_path)
    records = read_jsonl(manifest_path_value)
    split = manifest_path_value.stem
    model = model or build_model(model_name, device, hub)
    if audio_cache is None:
        audio_cache = {}

    def get_prediction(path: Path):
        """Reuse a cached emotion prediction or infer a new one."""
        if path not in audio_cache:
            saved_path = cache_path(embedding_dir, path, model_name, hub)
            if reuse_predictions and saved_path.is_file():
                prediction = load_prediction(saved_path)
            else:
                prediction = infer_audio(model.generate, path)
                save_prediction(saved_path, prediction)
            audio_cache[path] = prediction
        return audio_cache[path]

    results = []
    for record in track_progress(
        records, total=len(records), desc=f"Emotion {split}", enabled=show_progress
    ):
        paths = {
            "source": manifest_path(record, config, "source_audio"),
            "edited": edit_path(output_root, split, record["id"]),
        }
        if record["task"] == "intensity":
            paths.update(
                {
                    f"{strength:g}": intensity_edit_path(
                        output_root, split, record["id"], strength
                    )
                    for strength in INTENSITY_STRENGTHS
                }
            )
        if record.get("target_audio") is not None:
            paths["target"] = manifest_path(record, config, "target_audio")
        results.append(
            calculate_record(
                record, {role: get_prediction(path) for role, path in paths.items()}
            )
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for emotion evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate emotion probabilities and Emotion2Vec metrics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--embedding-dir", type=Path, default=Path("results/embeddings")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hub", choices=("hf", "ms"), default="hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Ignore cached Emotion2Vec predictions"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run emotion evaluation from command-line arguments."""
    args = parse_args(argv)
    write_jsonl(
        args.output,
        evaluate(
            args.manifest,
            args.config,
            args.output_root,
            args.model,
            args.device,
            args.hub,
            args.embedding_dir,
            show_progress=not args.no_progress,
            reuse_predictions=not args.force,
        ),
    )


if __name__ == "__main__":
    main()
