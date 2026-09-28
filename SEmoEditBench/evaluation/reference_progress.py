from pathlib import Path

import numpy as np

from .common import (
    INTENSITY_STRENGTHS,
    load_config,
    manifest_path,
    read_jsonl,
    records_for_system,
    write_jsonl,
)
from .emotion import (
    build_model,
    cache_path,
    infer_audio,
    load_prediction,
    save_prediction,
)
from .report import DEFAULT_SPLITS, build_report, discover_systems, write_report


def relative_progress(source, target, edits):
    """Project edit embeddings onto the source-to-target direction."""
    direction = np.asarray(target, dtype=np.float64) - source
    denominator = float(np.dot(direction, direction))
    if denominator == 0:
        raise ValueError(
            "Source and target embeddings coincide; reference direction is undefined"
        )
    return {
        strength: float(
            np.dot(np.asarray(embedding, dtype=np.float64) - source, direction)
            / denominator
        )
        for strength, embedding in edits.items()
    }


def run(args):
    """Compute source-to-target embedding progress for a result set."""
    config = load_config(args.config)
    systems = discover_systems(args.results_root)
    cache = args.results_root / "_cache" / "embeddings"
    model = None
    predictions = {}

    def embedding(path, model_name, hub):
        """Load or compute an emotion embedding for audio."""
        nonlocal model
        key = (path, model_name, hub)
        if key not in predictions:
            saved = cache_path(cache, path, model_name, hub)
            if saved.is_file():
                prediction = load_prediction(saved)
            else:
                if model is None:
                    model = build_model(model_name, args.device, hub)
                prediction = infer_audio(model.generate, path)
                save_prediction(saved, prediction)
            predictions[key] = prediction[2]
        return predictions[key]

    settings = {
        system: read_jsonl(root / "evaluation.jsonl")[0]
        for system, root in systems.items()
    }
    model_settings = {
        (value["emotion_model"], value["emotion_hub"]) for value in settings.values()
    }
    if len(model_settings) != 1:
        raise ValueError(
            "Reference progress requires the same Emotion2Vec model and hub for every system"
        )
    model_name, hub = model_settings.pop()
    for split in DEFAULT_SPLITS:
        if not split.startswith("intensity_"):
            continue
        manifest = read_jsonl(args.manifest_dir / f"{split}.jsonl")
        supported = {
            system: root
            for system, root in systems.items()
            if (root / split / "summary.jsonl").exists()
            and read_jsonl(root / split / "summary.jsonl")[0].get("status")
            != "unsupported"
        }
        if not supported:
            continue
        common = {
            record["id"]
            for record in manifest
            if record["source_emotion"] == "neutral" and record.get("target_audio")
        }
        for system in supported:
            common &= {
                record["id"]
                for record in records_for_system(
                    system, settings[system]["manifests"][split]
                )
            }
        records = [record for record in manifest if record["id"] in common]
        if not records:
            raise ValueError(
                f"{split}: no common neutral-source cases with paired targets"
            )
        write_jsonl(
            args.results_root / "_manifests" / f"reference_progress_{split}.jsonl",
            records,
        )
        print(
            f"{split}: {len(records)} identical cases for {len(supported)} systems",
            flush=True,
        )
        for system, root in supported.items():
            path = root / split / "intensity.jsonl"
            cases = read_jsonl(path)
            by_id = {record["id"]: record for record in cases}
            for case in cases:
                case["reference_progress"] = None
                case["reference_progress_range"] = None
            for record in records:
                edits = {
                    f"{strength:g}": embedding(
                        (
                            Path(settings[system]["output_root"])
                            / split
                            / record["id"]
                            / f"{strength:g}.wav"
                        ).resolve(),
                        model_name,
                        hub,
                    )
                    for strength in INTENSITY_STRENGTHS
                }
                progress = relative_progress(
                    embedding(
                        manifest_path(record, config, "source_audio"), model_name, hub
                    ),
                    embedding(
                        manifest_path(record, config, "target_audio"), model_name, hub
                    ),
                    edits,
                )
                by_id[record["id"]].update(
                    reference_progress=progress,
                    reference_progress_range=progress["1"] - progress["0"],
                )
            write_jsonl(path, cases)
            summary_path = root / split / "summary.jsonl"
            (summary,) = read_jsonl(summary_path)
            selected = [by_id[record["id"]] for record in records]
            for group, values in [(summary, selected)] + [
                (
                    summary[label],
                    [case for case in selected if case["target_intensity"] == label],
                )
                for label in {case["target_intensity"] for case in cases}
            ]:
                group["reference_progress_cases"] = len(values)
                group["reference_progress_range"] = (
                    sum(case["reference_progress_range"] for case in values)
                    / len(values)
                    if values
                    else None
                )
            summary["reference_progress_systems"] = sorted(supported)
            write_jsonl(summary_path, [summary])
    rows, directions = build_report(args.results_root, args.manifest_dir)
    write_report(args.results_root / "comparison_report", rows, directions)
    return 0
