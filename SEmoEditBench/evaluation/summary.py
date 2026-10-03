from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .common import common_intensity_records, macro_mean, read_jsonl, write_jsonl


def merge_results(results_root: Path, split: str, manifest: list[dict[str, Any]]):
    """Join metric files by benchmark case ID."""
    sources = {}
    for name in ("wer", "emotion", "speaker", "quality", "intensity"):
        path = results_root / split / f"{name}.jsonl"
        if path.exists():
            sources[name] = {record["id"]: record for record in read_jsonl(path)}
    merged = []
    for record in manifest:
        case_id = record["id"]
        merged_record = {"id": case_id}
        for values in sources.values():
            if case_id in values:
                merged_record.update(values[case_id])
        if "target_intensity" in record:
            merged_record["target_intensity"] = record["target_intensity"]
        merged.append(merged_record)
    return merged


def aggregate(
    records: Iterable[dict[str, Any]], task: str, include_directions: bool = True
) -> dict[str, Any]:
    """Macro-average the applicable metrics across benchmark cases."""
    records = list(records)
    metric_names = {
        "replacement": (
            "target_emotion_probability",
            "source_emotion_suppression",
            "emotion_similarity",
            "directional_editing_score",
            "delta_wer",
            "speaker_similarity",
            "utmos",
        ),
        "erasure": (
            "neutral_probability",
            "source_emotion_suppression",
            "emotion_similarity",
            "directional_editing_score",
            "delta_wer",
            "speaker_similarity",
            "utmos",
        ),
        "intensity": (
            "intensity_monotonicity",
            "effective_intensity_control",
            "effective_intensity_control_embedding",
            "reference_progress_range",
            "delta_wer",
            "speaker_similarity",
            "utmos",
        ),
    }[task]
    summary: dict[str, Any] = {"successful_cases": len(records)}
    for metric_name in metric_names:
        values = [
            float(record[metric_name])
            for record in records
            if record.get(metric_name) is not None
        ]
        summary[metric_name] = macro_mean(values)
    if task == "intensity" and include_directions:
        for intensity in sorted({record["target_intensity"] for record in records}):
            intensity_records = [
                record
                for record in records
                if record.get("target_intensity") == intensity
            ]
            summary[intensity] = aggregate(
                intensity_records, task, include_directions=False
            )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for metric aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate EmoEditBench case-level metric files."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("replacement", "erasure", "intensity"), required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run metric aggregation from command-line arguments."""
    args = parse_args(argv)
    manifest = read_jsonl(args.manifest)
    if args.task == "intensity":
        manifest = common_intensity_records(manifest)
    merged = merge_results(args.results_root, args.manifest.stem, manifest)
    write_jsonl(args.output, [aggregate(merged, args.task)])


if __name__ == "__main__":
    main()
