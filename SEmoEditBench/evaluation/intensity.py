from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .common import INTENSITY_COMMON_EMOTIONS, read_jsonl, write_jsonl
from .effective_intensity import (
    effective_intensity_control,
    effective_intensity_control_embedding,
)

EIC_EMBEDDING_EPSILON = 0.0
EIC_EMBEDDING_TAU = 1.0


def calculate_from_emotion_results(
    manifest: list[dict[str, Any]], emotion_results: list[dict[str, Any]]
):
    """Validate and extract Emotion2Vec-based intensity metrics."""
    by_id = {record["id"]: record for record in emotion_results}
    if len(by_id) != len(emotion_results):
        raise ValueError("Emotion result file contains duplicate ids")
    results = []
    for record in manifest:
        case_id = record["id"]
        result = by_id.get(case_id)
        if result is None:
            raise ValueError(f"Missing Emotion2Vec result for {case_id}")
        required = ("intensity_probabilities", "intensity_monotonicity")
        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(
                f"{case_id}: missing Emotion2Vec metrics: {', '.join(missing)}"
            )
        reference_progress = result.get("reference_progress")
        is_common_case = (
            record["source_emotion"] == "neutral"
            and record["target_emotion"] in INTENSITY_COMMON_EMOTIONS
        )
        results.append(
            {
                "id": case_id,
                "target_intensity": record["target_intensity"],
                "intensity_probabilities": result["intensity_probabilities"],
                "intensity_monotonicity": result["intensity_monotonicity"],
                "effective_intensity_control": effective_intensity_control(
                    result["intensity_probabilities"], 0.0, 0.2
                ),
                "eic_epsilon": 0.0,
                "eic_tau": 0.2,
                "effective_intensity_control_embedding": (
                    effective_intensity_control_embedding(
                        reference_progress, EIC_EMBEDDING_EPSILON, EIC_EMBEDDING_TAU
                    )
                    if reference_progress and is_common_case
                    else None
                ),
                "eic_embedding_epsilon": EIC_EMBEDDING_EPSILON,
                "eic_embedding_tau": EIC_EMBEDDING_TAU,
                "reference_progress": reference_progress,
                "reference_progress_range": result.get("reference_progress_range"),
            }
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for intensity evaluation."""
    parser = argparse.ArgumentParser(
        description="Extract Emotion2Vec-based intensity metrics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emotion-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run intensity evaluation from command-line arguments."""
    args = parse_args(argv)
    write_jsonl(
        args.output,
        calculate_from_emotion_results(
            read_jsonl(args.manifest), read_jsonl(args.emotion_results)
        ),
    )


if __name__ == "__main__":
    main()
