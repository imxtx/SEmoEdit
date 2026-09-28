from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .common import read_jsonl, records_for_system, write_jsonl
from .report import DEFAULT_SPLITS, discover_systems


STRENGTHS = ("0", "0.25", "0.5", "0.75", "1")


def effective_intensity_control(
    probabilities: Mapping[str, float], epsilon: float, tau: float
) -> float:
    """Average signed, thresholded changes over all ordered strength pairs."""
    if not (0 <= epsilon < tau <= 1):
        raise ValueError("Require 0 <= epsilon < tau <= 1")
    values = [float(probabilities[strength]) for strength in STRENGTHS]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Intensity probabilities must be finite and in [0, 1]")
    scores = []
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            difference = second - first
            magnitude = min(
                1.0, max(0.0, (abs(difference) - epsilon) / (tau - epsilon))
            )
            scores.append(math.copysign(magnitude, difference))
    return sum(scores) / len(scores)


def effective_intensity_control_embedding(
    reference_progress: Mapping[str, float], epsilon: float, tau: float
) -> float:
    """Score ordered strength pairs using embedding progress."""
    if not (0 <= epsilon < tau):
        raise ValueError("Require 0 <= epsilon < tau")
    values = [float(reference_progress[strength]) for strength in STRENGTHS]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Reference progress values must be finite")
    scores = []
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            difference = second - first
            magnitude = min(
                1.0, max(0.0, (abs(difference) - epsilon) / (tau - epsilon))
            )
            scores.append(math.copysign(magnitude, difference))
    return sum(scores) / len(scores)


def supplement(
    results_root: Path, manifest_dir: Path, output: Path, epsilon: float, tau: float
):
    """Write additional intensity-control scores from saved results."""
    effective_intensity_control(dict.fromkeys(STRENGTHS, 0.0), epsilon, tau)
    if output.resolve().is_relative_to(results_root.resolve()):
        raise ValueError(
            "Use a separate output directory outside the existing results root"
        )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for system, system_root in discover_systems(results_root).items():
        for split in DEFAULT_SPLITS:
            if not split.startswith("intensity_") or not (system_root / split).is_dir():
                continue
            (summary,) = read_jsonl(system_root / split / "summary.jsonl")
            manifest_path = system_root / "_manifests" / f"{split}.jsonl"
            if not manifest_path.exists():
                manifest_path = manifest_dir / f"{split}.jsonl"
            manifest = records_for_system(system, read_jsonl(manifest_path))
            row = {
                "system": system,
                "split": split,
                "status": summary.get("status", "ok"),
                "expected_cases": len(manifest),
                "scored_cases": 0,
                "intensity_monotonicity": None,
                "effective_intensity_control": None,
                "epsilon": epsilon,
                "tau": tau,
            }
            if row["status"] != "unsupported":
                emotion_path = system_root / split / "emotion.jsonl"
                emotions = read_jsonl(emotion_path)
                by_id = {record["id"]: record for record in emotions}
                expected_ids = {record["id"] for record in manifest}
                if len(by_id) != len(emotions) or set(by_id) != expected_ids:
                    raise ValueError(
                        f"{emotion_path}: duplicate IDs or manifest coverage mismatch"
                    )
                cases = []
                for record in manifest:
                    emotion = by_id[record["id"]]
                    cases.append(
                        {
                            "id": record["id"],
                            "target_intensity": record["target_intensity"],
                            "intensity_probabilities": emotion[
                                "intensity_probabilities"
                            ],
                            "intensity_monotonicity": emotion["intensity_monotonicity"],
                            "effective_intensity_control": effective_intensity_control(
                                emotion["intensity_probabilities"],
                                epsilon,
                                tau,
                            ),
                            "epsilon": epsilon,
                            "tau": tau,
                        }
                    )
                row["scored_cases"] = len(cases)
                for metric in ("intensity_monotonicity", "effective_intensity_control"):
                    row[metric] = sum(case[metric] for case in cases) / len(cases)
                write_jsonl(
                    output / system / split / "effective_intensity.jsonl", cases
                )
            rows.append(row)
    write_jsonl(output / "summary.jsonl", rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "metric": "effective_intensity_control",
        "version": 1,
        "epsilon": epsilon,
        "tau": tau,
        "calibration": "not validated by this command",
        "results_root": str(results_root.resolve()),
        "manifest_dir": str(manifest_dir.resolve()),
        "strengths": list(STRENGTHS),
        "range": [-1, 1],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Effective intensity control supplement",
        "",
        f"epsilon={epsilon:g}, tau={tau:g}. Threshold calibration is not validated by this command.",
        "Existing metrics and files are unchanged. Means use each system's original evaluation subset; subsets can differ.",
        "",
        "| System | Split | Cases | Monotonicity | EIC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        monotonicity = (
            "N/A"
            if row["intensity_monotonicity"] is None
            else f"{row['intensity_monotonicity']:.6f}"
        )
        score = (
            "N/A"
            if row["effective_intensity_control"] is None
            else f"{row['effective_intensity_control']:.6f}"
        )
        lines.append(
            f"| {row['system']} | {row['split']} | {row['scored_cases']}/{row['expected_cases']} | {monotonicity} | {score} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Run intensity-control supplement from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute EIC from saved Emotion2Vec probabilities without inference."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--tau", type=float, required=True)
    args = parser.parse_args(argv)
    rows = supplement(
        args.results_root, args.manifest_dir, args.output, args.epsilon, args.tau
    )
    print(
        f"Wrote {len(rows)} split summaries and {sum(row['scored_cases'] for row in rows)} case scores to {args.output}"
    )


if __name__ == "__main__":
    main()
