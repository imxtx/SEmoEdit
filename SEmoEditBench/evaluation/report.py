from __future__ import annotations

import argparse
import csv
import html
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .common import common_intensity_records, macro_mean, read_jsonl, records_for_system


DEFAULT_SPLITS = (
    "replacement_same_dataset_same_speaker",
    "replacement_same_dataset_cross_speaker",
    "replacement_cross_dataset_cross_speaker",
    "erasure_same_dataset_same_speaker",
    "erasure_same_dataset_cross_speaker",
    "erasure_cross_dataset_cross_speaker",
    "intensity_same_dataset_same_speaker",
    "intensity_same_dataset_cross_speaker",
    "intensity_cross_dataset_cross_speaker",
)

METRICS = {
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
}

METRIC_LABELS = {
    "target_emotion_probability": "TEP",
    "neutral_probability": "NP",
    "source_emotion_suppression": "SES",
    "intensity_monotonicity": "Monotonicity",
    "effective_intensity_control": "EIC (experimental, epsilon=0, tau=0.2)",
    "effective_intensity_control_embedding": "EIC-Emb (eps=0, tau=1)",
    "reference_progress_range": "Reference progress range ↑",
    "emotion_similarity": "E-SIM",
    "directional_editing_score": "DES",
    "delta_wer": "delta WER",
    "speaker_similarity": "S-SIM",
    "utmos": "UTMOS",
}


def applicable_metrics(task: str, split: str) -> tuple[str, ...]:
    """List the objective metrics for a benchmark task."""
    return METRICS[task]


def discover_systems(
    results_root: Path, requested: Sequence[str] | None = None
) -> dict[str, Path]:
    """Locate evaluated system output directories."""
    if requested:
        systems = {name: results_root / name for name in requested}
    elif any((results_root / split).is_dir() for split in DEFAULT_SPLITS):
        systems = {results_root.name: results_root}
    else:
        systems = {
            path.name: path
            for path in sorted(results_root.iterdir())
            if path.is_dir()
            and any((path / split).is_dir() for split in DEFAULT_SPLITS)
        }
    if not systems:
        raise FileNotFoundError(
            f"No system result directories found under {results_root}"
        )
    missing = [
        f"{name} ({path})" for name, path in systems.items() if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing system result directories: " + ", ".join(missing)
        )
    return systems


def _by_id(records: Iterable[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    """Index case-level results by benchmark case ID."""
    indexed = {}
    for record in records:
        case_id = record.get("id")
        if not case_id or case_id in indexed:
            raise ValueError(f"{path}: duplicate or missing id: {case_id!r}")
        indexed[case_id] = record
    return indexed


def _load_case_results(
    system_root: Path, split: str, manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge all available metrics for cases in one split."""
    result_dir = system_root / split
    sources = {}
    for name in ("wer", "emotion", "speaker", "quality", "intensity"):
        path = result_dir / f"{name}.jsonl"
        if path.exists():
            sources[name] = _by_id(read_jsonl(path), path)
    if not sources:
        return []
    merged = []
    for record in manifest:
        merged_record = {
            "id": record["id"],
            "target_intensity": record.get("target_intensity"),
        }
        for values in sources.values():
            if record["id"] in values:
                merged_record.update(values[record["id"]])
        merged.append(merged_record)
    return merged


def _mean(records: Iterable[dict[str, Any]], metric: str) -> float | None:
    """Compute the mean of available values for one metric."""
    values = [
        float(record[metric]) for record in records if record.get(metric) is not None
    ]
    return macro_mean(values)


def _fallback_summary(system_root: Path, split: str) -> dict[str, Any] | None:
    """Read a split summary when case-level metrics are unavailable."""
    path = system_root / split / "summary.jsonl"
    if not path.exists():
        return None
    summaries = read_jsonl(path)
    if len(summaries) != 1:
        raise ValueError(f"{path}: expected exactly one summary record")
    return summaries[0]


def _build_row(
    system: str, system_root: Path, split: str, manifest_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Combine one system and split into a comparison-report row."""
    manifest_path = manifest_dir / f"{split}.jsonl"
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError(f"{manifest_path}: manifest is empty")
    task = manifest[0]["task"]
    metrics = applicable_metrics(task, split)
    fallback = _fallback_summary(system_root, split)
    if fallback is not None and fallback.get("status") == "unsupported":
        row = {
            "system": system,
            "split": split,
            "task": task,
            "setting": manifest[0]["setting"],
            **fallback,
            **{metric: None for metric in metrics},
        }
        directions = [
            {
                "direction": intensity,
                "cases": 0,
                "status": "unsupported",
                **{metric: None for metric in metrics},
            }
            for intensity in (
                sorted({record["target_intensity"] for record in manifest})
                if task == "intensity"
                else []
            )
        ]
        return row, directions
    manifest = records_for_system(system, manifest)
    if task == "intensity":
        manifest = common_intensity_records(manifest)
    case_results = _load_case_results(system_root, split, manifest)
    if case_results:
        successful = [
            result
            for record, result in zip(manifest, case_results, strict=True)
            if all(
                result.get(metric) is not None
                for metric in metrics
                if metric not in ("emotion_similarity", "directional_editing_score")
                or record.get("target_audio")
            )
        ]
        row = {
            "system": system,
            "split": split,
            "task": task,
            "setting": manifest[0]["setting"],
            "expected_cases": len(manifest),
            "successful_cases": len(successful),
            "failed_cases": len(manifest) - len(successful),
        }
        for metric in metrics:
            row[metric] = _mean(case_results, metric)
        directions = []
        if task == "intensity":
            for intensity in sorted(
                {record["target_intensity"] for record in manifest}
            ):
                intensity_records = [
                    record
                    for record in case_results
                    if record.get("target_intensity") == intensity
                ]
                directions.append(_metric_row(intensity, intensity_records, metrics))
        row["failure_rate"] = row["failed_cases"] / row["expected_cases"]
        return row, directions
    if fallback is None:
        raise FileNotFoundError(
            f"{system_root / split}: no metric or summary output found"
        )
    successful_cases = int(fallback.get("successful_cases", 0))
    row = {
        "system": system,
        "split": split,
        "task": task,
        "setting": manifest[0]["setting"],
        "expected_cases": len(manifest),
        "successful_cases": successful_cases,
        "failed_cases": len(manifest) - successful_cases,
        "failure_rate": (len(manifest) - successful_cases) / len(manifest),
    }
    for metric in metrics:
        row[metric] = fallback.get(metric)
    directions = []
    if task == "intensity":
        for intensity in sorted({record["target_intensity"] for record in manifest}):
            direction_summary = fallback.get(intensity, {})
            directions.append(
                {
                    "direction": intensity,
                    "cases": direction_summary.get("successful_cases", 0),
                    **{metric: direction_summary.get(metric) for metric in metrics},
                }
            )
    return row, directions


def _metric_row(
    direction: str, records: list[dict[str, Any]], metrics: tuple[str, ...]
) -> dict[str, Any]:
    """Aggregate metrics for one target-intensity subgroup."""
    return {
        "direction": direction,
        "cases": len(records),
        **{metric: _mean(records, metric) for metric in metrics},
    }


def build_report(
    results_root: Path,
    manifest_dir: Path,
    systems: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
):
    """Assemble per-split and per-intensity comparison rows."""
    system_paths = discover_systems(results_root, systems)
    if splits:
        selected_splits = list(splits)
    else:
        selected_splits = [
            split
            for split in DEFAULT_SPLITS
            if (manifest_dir / f"{split}.jsonl").is_file()
            and any(
                (system_root / split).is_dir() for system_root in system_paths.values()
            )
        ]
        if not selected_splits:
            raise FileNotFoundError(
                f"No reportable splits found under {manifest_dir} and {results_root}"
            )
    rows = []
    directions = []
    for system, system_root in system_paths.items():
        for split in selected_splits:
            if not (system_root / split).is_dir():
                continue
            row, split_directions = _build_row(system, system_root, split, manifest_dir)
            rows.append(row)
            directions.extend(
                {"system": system, "split": split, **direction}
                for direction in split_directions
            )
    return rows, directions


def _format_value(value: Any) -> str:
    """Format a metric value for report tables."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    """Render comparison rows as a Markdown table."""
    is_direction_table = "cases" in rows[0]
    if is_direction_table:
        headers = ["System", "Cases", *(METRIC_LABELS[metric] for metric in metrics)]
    else:
        headers = [
            "System",
            "N",
            "Failed",
            "Failure rate",
            *(METRIC_LABELS[metric] for metric in metrics),
        ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        if is_direction_table:
            values = [row["system"], row["cases"]]
        else:
            values = [
                row["system"],
                row["successful_cases"],
                row["failed_cases"],
                "N/A"
                if row.get("status") == "unsupported"
                else f"{row['failure_rate']:.2%}",
            ]
        values.extend(
            "N/A"
            if row.get("status") == "unsupported"
            else _format_value(row.get(metric))
            for metric in metrics
        )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def render_markdown(
    rows: list[dict[str, Any]], directions: list[dict[str, Any]]
) -> str:
    """Render a Markdown comparison report."""
    lines = [
        "# EmoEditBench Evaluation Report",
        "",
        "Results are reported independently for each benchmark split.",
        "",
    ]
    for split in DEFAULT_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        metrics = applicable_metrics(split_rows[0]["task"], split)
        lines.extend([f"## {split}", "", _table(split_rows, metrics), ""])
        if split_rows[0]["task"] == "intensity":
            for direction in sorted(
                {row["direction"] for row in directions if row["split"] == split}
            ):
                direction_rows = [
                    row
                    for row in directions
                    if row["split"] == split and row["direction"] == direction
                ]
                if direction_rows:
                    lines.extend(
                        [f"### {direction}", "", _table(direction_rows, metrics), ""]
                    )
    return "\n".join(lines).rstrip() + "\n"


def _html_table(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    """Render comparison rows as an HTML table."""
    is_direction_table = "cases" in rows[0]
    if is_direction_table:
        headers = ["System", "Cases", *(METRIC_LABELS[metric] for metric in metrics)]
    else:
        headers = [
            "System",
            "N",
            "Failed",
            "Failure rate",
            *(METRIC_LABELS[metric] for metric in metrics),
        ]
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = []
    for row in rows:
        if is_direction_table:
            values = [row["system"], row["cases"]]
        else:
            values = [
                row["system"],
                row["successful_cases"],
                row["failed_cases"],
                "N/A"
                if row.get("status") == "unsupported"
                else f"{row['failure_rate']:.2%}",
            ]
        values.extend(
            "N/A"
            if row.get("status") == "unsupported"
            else _format_value(row.get(metric))
            for metric in metrics
        )
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def render_html(rows: list[dict[str, Any]], directions: list[dict[str, Any]]) -> str:
    """Render a self-contained HTML comparison report."""
    sections = []
    for split in DEFAULT_SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        metrics = applicable_metrics(split_rows[0]["task"], split)
        section = [f"<h2>{html.escape(split)}</h2>", _html_table(split_rows, metrics)]
        if split_rows[0]["task"] == "intensity":
            for direction in sorted(
                {row["direction"] for row in directions if row["split"] == split}
            ):
                direction_rows = [
                    row
                    for row in directions
                    if row["split"] == split and row["direction"] == direction
                ]
                if direction_rows:
                    section.extend(
                        [f"<h3>{direction}</h3>", _html_table(direction_rows, metrics)]
                    )
        sections.append("\n".join(section))
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>EmoEditBench Evaluation Report</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#222}table{border-collapse:collapse;margin:0 0 1.5rem;width:100%;font-size:.9rem}th,td{border:1px solid #ddd;padding:.45rem;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f2f4f7}h1{margin-bottom:.25rem}h2{border-bottom:2px solid #444;padding-bottom:.25rem}</style>
</head><body><h1>EmoEditBench Evaluation Report</h1><p>Results are reported independently for each benchmark split.</p>
"""
        + "\n".join(sections)
        + "</body></html>\n"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Export comparison rows as CSV."""
    fields = [
        "system",
        "split",
        "task",
        "setting",
        "status",
        "expected_cases",
        "successful_cases",
        "failed_cases",
        "failure_rate",
        *METRIC_LABELS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    **(
                        {
                            metric: "N/A"
                            for metric in (*METRICS[row["task"]], "failure_rate")
                        }
                        if row.get("status") == "unsupported"
                        else {}
                    ),
                }
            )


def write_report(
    output: Path, rows: list[dict[str, Any]], directions: list[dict[str, Any]]
) -> None:
    """Write Markdown, HTML, CSV, and JSON comparison reports."""
    output.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "directions": directions}
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "summary.csv", rows)
    (output / "report.md").write_text(
        render_markdown(rows, directions), encoding="utf-8"
    )
    (output / "report.html").write_text(render_html(rows, directions), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for comparison-report generation."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown, HTML, CSV, and JSON reports for EmoEditBench systems."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Parent directory containing one result directory per system",
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument(
        "--system",
        action="append",
        help="System directory name; repeat to compare selected systems",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=DEFAULT_SPLITS,
        help="Split to include; repeat for selected splits",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run comparison-report generation from command-line arguments."""
    args = parse_args(argv)
    rows, directions = build_report(
        args.results_root, args.manifest_dir, args.system, args.split
    )
    write_report(args.output, rows, directions)
    print(
        f"Wrote report for {len({row['system'] for row in rows})} systems to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
