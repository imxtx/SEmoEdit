from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare.audio import load_config as load_benchmark_config
from prepare.audio import prepare_audio

from .asr import build_jobs, build_pipeline, transcribe
from .common import (
    INTENSITY_STRENGTHS,
    common_intensity_records,
    load_config,
    manifest_path,
    read_jsonl,
    records_for_system,
    write_jsonl,
)
from .emotion import DEFAULT_MODEL as DEFAULT_EMOTION_MODEL
from .emotion import build_model as build_emotion_model
from .emotion import evaluate as evaluate_emotion
from .intensity import calculate_from_emotion_results
from .quality import evaluate as evaluate_quality
from .report import build_report, write_report
from .speaker import DEFAULT_MODEL as DEFAULT_SPEAKER_MODEL
from .speaker import build_model as build_speaker_model
from .speaker import evaluate as evaluate_speaker
from .summary import aggregate, merge_results
from .wer import calculate as calculate_wer
from .wer import read_predictions


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


def build_parser() -> argparse.ArgumentParser:
    """Define the benchmark preparation and evaluation commands."""
    parser = argparse.ArgumentParser(
        prog="emoedit",
        description="Prepare and evaluate the EmoEditBench speech editing benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare audio for the fixed 600-case manifests."
    )
    prepare_parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmark.yaml")
    )
    prepare_parser.add_argument("--dry-run", action="store_true")
    prepare_parser.add_argument("--indextts-root", type=Path)
    prepare_parser.set_defaults(handler=run_prepare)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run all metrics for generated edited audio."
    )
    evaluate_parser.add_argument(
        "--manifest-dir", type=Path, default=Path("manifests600")
    )
    evaluate_parser.add_argument(
        "--config", type=Path, default=Path("configs/benchmark.yaml")
    )
    evaluate_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Parent of system directories, or one system directory",
    )
    evaluate_parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results600"),
        help="Parent directory for per-system results and comparison report",
    )
    evaluate_parser.add_argument(
        "--system",
        action="append",
        help="Select system directory names; repeat for multiple systems",
    )
    evaluate_parser.add_argument(
        "--pattern",
        default="*600cases",
        help="System directory pattern for automatic discovery",
    )
    evaluate_parser.add_argument(
        "--split",
        action="append",
        choices=DEFAULT_SPLITS,
        help="Repeat to evaluate selected splits",
    )
    evaluate_parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    evaluate_parser.add_argument("--emotion-model", default=DEFAULT_EMOTION_MODEL)
    evaluate_parser.add_argument("--speaker-model", default=DEFAULT_SPEAKER_MODEL)
    evaluate_parser.add_argument("--emotion-hub", choices=("hf", "ms"), default="hf")
    evaluate_parser.add_argument(
        "--device",
        default=None,
        help="Shared device for ASR, Emotion2Vec, speaker, and quality",
    )
    evaluate_parser.add_argument("--asr-device", default=None)
    evaluate_parser.add_argument(
        "--asr-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--chunk-length-s", type=float, default=30.0)
    evaluate_parser.add_argument("--no-progress", action="store_true")
    evaluate_parser.add_argument(
        "--force", action="store_true", help="Recompute all ASR and metric results"
    )
    evaluate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate edited audio files without loading models",
    )
    evaluate_parser.set_defaults(handler=run_evaluate)

    return parser


def run_prepare(args: argparse.Namespace) -> int:
    """Prepare the audio referenced by the fixed benchmark manifests."""
    config = load_benchmark_config(args.config)
    if args.indextts_root is not None:
        config["indextts2"]["root"] = str(args.indextts_root.expanduser().resolve())
    manifest_dir = Path(__file__).resolve().parents[1] / "manifests600"
    manifests = {
        split: read_jsonl(manifest_dir / f"{split}.jsonl") for split in DEFAULT_SPLITS
    }
    summary = prepare_audio(config, manifests, dry_run=args.dry_run)
    print(
        json.dumps({"command": "prepare", "dry_run": args.dry_run, **summary}, indent=2)
    )
    return 0


def selected_splits(args: argparse.Namespace) -> list[str]:
    """Resolve the requested benchmark split names."""
    splits = args.split or list(DEFAULT_SPLITS)
    missing = [
        split
        for split in splits
        if not (args.manifest_dir / f"{split}.jsonl").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing manifest files: {', '.join(missing)}")
    return splits


def discover_systems(args: argparse.Namespace, splits: list[str]) -> dict[str, Path]:
    """Locate evaluated system output directories."""
    if args.system:
        if any(Path(name).name != name or name in (".", "..") for name in args.system):
            raise ValueError("--system must be a directory name")
        systems = {name: args.output_root / name for name in args.system}
    elif any((args.output_root / split).is_dir() for split in splits):
        systems = {args.output_root.name: args.output_root}
    else:
        systems = {
            path.name: path
            for path in sorted(args.output_root.glob(args.pattern))
            if path.is_dir()
        }
    if not systems:
        raise ValueError(f"No systems found under {args.output_root}")
    for path in systems.values():
        if not path.is_dir():
            raise FileNotFoundError(path)
    return systems


def records_without_failed(
    root: Path, split: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Exclude cases marked as failed by a generator."""
    return [
        record
        for record in records
        if not (root / split / f"{record['id']}.failed").is_file()
    ]


def validate_outputs(systems, manifests):
    """Check generated audio before loading evaluation models."""
    import soundfile as sf

    failures = []
    skipped = set()
    for system, root in systems.items():
        supports_intensity = any(
            (root / split).is_dir()
            for split in DEFAULT_SPLITS
            if split.startswith("intensity_")
        )
        # An absent intensity directory marks unsupported strength control.
        for split, records in manifests.items():
            records = records_without_failed(
                root, split, records_for_system(system, records)
            )
            if not records:
                skipped.add((system, split))
                print(f"Skip {system}/{split}: no supported cases")
                continue
            if records[0]["task"] == "intensity" and not supports_intensity:
                skipped.add((system, split))
                print(
                    f"Skip {system}/{split}: intensity control unsupported (no intensity split directories)"
                )
                continue
            for record in records:
                paths = [root / split / f"{record['id']}.wav"]
                if record["task"] == "intensity":
                    paths.extend(
                        root / split / record["id"] / f"{strength:g}.wav"
                        for strength in INTENSITY_STRENGTHS
                    )
                for path in paths:
                    if not path.is_file() or path.stat().st_size == 0:
                        failures.append(str(path))
                        continue
                    info = sf.info(path)
                    if info.frames == 0:
                        failures.append(str(path))
        print(
            f"Checked {system}: {sum(len(records_without_failed(root, split, records_for_system(system, records))) for split, records in manifests.items())} cases"
        )
    if failures:
        raise ValueError(
            f"Missing or empty WAV files ({len(failures)}):\n" + "\n".join(failures)
        )
    return skipped


def result_is_complete(path: Path, records: list[dict[str, Any]], force: bool) -> bool:
    """Check whether a saved metric file covers all expected IDs."""
    if force or not path.is_file():
        return False
    results = read_jsonl(path)
    return len(results) == len(records) and {result["id"] for result in results} == {
        record["id"] for record in records
    }


def build_quality_model():
    """Load the UTMOS quality estimator."""
    from utmosv2 import create_model

    return create_model(pretrained=True)


def run_evaluate(args: argparse.Namespace) -> int:
    """Validate outputs and compute all objective metrics."""
    if args.batch_size <= 0 or args.chunk_length_s <= 0:
        raise ValueError("--batch-size and --chunk-length-s must be positive")
    splits = selected_splits(args)
    manifests = {
        split: read_jsonl(args.manifest_dir / f"{split}.jsonl") for split in splits
    }
    for split, records in manifests.items():
        ids = [record["id"] for record in records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{split}: empty manifest or duplicate case IDs")
    systems = discover_systems(args, splits)
    skipped = validate_outputs(systems, manifests)
    system_manifests = {
        system: {
            split: records_without_failed(
                root, split, records_for_system(system, records)
            )
            for split, records in manifests.items()
        }
        for system, root in systems.items()
    }
    config = load_config(args.config)
    for split, records in manifests.items():
        if all((system, split) in skipped for system in systems):
            continue
        for record in records:
            if not any(
                record in system_manifests[system][split]
                for system in systems
                if (system, split) not in skipped
            ):
                continue
            manifest_path(record, config, "source_audio")
            if record["task"] != "intensity" and record.get("target_audio"):
                manifest_path(record, config, "target_audio")
    if args.dry_run:
        return 0

    settings = {
        "protocol": 2,
        "manifest_dir": str(args.manifest_dir.resolve()),
        "manifests": manifests,
        "config": config,
        "asr_model": args.asr_model,
        "asr_dtype": args.asr_dtype,
        "chunk_length_s": args.chunk_length_s,
        "emotion_model": args.emotion_model,
        "emotion_hub": args.emotion_hub,
        "speaker_model": args.speaker_model,
        "quality_model": "utmosv2/pretrained",
    }
    for system, root in systems.items():
        result_root = args.results_root / system
        settings_path = result_root / "evaluation.jsonl"
        expected_settings = {
            **settings,
            "manifests": system_manifests[system],
            "output_root": str(root.resolve()),
        }
        if settings_path.exists():
            if read_jsonl(settings_path) != [expected_settings] and not args.force:
                raise ValueError(
                    f"{system}: evaluation settings changed; use --force or a new --results-root"
                )
        elif any(result_root.glob("*/*.jsonl")) and not args.force:
            raise ValueError(
                f"{system}: unversioned results; use --force or a new --results-root"
            )
    for system, root in systems.items():
        result_root = args.results_root / system
        if args.force:
            for split in splits:
                for name in (
                    "source_asr",
                    "edit_asr",
                    "wer",
                    "emotion",
                    "speaker",
                    "quality",
                    "intensity",
                    "summary",
                ):
                    (result_root / split / f"{name}.jsonl").unlink(missing_ok=True)
        write_jsonl(
            result_root / "evaluation.jsonl",
            [
                {
                    **settings,
                    "manifests": system_manifests[system],
                    "output_root": str(root.resolve()),
                }
            ],
        )
        for split, records in system_manifests[system].items():
            if (system, split) in skipped:
                write_jsonl(
                    result_root / split / "summary.jsonl",
                    [
                        {
                            "status": "unsupported",
                            "reason": "no supported cases"
                            if not records
                            else "no intensity split directories",
                            "expected_cases": len(records),
                            "successful_cases": 0,
                            "failed_cases": 0,
                            "failure_rate": None,
                            **{
                                metric: None
                                for metric in (
                                    "intensity_monotonicity",
                                    "effective_intensity_control",
                                    "effective_intensity_control_embedding",
                                    "reference_progress_range",
                                    "delta_wer",
                                    "speaker_similarity",
                                    "utmos",
                                )
                            },
                        }
                    ],
                )
            else:
                summary_path = result_root / split / "summary.jsonl"
                if (
                    summary_path.is_file()
                    and read_jsonl(summary_path)[0].get("status") == "unsupported"
                ):
                    summary_path.unlink()

    manifest_paths = {}
    for system in systems:
        for split, records in system_manifests[system].items():
            path = args.manifest_dir / f"{split}.jsonl"
            if records != manifests[split]:
                path = args.results_root / system / "_manifests" / f"{split}.jsonl"
                write_jsonl(path, records)
            manifest_paths[system, split] = path

    device = args.device

    def resolved_device():
        """Select the evaluation device once for all metric models."""
        nonlocal device
        if device is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    wer_pending = [
        (system, split)
        for system in systems
        for split in splits
        if (system, split) not in skipped
        if not result_is_complete(
            args.results_root / system / split / "wer.jsonl",
            system_manifests[system][split],
            args.force,
        )
    ]
    edit_pending = [
        (system, split)
        for system in systems
        for split in splits
        if (system, split) not in skipped
        if not result_is_complete(
            args.results_root / system / split / "edit_asr.jsonl",
            system_manifests[system][split],
            args.force,
        )
    ]
    source_jobs = {
        (system, split): build_jobs(
            manifest_paths[system, split], args.config, systems[system], "source"
        )
        for system, split in wer_pending
    }
    unique_jobs = {}
    for jobs in source_jobs.values():
        for job in jobs:
            unique_jobs.setdefault((job["audio_path"], job["language"]), job)
    source_cache = args.results_root / "_cache" / "source_asr.jsonl"
    source_settings_path = args.results_root / "_cache" / "source_asr_settings.jsonl"
    source_settings = {
        "model": args.asr_model,
        "dtype": args.asr_dtype,
        "chunk_length_s": args.chunk_length_s,
        "jobs": list(unique_jobs.values()),
    }
    source_predictions = []
    if not args.force and source_cache.is_file() and source_settings_path.is_file():
        if read_jsonl(source_settings_path) == [source_settings]:
            source_predictions = read_jsonl(source_cache)
    source_by_audio = {
        (record["audio_path"], record["language"]): record
        for record in source_predictions
    }
    need_source = bool(unique_jobs) and source_by_audio.keys() != unique_jobs.keys()
    if need_source or edit_pending:
        model = build_pipeline(
            args.asr_model, args.asr_device or resolved_device(), args.asr_dtype
        )
        if need_source:
            source_predictions = transcribe(
                list(unique_jobs.values()),
                model,
                args.batch_size,
                args.chunk_length_s,
                show_progress=not args.no_progress,
            )
            source_by_audio = {
                (record["audio_path"], record["language"]): record
                for record in source_predictions
            }
            write_jsonl(source_cache, source_predictions)
            write_jsonl(source_settings_path, [source_settings])
        for system, split in edit_pending:
            jobs = build_jobs(
                manifest_paths[system, split], args.config, systems[system], "edit"
            )
            predictions = transcribe(
                jobs,
                model,
                args.batch_size,
                args.chunk_length_s,
                show_progress=not args.no_progress,
            )
            write_jsonl(
                args.results_root / system / split / "edit_asr.jsonl", predictions
            )
        del model

    for system, split in wer_pending:
        records = system_manifests[system][split]
        result_dir = args.results_root / system / split
        predictions = [
            {**job, "text": source_by_audio[job["audio_path"], job["language"]]["text"]}
            for job in source_jobs[system, split]
        ]
        write_jsonl(result_dir / "source_asr.jsonl", predictions)
        write_jsonl(
            result_dir / "wer.jsonl",
            calculate_wer(
                records,
                {record["id"]: record for record in predictions},
                read_predictions(result_dir / "edit_asr.jsonl"),
            ),
        )

    for metric in ("emotion", "speaker", "quality"):
        pending = [
            (system, split)
            for system in systems
            for split in splits
            if (system, split) not in skipped
            if not result_is_complete(
                args.results_root / system / split / f"{metric}.jsonl",
                system_manifests[system][split],
                args.force,
            )
        ]
        if not pending:
            print(f"Reused all {metric} results")
            continue
        if metric == "emotion":
            model = build_emotion_model(
                args.emotion_model, resolved_device(), args.emotion_hub
            )
        elif metric == "speaker":
            model = build_speaker_model(args.speaker_model, resolved_device())
        else:
            model = build_quality_model()
        audio_cache = {}
        for system, split in pending:
            print(f"Evaluate {metric}: {system}/{split}", flush=True)
            manifest = manifest_paths[system, split]
            options = {"model": model, "show_progress": not args.no_progress}
            if metric == "emotion":
                results = evaluate_emotion(
                    manifest,
                    args.config,
                    systems[system],
                    args.emotion_model,
                    resolved_device(),
                    args.emotion_hub,
                    args.results_root / "_cache" / "embeddings",
                    reuse_predictions=False,
                    audio_cache=audio_cache,
                    **options,
                )
            elif metric == "speaker":
                results = evaluate_speaker(
                    manifest,
                    args.config,
                    systems[system],
                    args.speaker_model,
                    resolved_device(),
                    **options,
                )
            else:
                results = evaluate_quality(
                    manifest, systems[system], resolved_device(), **options
                )
            write_jsonl(args.results_root / system / split / f"{metric}.jsonl", results)
        del model, audio_cache, options

    for system in systems:
        for split, records in system_manifests[system].items():
            if (system, split) in skipped:
                continue
            result_root = args.results_root / system
            comparison_records = (
                common_intensity_records(records)
                if records[0]["task"] == "intensity"
                else records
            )
            if records[0]["task"] == "intensity":
                write_jsonl(
                    result_root / split / "intensity.jsonl",
                    calculate_from_emotion_results(
                        records,
                        read_jsonl(result_root / split / "emotion.jsonl"),
                    ),
                )
            summary = aggregate(
                merge_results(result_root, split, comparison_records),
                records[0]["task"],
            )
            summary.update(
                expected_cases=len(comparison_records), failed_cases=0, failure_rate=0.0
            )
            write_jsonl(result_root / split / "summary.jsonl", [summary])
    rows, directions = build_report(
        args.results_root,
        args.manifest_dir,
        systems=list(systems),
        splits=splits,
    )
    write_report(args.results_root / "comparison_report", rows, directions)
    print(
        f"Wrote {len(systems)} evaluated systems and comparison report for {len({row['system'] for row in rows})} systems to {args.results_root}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run benchmark evaluation from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
