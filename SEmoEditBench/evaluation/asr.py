from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .common import (
    edit_path,
    language_for,
    load_config,
    manifest_path,
    read_jsonl,
    track_progress,
    write_jsonl,
)


DEFAULT_MODEL = "openai/whisper-large-v3"


def source_texts_from_manifest(
    records: list[dict[str, Any]], source: Path | str
) -> dict[str, str] | None:
    """Read complete source ASR text cached in a manifest."""
    texts: dict[str, str] = {}
    seen_ids: set[str] = set()
    complete = True
    for record in records:
        case_id = record.get("id")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"{source}: duplicate or missing manifest id: {case_id!r}")
        seen_ids.add(case_id)
        text = record.get("source_asr")
        if text is None or text == "":
            complete = False
            continue
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source}: invalid source_asr for {case_id}")
        texts[case_id] = text
    return texts if complete else None


def validate_source_predictions(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]], source: Path | str
) -> dict[str, str]:
    """Check cached source ASR texts against manifest IDs."""
    expected_ids = [record.get("id") for record in records]
    if any(not case_id for case_id in expected_ids) or len(expected_ids) != len(
        set(expected_ids)
    ):
        raise ValueError(f"{source}: manifest contains duplicate or missing ids")
    texts: dict[str, str] = {}
    for prediction in predictions:
        case_id = prediction.get("id")
        if not case_id or case_id in texts:
            raise ValueError(
                f"{source}: duplicate or missing prediction id: {case_id!r}"
            )
        if prediction.get("role", "source") != "source":
            raise ValueError(
                f"{source}: prediction {case_id} is not a source ASR result"
            )
        text = prediction.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{source}: missing or empty source ASR text for {case_id}"
            )
        texts[case_id] = text
    expected = set(expected_ids)
    missing = expected - texts.keys()
    extra = texts.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing ids: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"extra ids: {', '.join(sorted(extra))}")
        raise ValueError(
            f"{source}: source ASR cache does not match manifest ({'; '.join(details)})"
        )
    return texts


def write_source_texts_to_manifest(
    manifest_path_value: Path, texts: dict[str, str]
) -> None:
    """Store verified source ASR text in manifest records."""
    records = read_jsonl(manifest_path_value)
    expected_ids = {record.get("id") for record in records}
    if expected_ids != texts.keys():
        missing = expected_ids - texts.keys()
        extra = texts.keys() - expected_ids
        raise ValueError(
            f"{manifest_path_value}: cannot write source ASR cache "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    for record in records:
        record["source_asr"] = texts[record["id"]]
    write_jsonl(manifest_path_value, records)


def source_prediction_records(
    jobs: list[dict[str, str]], texts: dict[str, str]
) -> list[dict[str, str]]:
    """Convert cached source ASR texts to result records."""
    job_ids = {job["id"] for job in jobs}
    if job_ids != texts.keys():
        raise ValueError("Source ASR texts do not match source jobs")
    return [{**job, "text": texts[job["id"]]} for job in jobs]


def build_pipeline(
    model: str, device: str | None, dtype_name: str
) -> Callable[..., Any]:
    """Load the configured speech-recognition pipeline."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if dtype_name == "auto":
        resolved_dtype = (
            torch.float16 if resolved_device.startswith("cuda") else torch.float32
        )
    else:
        resolved_dtype = getattr(torch, dtype_name)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model,
        dtype=resolved_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    asr_model.to(resolved_device)
    processor = AutoProcessor.from_pretrained(model)
    return pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=resolved_dtype,
        device=resolved_device,
    )


def build_jobs(
    manifest_path_value: Path, config_path: Path, output_root: Path, role: str
):
    """Create source or edited-audio ASR jobs from one manifest."""
    config = load_config(config_path)
    records = read_jsonl(manifest_path_value)
    split = manifest_path_value.stem
    jobs = []
    for record in records:
        path = (
            manifest_path(record, config, "source_audio")
            if role == "source"
            else edit_path(output_root, split, record["id"])
        )
        jobs.append(
            {
                "id": record["id"],
                "role": role,
                "audio_path": str(path),
                "language": language_for(record),
            }
        )
    return jobs


def evaluate_manifest(
    manifest_path_value: Path,
    config_path: Path,
    output_root: Path,
    role: str,
    asr_pipeline: Callable[..., Any],
    batch_size: int,
    chunk_length_s: float,
    show_progress: bool = True,
):
    """Transcribe the selected audio role for one manifest."""
    jobs = build_jobs(manifest_path_value, config_path, output_root, role)
    return transcribe(jobs, asr_pipeline, batch_size, chunk_length_s, show_progress)


def transcribe(
    jobs: list[dict[str, str]],
    asr_pipeline: Callable[..., Any],
    batch_size: int,
    chunk_length_s: float,
    show_progress: bool = True,
):
    """Run ASR over pending audio jobs."""
    from torch.utils.data import Subset

    results = [None] * len(jobs)
    for language in dict.fromkeys(job["language"] for job in jobs):
        indices = [
            index for index, job in enumerate(jobs) if job["language"] == language
        ]
        audio_paths = [jobs[index]["audio_path"] for index in indices]
        dataset = Subset(audio_paths, range(len(audio_paths)))
        outputs = asr_pipeline(
            dataset,
            batch_size=batch_size,
            chunk_length_s=chunk_length_s,
            ignore_warning=True,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        role = jobs[indices[0]]["role"]
        progress = track_progress(
            outputs,
            total=len(indices),
            desc=f"ASR {role} ({language})",
            enabled=show_progress,
        )
        for index, output in zip(indices, progress, strict=True):
            results[index] = {**jobs[index], "text": output["text"].strip()}
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for ASR evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Whisper ASR for an EmoEditBench split."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root containing <split>/<case_id>.wav edited outputs",
    )
    parser.add_argument("--role", choices=("source", "edit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length-s", type=float, default=30.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run ASR evaluation from command-line arguments."""
    args = parse_args(argv)
    if args.batch_size <= 0 or args.chunk_length_s <= 0:
        raise ValueError("--batch-size and --chunk-length-s must be positive")
    if args.role == "edit" and args.output_root is None:
        raise ValueError("--output-root is required for --role edit")
    records = read_jsonl(args.manifest)
    jobs = build_jobs(
        args.manifest, args.config, args.output_root or Path("."), args.role
    )
    if not jobs:
        write_jsonl(args.output, [])
        return
    if args.role == "source":
        source_texts = source_texts_from_manifest(records, args.manifest)
        if source_texts is not None:
            write_jsonl(args.output, source_prediction_records(jobs, source_texts))
            return
    model = build_pipeline(args.model, args.device, args.dtype)
    predictions = transcribe(
        jobs,
        model,
        args.batch_size,
        args.chunk_length_s,
        show_progress=not args.no_progress,
    )
    if args.role == "source":
        source_texts = validate_source_predictions(records, predictions, args.output)
        write_source_texts_to_manifest(args.manifest, source_texts)
    write_jsonl(args.output, predictions)


if __name__ == "__main__":
    main()
