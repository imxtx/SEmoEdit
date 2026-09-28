from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .common import language_for, load_config, normalize_text, read_jsonl, write_jsonl


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Index saved ASR predictions by case ID."""
    predictions = {}
    for record in read_jsonl(path):
        case_id = record.get("id")
        if not case_id or case_id in predictions:
            raise ValueError(f"{path}: duplicate or missing prediction id: {case_id!r}")
        if not record.get("text") and record.get("text") != "":
            raise ValueError(f"{path}: missing text for {case_id}")
        predictions[case_id] = record
    return predictions


def error_rate(reference: str, hypothesis: str, language: str) -> float:
    """Compute language-appropriate word or character error rate."""
    import jiwer

    reference = normalize_text(reference, language)
    hypothesis = normalize_text(hypothesis, language)
    return float(jiwer.wer(reference, hypothesis))


def calculate(
    manifest: list[dict[str, Any]],
    source_predictions: dict[str, dict[str, Any]],
    edit_predictions: dict[str, dict[str, Any]],
):
    """Compute change in transcription error for each case."""
    results = []
    for record in manifest:
        case_id = record["id"]
        if case_id not in source_predictions or case_id not in edit_predictions:
            raise ValueError(f"Missing source or edit ASR prediction for {case_id}")
        language = language_for(record)
        source_wer = error_rate(
            record["text"], source_predictions[case_id]["text"], language
        )
        edit_wer = error_rate(
            record["text"], edit_predictions[case_id]["text"], language
        )
        results.append(
            {
                "id": case_id,
                "language": language,
                "source_wer": source_wer,
                "edit_wer": edit_wer,
                "delta_wer": edit_wer - source_wer,
            }
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for transcription-error evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate source/edit WER or CER for an EmoEditBench split."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-asr", type=Path, required=True)
    parser.add_argument("--edit-asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run transcription-error evaluation from command-line arguments."""
    args = parse_args(argv)
    records = read_jsonl(args.manifest)
    results = calculate(
        records, read_predictions(args.source_asr), read_predictions(args.edit_asr)
    )
    write_jsonl(args.output, results)


if __name__ == "__main__":
    main()
