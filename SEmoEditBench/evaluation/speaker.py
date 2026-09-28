from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .common import (
    edit_path,
    load_config,
    manifest_path,
    read_jsonl,
    track_progress,
    write_jsonl,
)


DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def build_model(model: str, device: str):
    """Load the ECAPA-TDNN speaker verification model."""
    from speechbrain.inference.speaker import SpeakerRecognition

    return SpeakerRecognition.from_hparams(source=model, run_opts={"device": device})


def scalar_score(value) -> float:
    """Convert a model score tensor to a Python float."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def evaluate(
    manifest_path_value: Path,
    config_path: Path,
    output_root: Path,
    model_name: str,
    device: str,
    model=None,
    show_progress: bool = True,
):
    """Compute speaker-similarity evaluation metrics for one benchmark split."""
    config = load_config(config_path)
    records = read_jsonl(manifest_path_value)
    split = manifest_path_value.stem
    model = model or build_model(model_name, device)
    results = []
    for record in track_progress(
        records, total=len(records), desc=f"Speaker {split}", enabled=show_progress
    ):
        source = manifest_path(record, config, "source_audio")
        edited = edit_path(output_root, split, record["id"])
        score, _ = model.verify_files(str(source), str(edited))
        results.append({"id": record["id"], "speaker_similarity": scalar_score(score)})
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for speaker-similarity evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate source/edit speaker similarity with ECAPA-TDNN."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run speaker-similarity evaluation from command-line arguments."""
    args = parse_args(argv)
    write_jsonl(
        args.output,
        evaluate(
            args.manifest,
            args.config,
            args.output_root,
            args.model,
            args.device,
            show_progress=not args.no_progress,
        ),
    )


if __name__ == "__main__":
    main()
