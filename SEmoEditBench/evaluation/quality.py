from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .common import edit_path, read_jsonl, track_progress, write_jsonl


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
    manifest_path: Path,
    output_root: Path,
    device: str,
    model=None,
    show_progress: bool = True,
):
    """Compute speech-quality evaluation metrics for one benchmark split."""
    import utmosv2

    records = read_jsonl(manifest_path)
    split = manifest_path.stem
    model = model or utmosv2.create_model(pretrained=True)
    return [
        {
            "id": record["id"],
            "utmos": scalar_score(
                model.predict(
                    input_path=str(edit_path(output_root, split, record["id"])),
                    device=device,
                )
            ),
        }
        for record in track_progress(
            records, total=len(records), desc=f"Quality {split}", enabled=show_progress
        )
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for speech-quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate edited audio quality with UTMOSv2."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run speech-quality evaluation from command-line arguments."""
    args = parse_args(argv)
    write_jsonl(
        args.output,
        evaluate(
            args.manifest,
            args.output_root,
            args.device,
            show_progress=not args.no_progress,
        ),
    )


if __name__ == "__main__":
    main()
