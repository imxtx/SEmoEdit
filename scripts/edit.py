"""Edit one sample without benchmark manifests; run in the backbone environment."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semoedit.inference import edit, load_editor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("f5_tts", "cosyvoice2", "indextts2")
    )
    parser.add_argument("--config", type=Path, default=root / "configs/inference.yaml")
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument(
        "--text", required=True, help="Text spoken in the source reference"
    )
    parser.add_argument("--target-reference-audio", type=Path, required=True)
    parser.add_argument("--target-reference-text", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--vocoder", type=Path, help="F5-TTS vocoder directory")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--sway", type=float, help="F5-TTS sway sampling coefficient")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.strength <= 1:
        parser.error("--strength must be between 0 and 1")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["models"][args.model].copy()
    settings.pop("python")
    for field in ("checkpoint", "upstream_root", "vocoder", "steps", "cfg", "sway"):
        value = getattr(args, field)
        if value is not None:
            settings[field] = (
                value.expanduser().resolve() if isinstance(value, Path) else value
            )
    for field in ("checkpoint", "upstream_root", "vocoder"):
        if field in settings:
            settings[field] = (
                config_path.parent / Path(settings[field]).expanduser()
            ).resolve()
    runtime = config["runtime"].copy()
    for field in ("seed", "tau"):
        value = getattr(args, field)
        if value is not None:
            runtime[field] = value
    case = {
        "source_audio": args.source_audio,
        "text": args.text,
        "target_reference_audio": args.target_reference_audio,
        "target_reference_text": args.target_reference_text,
    }
    for field in ("source_audio", "target_reference_audio"):
        path = case[field].expanduser().resolve()
        sf.info(path)
        case[field] = str(path)
    editor = load_editor(args.model, **settings)
    waveform, diagnostics = edit(
        editor, case, args.strength, runtime["seed"], runtime["tau"]
    )
    samples = waveform.detach().float().cpu().numpy().reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Model produced an empty or non-finite waveform")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, samples, editor.sample_rate, subtype="FLOAT")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": args.model,
                "case": case,
                "model_config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in settings.items()
                },
                "runtime": runtime,
                "strength": args.strength,
                "sample_rate": editor.sample_rate,
                "steps": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"WAV {args.output.resolve()} ({len(samples) / editor.sample_rate:.3f}s)")


if __name__ == "__main__":
    main()
