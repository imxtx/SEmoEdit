"""Edit one sample without benchmark manifests; run in the backbone environment."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semoedit.inference import edit, load_editor
from semoedit.bridging import EmotionBridge


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
    parser.add_argument(
        "--no-emotion-bridging",
        action="store_false",
        dest="emotion_bridging",
        help="Use direct transport only for intermediate strengths",
    )
    parser.add_argument(
        "--donor-index", type=Path, help="Complete donor index directory"
    )
    parser.add_argument(
        "--emotion-model", type=Path, help="Local Emotion2Vec checkpoint"
    )
    parser.add_argument(
        "--emotion-python", type=Path, help="Emotion2Vec environment executable"
    )
    parser.add_argument(
        "--bridge-python", type=Path, help="IndexTTS2 environment executable"
    )
    parser.add_argument(
        "--bridge-checkpoint", type=Path, help="IndexTTS2 checkpoint directory"
    )
    parser.add_argument(
        "--bridge-upstream-root", type=Path, help="IndexTTS2 source directory"
    )
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
    bridging = None
    if args.emotion_bridging and 0 < args.strength < 1:
        defaults = {
            "donor_index": config["bridging"]["index_root"],
            "emotion_model": config["bridging"]["emotion2vec_model"],
            "emotion_python": runtime["benchmark_python"],
            "bridge_python": config["models"]["indextts2"]["python"],
            "bridge_checkpoint": config["models"]["indextts2"]["checkpoint"],
            "bridge_upstream_root": config["models"]["indextts2"]["upstream_root"],
        }
        bridge_paths = {}
        for field, default in defaults.items():
            override = getattr(args, field)
            path = (
                override if override is not None else config_path.parent / default
            ).expanduser()
            bridge_paths[field] = (
                Path(os.path.abspath(path))
                if field.endswith("python")
                else path.resolve()
            )
        bridging = EmotionBridge(
            index_root=bridge_paths["donor_index"],
            emotion2vec_model=bridge_paths["emotion_model"],
            emotion_python=bridge_paths["emotion_python"],
            index_python=bridge_paths["bridge_python"],
            index_checkpoint=bridge_paths["bridge_checkpoint"],
            index_upstream_root=bridge_paths["bridge_upstream_root"],
            work_dir=args.output.with_suffix("").with_name(
                args.output.stem + "_bridging"
            ),
        )
    editor = load_editor(args.model, **settings)
    waveform, diagnostics = edit(
        editor,
        case,
        args.strength,
        runtime["seed"],
        runtime["tau"],
        bridging=bridging,
        emotion_bridging=args.emotion_bridging,
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
                "emotion_bridging": args.emotion_bridging,
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
