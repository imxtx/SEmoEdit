import argparse
import json
from pathlib import Path
import sys

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semoedit.adapters.common import seed_all
from semoedit.adapters.indextts2 import IndexEditor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cfg", type=float, required=True)
    args = parser.parse_args()

    editor = None
    rows = [
        json.loads(line)
        for line in args.matches.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for number, row in enumerate(rows, 1):
        destination = args.output_root / row["split"] / f"{row['job_id']}.wav"
        metadata_path = destination.with_suffix(".json")
        provenance = {
            key: row[key]
            for key in (
                "job_id",
                "source_audio",
                "donor_audio",
                "text",
                "condition_seed",
            )
        }
        if metadata_path.is_file():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != provenance:
                raise ValueError(f"Bridge target inputs changed for {row['job_id']}")
            if destination.is_file() and sf.info(destination).frames:
                print(f"SKIP TARGET {number}/{len(rows)} {row['job_id']}", flush=True)
                continue
        if editor is None:
            editor = IndexEditor(
                args.checkpoint, args.upstream_root, args.steps, args.cfg
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        seed_all(int(row["condition_seed"]))
        result = editor.tts.infer(
            spk_audio_prompt=row["source_audio"],
            emo_audio_prompt=row["donor_audio"],
            emo_alpha=1.0,
            text=row["text"],
            output_path=str(destination),
            use_random=False,
            verbose=False,
            max_text_tokens_per_segment=int(editor.tts.cfg.gpt.max_text_tokens) - 1,
        )
        if result != str(destination) or sf.info(destination).frames == 0:
            raise ValueError(f"IndexTTS2 did not write {destination}")
        metadata_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"TARGET {number}/{len(rows)} {row['job_id']}", flush=True)


if __name__ == "__main__":
    main()
