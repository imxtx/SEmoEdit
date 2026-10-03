from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line options for CREMA-D reference synthesis."""
    parser = argparse.ArgumentParser(
        description="Generate CREMA-D VC references with IndexTTS2."
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--use-fp16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-cuda-kernel", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-deepspeed", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use-qwen-emo", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def read_jobs(path: Path) -> list[dict[str, str]]:
    """Read prepared synthesis jobs from a JSON Lines file."""
    jobs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                jobs.append(json.loads(line))
    return jobs


def main() -> None:
    """Run CREMA-D reference synthesis from command-line arguments."""
    args = parse_args()
    from indextts.infer_v2 import IndexTTS2

    cache = args.checkpoint_dir.resolve() / "hf_cache"
    tts = IndexTTS2(
        cfg_path=str(args.config.resolve()),
        model_dir=str(args.checkpoint_dir.resolve()),
        use_fp16=args.use_fp16,
        device=args.device,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        use_qwen_emo=args.use_qwen_emo,
        aux_paths={
            "w2v_bert": str(cache / "w2v-bert-2.0"),
            "semantic_codec": str(cache / "semantic_codec/model.safetensors"),
            "campplus": str(cache / "campplus_cn_common.bin"),
            "bigvgan": str(cache / "bigvgan"),
        },
    )
    jobs = read_jobs(args.jobs)
    for job in tqdm(jobs, desc="Generate CREMA-D references"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        destination = Path(job["output_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        tts.infer(
            spk_audio_prompt=job["source_audio"],
            text=job["text"],
            output_path=str(destination),
            emo_audio_prompt=job["donor_audio"],
            verbose=False,
        )


if __name__ == "__main__":
    main()
