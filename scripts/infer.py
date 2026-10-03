"""Generate source speech, then apply target-length SEmoEdit."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_data import (
    BENCHMARK,
    load_cases,
    load_config,
    output_paths,
    prepare_inputs,
)


def write_json(path, value):
    """Atomically save generation metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_wave(path, waveform, sample_rate):
    """Validate and atomically save a generated waveform."""
    samples = waveform.detach().float().cpu().numpy().reshape(-1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Invalid generated waveform: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.wav")
    sf.write(temporary, samples, sample_rate, subtype="FLOAT")
    temporary.replace(path)


def main():
    """Run SEmoEdit inference over selected benchmark cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("f5_tts", "cosyvoice2", "indextts2")
    )
    parser.add_argument(
        "--task", choices=("all", "replacement", "erasure", "intensity"), default="all"
    )
    parser.add_argument(
        "--config", type=Path, default=BENCHMARK / "configs/benchmark.yaml"
    )
    parser.add_argument("--manifest-dir", type=Path, default=BENCHMARK / "manifests600")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument("--sway", type=float)
    parser.add_argument("--instruction-language", choices=("en", "zh"))
    parser.add_argument(
        "--vocoder-dir",
        type=Path,
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Repeat to select multiple splits; default: all nine",
    )
    parser.add_argument(
        "--case-id", action="append", help="Repeat to select exact manifest IDs"
    )
    parser.add_argument("--case-id-file", type=Path)
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit total selected cases, not steps or strength levels",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs and print counts without loading models or writing outputs",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run alpha=0 and same-condition network checks; save source reconstruction",
    )
    args = parser.parse_args()
    if args.instruction_language and args.model == "f5_tts":
        parser.error("Text-conditioned SEmoEdit supports only cosyvoice2 and indextts2")
    if args.model == "f5_tts" and (args.vocoder_dir is None or args.sway is None):
        parser.error("F5-TTS requires --vocoder-dir and --sway")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    config = load_config(args.config.resolve())
    cases = load_cases(args.manifest_dir, config, args.instruction_language)
    if args.task != "all":
        cases = [case for case in cases if case["task"] == args.task]
    available = {case["split"] for case in cases}
    if args.split and set(args.split) - available:
        parser.error(f"Unknown splits: {set(args.split) - available}")
    if args.split:
        cases = [case for case in cases if case["split"] in args.split]
    selected_ids = set(args.case_id or ())
    if args.case_id_file:
        selected_ids.update(args.case_id_file.read_text(encoding="utf-8").splitlines())
    if selected_ids:
        missing = selected_ids - {case["id"] for case in cases}
        if missing:
            parser.error(f"Unknown case IDs in selected splits: {missing}")
        cases = [case for case in cases if case["id"] in selected_ids]
    if args.max_cases:
        cases = cases[: args.max_cases]
    prepare_inputs(cases, config)
    output_root = args.output_root.resolve()
    print(
        json.dumps(
            {
                "model": args.model,
                "instruction_language": args.instruction_language,
                "cases": len(cases),
                "splits": dict(Counter(case["split"] for case in cases)),
                "trajectories": sum(
                    5 if case["task"] == "intensity" else 1 for case in cases
                ),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    import torch
    from semoedit.inference import edit_prepared, load_editor
    from semoedit.transport import derived_seed, integrate

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; select an available GPU with CUDA_VISIBLE_DEVICES"
        )
    checkpoint = args.checkpoint
    tau = args.tau
    run_dir = output_root / "_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = {
            "model": args.model,
            "checkpoint": str(checkpoint.resolve()),
            "upstream_root": str(args.upstream_root.resolve()),
            "seed": args.seed,
            "vocoder_dir": (
                str(args.vocoder_dir.resolve()) if args.model == "f5_tts" else None
            ),
            "steps": args.steps,
            "cfg": args.cfg,
            "sway": args.sway,
            "n_avg": 1,
            "tau": tau,
            "method": "generated_source_target_length_v1",
            "config": config,
        }
        settings_path = run_dir / "settings.json"
        if args.instruction_language:
            settings.update(
                method="generated_source_target_length_instruct_v1",
                instruction_language=args.instruction_language,
                speaker_reference="source_audio",
                native_interface=(
                    "inference_instruct2"
                    if args.model == "cosyvoice2"
                    else "infer(use_emo_text=True)"
                ),
                use_qwen_emo=args.model == "indextts2",
            )
        if (
            settings_path.exists()
            and json.loads(settings_path.read_text(encoding="utf-8")) != settings
        ):
            raise ValueError("Run settings changed; select a new --output-root")
        write_json(settings_path, settings)
        # Keep the exact case definitions alongside outputs for resumable runs.
        for path in sorted(args.manifest_dir.glob("*.jsonl")):
            saved = run_dir / "manifests" / path.name
            if saved.exists() and saved.read_text(encoding="utf-8") != path.read_text(
                encoding="utf-8"
            ):
                raise ValueError(
                    f"Manifest changed: {path}; select a new --output-root"
                )
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, saved)
        editor = load_editor(
            args.model,
            checkpoint,
            args.upstream_root,
            args.steps,
            args.cfg,
            vocoder=args.vocoder_dir,
            sway=args.sway,
            instruct=bool(args.instruction_language),
        )
        write_json(
            run_dir / "runtime.json",
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "dtype": str(editor.dtype),
                "time_grid": editor.times.cpu().tolist(),
                "sample_rate": editor.sample_rate,
            },
        )
        for index, case in enumerate(cases, start=1):
            paths, ordinary = output_paths(output_root, case)
            metadata_path = run_dir / case["split"] / f"{case['id']}.json"
            if metadata_path.exists():
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                if previous["case"] != case:
                    raise ValueError(
                        f"Input changed for {case['id']}; select a new --output-root"
                    )
                if previous["status"] == "complete" and (
                    not args.validate or previous["validated"]
                ):
                    artifacts = metadata_path.with_suffix("")
                    required = [
                        *paths.values(),
                        ordinary,
                        artifacts / "source_native.wav",
                        artifacts / "before.wav",
                        artifacts / "source.pt",
                    ]
                    if all(path.is_file() for path in required):
                        print(f"SKIP {index}/{len(cases)} {case['id']}", flush=True)
                        continue
            condition_seed = case.get(
                "condition_seed", derived_seed(args.seed, case["source_index"], 0)
            )
            edit_seed = case.get(
                "edit_seed", derived_seed(args.seed, case["source_index"], 1)
            )
            vocoder_seed = case.get(
                "vocoder_seed", derived_seed(args.seed, case["source_index"], 2)
            )
            metadata = {
                "case": case,
                "status": "running",
                "validated": False,
                "condition_seed": condition_seed,
                "edit_seed": edit_seed,
                "vocoder_seed": vocoder_seed,
            }
            write_json(metadata_path, metadata)
            print(f"START {index}/{len(cases)} {case['id']}", flush=True)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            try:
                source, source_branch, target_branch = editor.prepare(
                    case, condition_seed
                )
                if args.instruction_language:
                    metadata["instructions"] = {
                        "language": args.instruction_language,
                        "source": editor.source["instruction"],
                        "target": editor.target["instruction"],
                        "speaker_reference": case["source_audio"],
                    }
                    if args.model == "indextts2":
                        metadata["emotion_vectors"] = {
                            "source": editor.source["emotion_vector"],
                            "target": editor.target["emotion_vector"],
                            "parser_path": str(
                                checkpoint / editor.tts.cfg.qwen_emo_path
                            ),
                        }
                metadata.update(
                    source_frames=source.shape[-1],
                    source_prefix_frames=source_branch["prefix"].shape[-1],
                    target_prefix_frames=target_branch["prefix"].shape[-1],
                )
                native_source = editor.source["full"][..., editor.source["start"] :]
                metadata.update(
                    source_generation_frames=native_source.shape[-1],
                    target_generation_frames=source.shape[-1],
                    alignment_ratio=source.shape[-1] / native_source.shape[-1],
                    time_grid=editor.times.cpu().tolist(),
                )
                artifacts = metadata_path.with_suffix("")
                artifacts.mkdir(parents=True, exist_ok=True)
                save_wave(
                    artifacts / "source_native.wav",
                    editor.decode(native_source, vocoder_seed),
                    editor.sample_rate,
                )
                save_wave(
                    artifacts / "before.wav",
                    editor.decode(source, vocoder_seed),
                    editor.sample_rate,
                )
                torch.save(
                    {
                        "source_native_gen": native_source.cpu(),
                        "source_aligned_gen": source.cpu(),
                    },
                    artifacts / "source.pt",
                )
                metadata["before_audio"] = str(artifacts / "before.wav")
                metadata["source_native_audio"] = str(artifacts / "source_native.wav")
                if args.validate:
                    if args.instruction_language:
                        if args.model == "cosyvoice2":
                            repeated = editor.capture(
                                case["source_audio"],
                                case["source_instruction"],
                                case["text"],
                                condition_seed,
                            )
                            source_mu = source_branch["mu"][
                                ..., source_branch["prefix"].shape[-1] :
                            ]
                            target_mu = target_branch["mu"][
                                ..., target_branch["prefix"].shape[-1] :
                            ]
                        else:
                            repeated = editor.capture(
                                case["source_audio"],
                                case["text"],
                                condition_seed,
                                case["source_instruction"],
                            )
                            if (
                                repeated["emotion_vector"]
                                != editor.source["emotion_vector"]
                            ):
                                raise ValueError(
                                    "Same emotion text produced different emotion vectors"
                                )
                            source_mu = source_branch["mu"][
                                :, source_branch["prefix"].shape[-1] :
                            ]
                            target_mu = target_branch["mu"][
                                :, target_branch["prefix"].shape[-1] :
                            ]
                        torch.testing.assert_close(
                            repeated["mu"], editor.source["mu"], atol=1e-5, rtol=0
                        )
                        torch.testing.assert_close(
                            repeated["full"], editor.source["full"], atol=1e-5, rtol=0
                        )
                        metadata["same_instruction_generation_max_error"] = float(
                            (repeated["full"] - editor.source["full"]).abs().max()
                        )
                        metadata["condition_mu_max_difference"] = float(
                            (target_mu - source_mu).abs().max()
                        )
                        del repeated
                    noop, _ = integrate(
                        source,
                        source_branch,
                        source_branch,
                        editor.times,
                        editor.velocity,
                        edit_seed,
                        1,
                        tau,
                    )
                    zero, _ = integrate(
                        source,
                        source_branch,
                        target_branch,
                        editor.times,
                        editor.velocity,
                        edit_seed,
                        0,
                        tau,
                    )
                    tolerance = 1e-3 if source.dtype == torch.float16 else 1e-5
                    torch.testing.assert_close(noop, source, atol=tolerance, rtol=0)
                    torch.testing.assert_close(zero, source, atol=0, rtol=0)
                    metadata["noop_max_error"] = float((noop - source).abs().max())
                    metadata["validated"] = True
                    save_wave(
                        metadata_path.with_suffix(".source_reconstruction.wav"),
                        editor.decode(source, vocoder_seed),
                        editor.sample_rate,
                    )
                metadata["strengths"] = {}
                for strength, path in paths.items():
                    waveform, steps = edit_prepared(
                        editor,
                        source,
                        source_branch,
                        target_branch,
                        edit_seed,
                        strength,
                        tau,
                        vocoder_seed,
                    )
                    save_wave(path, waveform, editor.sample_rate)
                    metadata["strengths"][f"{strength:g}"] = {
                        "wav": str(path),
                        "steps": steps,
                    }
                    print(f"WAV alpha={strength:g} {path}", flush=True)
                if case["task"] == "intensity":
                    # The ordinary path is the alpha=1 waveform used by shared metrics.
                    shutil.copyfile(paths[1.0], ordinary)
                torch.cuda.synchronize()
                metadata.update(
                    status="complete",
                    seconds=time.monotonic() - started,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                )
            except Exception as error:
                metadata.update(
                    status="failed", error=f"{type(error).__name__}: {error}"
                )
                write_json(metadata_path, metadata)
                raise
            write_json(metadata_path, metadata)
            print(f"DONE {case['id']} {metadata['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
