import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_data import load_cases, load_config, STRENGTHS
from semoedit.transport import derived_seed


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def resolve(config_dir, value):
    path = Path(value).expanduser()
    # Dereferencing .venv/bin/python would bypass the virtual environment.
    return Path(os.path.abspath(path if path.is_absolute() else config_dir / path))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_command(command, environment):
    print("RUN", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, env=environment)


def infer_command(
    python, model, model_config, config, manifest_dir, output_root, runtime
):
    command = [
        python,
        SCRIPTS / "infer.py",
        "--model",
        model,
        "--config",
        config,
        "--manifest-dir",
        manifest_dir,
        "--output-root",
        output_root,
        "--checkpoint",
        model_config["checkpoint"],
        "--upstream-root",
        model_config["upstream_root"],
        "--seed",
        runtime["seed"],
        "--tau",
        runtime["tau"],
        "--steps",
        model_config["steps"],
        "--cfg",
        model_config["cfg"],
    ]
    if model == "f5_tts":
        command += [
            "--vocoder-dir",
            model_config["vocoder"],
            "--sway",
            model_config["sway"],
        ]
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run one SEmoEdit backbone on SEmoEditBench"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference.yaml")
    parser.add_argument(
        "--model", required=True, choices=("f5_tts", "cosyvoice2", "indextts2")
    )
    parser.add_argument(
        "--task", default="all", choices=("all", "replacement", "erasure", "intensity")
    )
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    benchmark = settings["benchmark"]
    benchmark_config = resolve(config_dir, benchmark["config"])
    manifest_dir = resolve(config_dir, benchmark["manifest_dir"])
    output_root = resolve(config_dir, benchmark["output_root"])
    runtime = settings["runtime"]
    if runtime["n_avg"] != 1:
        raise ValueError("This implementation supports n_avg=1")
    for task in ("replacement", "erasure"):
        if settings["tasks"][task]["strength"] != 1:
            raise ValueError(f"Benchmark {task} strength must be 1")
    if tuple(settings["tasks"]["intensity"]["strengths"]) != STRENGTHS:
        raise ValueError(f"Benchmark intensity strengths must be {STRENGTHS}")
    if settings["tasks"]["intensity"]["second_pass_strength"] != 1:
        raise ValueError("Emotion bridging uses second-pass strength 1")
    if settings["bridging"]["synthesizer"] != "indextts2":
        raise ValueError("Emotion bridging target synthesis uses IndexTTS2")
    model_config = settings["models"][args.model].copy()
    for field in ("python", "upstream_root", "checkpoint"):
        model_config[field] = resolve(config_dir, model_config[field])
    if args.model == "f5_tts":
        model_config["vocoder"] = resolve(config_dir, model_config["vocoder"])
    benchmark_python = resolve(config_dir, runtime["benchmark_python"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(runtime["gpu"])
    system_root = output_root / f"{args.model}_semoedit_600cases"
    raw_root = system_root / "_run/raw"
    case_config = load_config(benchmark_config)
    all_cases = load_cases(manifest_dir, case_config)
    cases = all_cases
    if args.task != "all":
        cases = [case for case in cases if case["task"] == args.task]
    if args.case_id:
        missing = set(args.case_id) - {case["id"] for case in cases}
        if missing:
            parser.error(f"Unknown case IDs for selected task: {sorted(missing)}")
        cases = [case for case in cases if case["id"] in args.case_id]
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        cases = cases[: args.max_cases]
    print(
        json.dumps(
            {
                "model": args.model,
                "task": args.task,
                "cases": len(cases),
                "output_root": str(system_root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    stage_one = infer_command(
        model_config["python"],
        args.model,
        model_config,
        benchmark_config,
        manifest_dir,
        raw_root,
        runtime,
    )
    stage_one += ["--task", args.task]
    for case_id in args.case_id or ():
        stage_one += ["--case-id", case_id]
    if args.max_cases is not None:
        stage_one += ["--max-cases", args.max_cases]
    if args.validate:
        stage_one += ["--validate"]
    if args.dry_run:
        stage_one[0] = sys.executable
        run_command(stage_one + ["--dry-run"], environment)
        return

    settings_path = system_root / "_run/inference.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = config_path.read_text(encoding="utf-8")
    if (
        settings_path.exists()
        and settings_path.read_text(encoding="utf-8") != config_text
    ):
        raise ValueError(f"Configuration changed for {system_root}")
    settings_path.write_text(config_text, encoding="utf-8")
    run_command(stage_one, environment)

    for case in cases:
        if case["task"] == "intensity":
            continue
        source = raw_root / case["split"] / f"{case['id']}.wav"
        destination = system_root / case["split"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    intensity_cases = [case for case in cases if case["task"] == "intensity"]
    if not intensity_cases:
        return

    bridge = settings["bridging"]
    index_root = resolve(config_dir, bridge["index_root"])
    donor_corpus = resolve(config_dir, bridge["donor_corpus"])
    bridge_root = system_root / "_run/bridging"
    jobs_path = bridge_root / "jobs.jsonl"
    matches_path = bridge_root / "matches.jsonl"
    jobs = []
    for case in intensity_cases:
        metadata = json.loads(
            (raw_root / "_run" / case["split"] / f"{case['id']}.json").read_text(
                encoding="utf-8"
            )
        )
        for strength in STRENGTHS[1:]:
            label = f"{strength:g}"
            jobs.append(
                {
                    "job_id": f"{case['id']}__alpha_{label.replace('.', 'p')}",
                    "case_id": case["id"],
                    "split": case["split"],
                    "strength": strength,
                    "strength_label": label,
                    "raw_audio": str(
                        raw_root / case["split"] / case["id"] / f"{label}.wav"
                    ),
                    "source_audio": case["source_audio"],
                    "text": case["text"],
                    "condition_seed": metadata["condition_seed"],
                    "edit_seed": metadata["edit_seed"],
                    "vocoder_seed": metadata["vocoder_seed"],
                }
            )
    write_jsonl(jobs_path, jobs)
    retrieval = [
        benchmark_python,
        SCRIPTS / "bridge_retrieval.py",
        "--index-root",
        index_root,
        "--model",
        resolve(config_dir, bridge["emotion2vec_model"]),
        "--device",
        "cuda:0",
        "--batch-size",
        bridge["batch_size"],
    ]
    if bridge["donor_manifest"] is not None:
        retrieval += ["--donor-manifest", resolve(config_dir, bridge["donor_manifest"])]
    run_command(retrieval + ["build", "--corpus-root", donor_corpus], environment)
    run_command(
        retrieval + ["search", "--jobs", jobs_path, "--matches", matches_path],
        environment,
    )

    synthesizer = settings["models"]["indextts2"]
    target_root = bridge_root / "targets"
    run_command(
        [
            resolve(config_dir, synthesizer["python"]),
            SCRIPTS / "bridge_targets.py",
            "--matches",
            matches_path,
            "--output-root",
            target_root,
            "--upstream-root",
            resolve(config_dir, synthesizer["upstream_root"]),
            "--checkpoint",
            resolve(config_dir, synthesizer["checkpoint"]),
            "--steps",
            synthesizer["steps"],
            "--cfg",
            synthesizer["cfg"],
        ],
        environment,
    )
    bridge_manifests = bridge_root / "manifests"
    grouped = {}
    for case in all_cases:
        if case["task"] != "intensity":
            continue
        for strength in STRENGTHS[1:]:
            label = f"{strength:g}"
            job_id = f"{case['id']}__alpha_{label.replace('.', 'p')}"
            target = target_root / case["split"] / f"{job_id}.wav"
            grouped.setdefault(case["split"], []).append(
                {
                    "id": job_id,
                    "task": "replacement",
                    "dataset": case["dataset"],
                    "text": case["text"],
                    "source_audio": case["source_audio"],
                    "target_reference_audio": str(target.resolve()),
                    "target_reference_dataset": case["dataset"],
                    "target_reference_text": case["text"],
                    "source_emotion": case["source_emotion"],
                    "target_emotion": case["target_emotion"],
                    "condition_seed": case.get(
                        "condition_seed",
                        derived_seed(runtime["seed"], case["source_index"], 0),
                    ),
                    "edit_seed": case.get(
                        "edit_seed",
                        derived_seed(runtime["seed"], case["source_index"], 1),
                    ),
                    "vocoder_seed": case.get(
                        "vocoder_seed",
                        derived_seed(runtime["seed"], case["source_index"], 2),
                    ),
                }
            )
    for split, rows in grouped.items():
        write_jsonl(bridge_manifests / f"{split}.jsonl", rows)
    results_root = bridge_root / "results"
    stage_two = infer_command(
        model_config["python"],
        args.model,
        model_config,
        benchmark_config,
        bridge_manifests,
        results_root,
        runtime,
    )
    if args.validate:
        stage_two += ["--validate"]
    selected_ids_path = bridge_root / "selected_ids.txt"
    selected_ids_path.write_text(
        "".join(f"{job['job_id']}\n" for job in jobs), encoding="utf-8"
    )
    stage_two += ["--case-id-file", selected_ids_path]
    run_command(stage_two + ["--task", "replacement"], environment)

    for case in intensity_cases:
        split = case["split"]
        case_id = case["id"]
        directory = system_root / split / case_id
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_root / split / case_id / "0.wav", directory / "0.wav")
        for strength in STRENGTHS[1:]:
            label = f"{strength:g}"
            job_id = f"{case_id}__alpha_{label.replace('.', 'p')}"
            shutil.copyfile(
                results_root / split / f"{job_id}.wav", directory / f"{label}.wav"
            )
        shutil.copyfile(directory / "1.wav", system_root / split / f"{case_id}.wav")
    print(f"DONE {args.model}: {len(cases)} cases", flush=True)


if __name__ == "__main__":
    main()
