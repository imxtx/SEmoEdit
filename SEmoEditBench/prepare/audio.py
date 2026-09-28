from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config_dir = path.resolve().parent
    for section in ("dataset_roots", "raw_dataset_roots"):
        for dataset, value in config[section].items():
            root = Path(value).expanduser()
            config[section][dataset] = str(
                (root if root.is_absolute() else config_dir / root).resolve()
            )
    for name in ("root", "checkpoint_dir", "config"):
        model_path = Path(config["indextts2"][name]).expanduser()
        config["indextts2"][name] = str(
            (model_path if model_path.is_absolute() else config_dir / model_path).resolve()
        )
    return config


def prepare_audio(
    config: dict[str, Any], manifests: dict[str, list[dict[str, Any]]], *, dry_run: bool
) -> dict[str, int]:
    copies: dict[Path, Path] = {}
    syntheses: dict[Path, dict[str, str]] = {}
    existing: set[Path] = set()
    for records in manifests.values():
        for record in records:
            for field in (
                "source_audio",
                "target_audio",
                "target_reference_audio",
                "target_reference_doner",
            ):
                value = record.get(field)
                if value is None:
                    continue
                if field == "target_reference_audio":
                    dataset = record["target_reference_dataset"]
                elif field == "target_reference_doner":
                    dataset = record["target_reference_doner_dataset"]
                else:
                    dataset = record["dataset"]
                destination = Path(config["dataset_roots"][dataset]) / value
                if destination in existing or destination in copies or destination in syntheses:
                    continue
                if destination.is_file():
                    existing.add(destination)
                    continue
                source = Path(config["raw_dataset_roots"][dataset]) / value
                if source.is_file():
                    copies[destination] = source
                    continue
                if field == "target_reference_audio" and record.get("target_reference_doner"):
                    reference = Path(value)
                    donor = Path(record["target_reference_doner"])
                    source_name = reference.stem.removeprefix(f"{donor.stem}_vc_")
                    raw_root = Path(config["raw_dataset_roots"][dataset])
                    source_audio = raw_root / donor.with_name(source_name + reference.suffix)
                    donor_audio = raw_root / donor
                    if not source_audio.is_file():
                        raise FileNotFoundError(source_audio)
                    if not donor_audio.is_file():
                        raise FileNotFoundError(donor_audio)
                    syntheses[destination] = {
                        "id": record["id"],
                        "source_audio": str(source_audio),
                        "donor_audio": str(donor_audio),
                        "text": record["target_reference_text"],
                        "output_path": str(destination),
                    }
                    continue
                raise FileNotFoundError(source)

    if not dry_run:
        for destination, source in tqdm(
            copies.items(), total=len(copies), desc="Copy audio", disable=not copies
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if syntheses:
            settings = config["indextts2"]
            indextts_root = Path(settings["root"])
            worker = Path(__file__).with_name("indextts_worker.py").resolve()
            with tempfile.TemporaryDirectory() as temporary:
                jobs_path = Path(temporary) / "jobs.jsonl"
                with jobs_path.open("w", encoding="utf-8") as handle:
                    for job in syntheses.values():
                        handle.write(json.dumps(job, ensure_ascii=False) + "\n")
                environment = os.environ.copy()
                environment["MPLCONFIGDIR"] = str(
                    Path("/tmp/emoeditbench-matplotlib").resolve()
                )
                environment["NUMBA_CACHE_DIR"] = str(
                    Path("/tmp/emoeditbench-numba").resolve()
                )
                subprocess.run(
                    [
                        str(indextts_root / ".venv/bin/python"),
                        str(worker),
                        "--jobs",
                        str(jobs_path),
                        "--checkpoint-dir",
                        str(settings["checkpoint_dir"]),
                        "--config",
                        str(settings["config"]),
                        "--device",
                        settings["device"],
                        "--use-fp16" if settings["use_fp16"] else "--no-use-fp16",
                        "--use-cuda-kernel"
                        if settings["use_cuda_kernel"]
                        else "--no-use-cuda-kernel",
                        "--use-deepspeed"
                        if settings["use_deepspeed"]
                        else "--no-use-deepspeed",
                        "--use-qwen-emo"
                        if settings["use_qwen_emo"]
                        else "--no-use-qwen-emo",
                    ],
                    cwd=indextts_root,
                    env=environment,
                    check=True,
                )
        for destination in (*copies, *syntheses):
            if not destination.is_file():
                raise FileNotFoundError(destination)

    return {
        "cases": sum(map(len, manifests.values())),
        "existing_audio": len(existing),
        "copied_audio": len(copies),
        "synthesized_references": len(syntheses),
    }
