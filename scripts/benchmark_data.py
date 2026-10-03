"""Frozen benchmark inputs and reference transcripts; no evaluation targets."""

import json
import re
from functools import lru_cache
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "SEmoEditBench"
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_config(path):
    """Load configuration and resolve relative dataset paths."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("dataset_roots", "raw_dataset_roots"):
        config[section] = {
            dataset: str((path.parent / value).resolve())
            for dataset, value in config[section].items()
        }
    return config


@lru_cache(maxsize=None)
def transcripts(path, dataset):
    """Load source-corpus transcripts by utterance ID."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if dataset == "ESD":
            fields = line.split("\t")
            if len(fields) >= 2:
                result[fields[0]] = fields[1].strip()
        else:
            match = re.match(r"^(\S+)\s+\[[^]]+\]:\s*(.*)$", line)
            if match:
                result[match[1]] = match[2].strip()
    return result


def reference_text(record, reference, config):
    """Find the target reference transcript from manifest or corpus."""
    if record.get("target_reference_text"):
        return record["target_reference_text"], "manifest.target_reference_text"
    dataset = record["target_reference_dataset"]
    raw_root = Path(config["raw_dataset_roots"][dataset])
    if dataset == "ESD":
        speaker = reference.stem.split("_")[0]
        path = raw_root / speaker / f"{speaker}.txt"
        return transcripts(path, dataset)[reference.stem], str(path)
    if dataset == "IEMOCAP":
        session = f"Session{int(reference.stem[3:5])}"
        dialog = reference.stem.rsplit("_", 1)[0]
        path = raw_root / session / "dialog/transcriptions" / f"{dialog}.txt"
        return transcripts(path, dataset)[reference.stem], str(path)
    if dataset == "RAVDESS":
        texts = {
            "01": "Kids are talking by the door.",
            "02": "Dogs are sitting by the door.",
        }
        return texts[reference.stem.split("-")[4]], "RAVDESS statement code"
    raise ValueError(
        f"{record['id']}: {dataset} reference requires target_reference_text"
    )


def load_cases(manifest_dir, config, instruction_language=None):
    """Select supported manifest cases and resolve their audio paths."""
    cases = []
    source_keys = {}
    for path in sorted(manifest_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source = (
                Path(config["dataset_roots"][record["dataset"]])
                / record["source_audio"]
            ).resolve()
            source_keys[str(source)] = (record["dataset"], record["source_audio"])
            # Text conditioning changes emotion instructions, not speaker audio.
            reference = (
                source
                if instruction_language
                else (
                    Path(config["dataset_roots"][record["target_reference_dataset"]])
                    / record["target_reference_audio"]
                ).resolve()
            )
            cases.append(
                {
                    "split": path.stem,
                    "id": record["id"],
                    "task": record["task"],
                    "dataset": record["dataset"],
                    "text": record["text"],
                    "source_audio": str(source),
                    "target_reference_audio": str(reference),
                    "target_reference_dataset": record["target_reference_dataset"],
                    "target_reference_text": record.get("target_reference_text"),
                    "source_emotion": record["source_emotion"],
                    "target_emotion": record["target_emotion"],
                    **{
                        key: record[key]
                        for key in ("condition_seed", "edit_seed", "vocoder_seed")
                        if key in record
                    },
                }
            )
            if instruction_language:
                for language in ("en", "zh"):
                    for branch in ("src", "tgt"):
                        field = f"{branch}_emo_description_{language}"
                        if (
                            not isinstance(record.get(field), str)
                            or not record[field].strip()
                        ):
                            raise ValueError(
                                f"{record['id']}: missing or empty {field}"
                            )
                cases[-1].update(
                    instruction_language=instruction_language,
                    source_instruction=record[
                        f"src_emo_description_{instruction_language}"
                    ],
                    target_instruction=record[
                        f"tgt_emo_description_{instruction_language}"
                    ],
                    source_instruction_field=f"src_emo_description_{instruction_language}",
                    target_instruction_field=f"tgt_emo_description_{instruction_language}",
                )
    if not cases:
        raise ValueError(f"No cases in {manifest_dir}")
    sources = sorted(source_keys, key=source_keys.__getitem__)
    indices = {source: index for index, source in enumerate(sources)}
    for case in cases:
        case["source_index"] = indices[case["source_audio"]]
    return cases


def prepare_inputs(cases, config):
    """Validate audio inputs and fill missing reference transcripts."""
    for case in cases:
        instruct = "instruction_language" in case
        for field in (
            ("source_audio",)
            if instruct
            else ("source_audio", "target_reference_audio")
        ):
            info = sf.info(case[field])
            if info.frames == 0 or info.duration > 30:
                raise ValueError(
                    f"{case['id']}: unsupported {field} duration {info.duration}"
                )
        if instruct:
            if not case["text"].strip():
                raise ValueError(f"{case['id']}: empty transcript")
            continue
        text, origin = reference_text(
            case, Path(case["target_reference_audio"]), config
        )
        if not text.strip() or not case["text"].strip():
            raise ValueError(f"{case['id']}: empty transcript")
        case["target_reference_text"] = text
        case["reference_text_origin"] = origin


def output_paths(root, case):
    """Use the evaluator's per-strength layout for intensity cases."""
    directory = root / case["split"]
    ordinary = directory / f"{case['id']}.wav"
    if case["task"] == "intensity":
        return {
            strength: directory / case["id"] / f"{strength:g}.wav"
            for strength in STRENGTHS
        }, ordinary
    return {1.0: ordinary}, ordinary
