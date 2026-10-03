import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semoedit.bridging import load_embedding_model, nearest_donors, utterance_embeddings


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def donor_records(corpus_root, manifest):
    records = read_jsonl(manifest)
    corpus_root = corpus_root.resolve()
    for record in records:
        record["audio"] = str(corpus_root / record["audio"])
    return records


def build(args):
    metadata_path = args.index_root / "metadata.jsonl"
    embeddings_path = args.index_root / "embeddings.npy"
    status_path = args.index_root / "status.json"
    records = donor_records(args.corpus_root, args.donor_manifest)
    if metadata_path.exists() and read_jsonl(metadata_path) != records:
        raise ValueError("Donor manifest changed; select a new --index-root")
    if not metadata_path.exists():
        write_jsonl(metadata_path, records)
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else None
    )
    if status and (
        status["model"] != args.model
        or status["corpus_root"] != str(args.corpus_root.resolve())
    ):
        raise ValueError(f"Donor index settings changed: {args.index_root}")
    completed = status["completed"] if status else 0
    if completed == len(records):
        if not embeddings_path.exists():
            raise FileNotFoundError(embeddings_path)
        print(f"INDEX READY {completed}", flush=True)
        return
    if completed and not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)
    model = load_embedding_model(args.model, args.device)
    matrix = (
        np.lib.format.open_memmap(embeddings_path, mode="r+")
        if embeddings_path.exists()
        else None
    )
    for embedding in utterance_embeddings(
        model, [row["audio"] for row in records[completed:]], args.batch_size
    ):
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(records), embedding.size),
            )
        matrix[completed] = embedding
        completed += 1
        if completed % args.batch_size == 0 or completed == len(records):
            matrix.flush()
            status_path.write_text(
                json.dumps(
                    {
                        "completed": completed,
                        "model": args.model,
                        "corpus_root": str(args.corpus_root.resolve()),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"INDEXED {completed}/{len(records)}", flush=True)


def search(args):
    records = read_jsonl(args.index_root / "metadata.jsonl")
    candidates = np.load(args.index_root / "embeddings.npy", mmap_mode="r")
    jobs = read_jsonl(args.jobs)
    model = load_embedding_model(args.model, args.device)
    queries = np.stack(
        list(
            utterance_embeddings(
                model, [row["raw_audio"] for row in jobs], args.batch_size
            )
        )
    )
    indices, scores = nearest_donors(queries, candidates)
    write_jsonl(
        args.matches,
        [
            {
                **job,
                "donor_audio": records[int(index)]["audio"],
                "donor_emotion": records[int(index)]["emotion"],
                "donor_speaker_id": records[int(index)]["speaker_id"],
                "donor_transcript": records[int(index)]["transcript"],
                "cosine_similarity": float(score),
            }
            for job, index, score in zip(jobs, indices, scores, strict=True)
        ],
    )
    print(f"MATCHED {len(jobs)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("build", "search"))
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.phase == "build":
        build(args)
    else:
        search(args)


if __name__ == "__main__":
    main()
