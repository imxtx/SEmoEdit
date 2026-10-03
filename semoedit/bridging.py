import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def load_embedding_model(model, device):
    """Load Emotion2Vec only when emotion bridging is requested."""
    from funasr import AutoModel

    return AutoModel(model=model, device=device, hub="hf", disable_update=True)


def utterance_embeddings(model, paths, batch_size):
    """Extract one Emotion2Vec embedding per audio file."""
    for start in range(0, len(paths), batch_size):
        results = model.generate(
            paths[start : start + batch_size],
            granularity="utterance",
            extract_embedding=True,
            batch_size_s=300,
        )
        if len(results) != len(paths[start : start + batch_size]):
            raise ValueError("Emotion2Vec returned an unexpected number of results")
        for result in results:
            embedding = np.asarray(result["feats"], dtype=np.float32)
            yield embedding.mean(axis=0) if embedding.ndim > 1 else embedding


def nearest_donors(query_embeddings, candidate_embeddings):
    candidates = np.array(candidate_embeddings, dtype=np.float32, copy=True)
    queries = np.array(query_embeddings, dtype=np.float32, copy=True)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    similarities = candidates @ queries.T
    indices = similarities.argmax(axis=0)
    scores = similarities[indices, np.arange(len(indices))]
    return indices, scores


class EmotionBridge:
    """Retrieve a speech donor and synthesize a reference in separate environments."""

    def __init__(
        self,
        *,
        index_root,
        emotion2vec_model,
        emotion_python,
        index_python,
        index_checkpoint,
        index_upstream_root,
        work_dir,
        device="cuda:0",
    ):
        self.index_root = Path(index_root).resolve()
        self.emotion2vec_model = Path(emotion2vec_model).resolve()
        # Preserve virtual-environment executables instead of resolving symlinks.
        self.emotion_python = os.path.abspath(emotion_python)
        self.index_python = os.path.abspath(index_python)
        self.index_checkpoint = Path(index_checkpoint).resolve()
        self.index_upstream_root = Path(index_upstream_root).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.device = device

    def _run(self, python, phase, *arguments):
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = os.pathsep.join(
            [package_root, environment.get("PYTHONPATH", "")]
        )
        subprocess.run(
            [python, "-m", "semoedit.bridging", phase, *map(str, arguments)],
            env=environment,
            check=True,
        )

    def reference(self, waveform, sample_rate, case, seed, source_index, strength):
        directory = self.work_dir / f"{source_index:05d}_alpha_{strength:g}"
        directory.mkdir(parents=True, exist_ok=True)
        first_pass = directory / "first_pass.wav"
        sf.write(
            first_pass,
            waveform.detach().float().cpu().numpy().reshape(-1),
            sample_rate,
            subtype="FLOAT",
        )
        match_path = directory / "donor.json"
        self._run(
            self.emotion_python,
            "retrieve",
            "--index-root",
            self.index_root,
            "--model",
            self.emotion2vec_model,
            "--device",
            self.device,
            "--audio",
            first_pass,
            "--output",
            match_path,
        )
        match = json.loads(match_path.read_text(encoding="utf-8"))
        reference = directory / "reference.wav"
        self._run(
            self.index_python,
            "synthesize",
            "--checkpoint",
            self.index_checkpoint,
            "--upstream-root",
            self.index_upstream_root,
            "--source-audio",
            Path(case["source_audio"]).resolve(),
            "--donor-audio",
            match["audio"],
            "--text",
            case["text"],
            "--seed",
            seed,
            "--output",
            reference,
        )
        return reference, {
            "donor": match,
            "index_root": str(self.index_root),
            "emotion2vec_model": str(self.emotion2vec_model),
            "index_checkpoint": str(self.index_checkpoint),
            "index_upstream_root": str(self.index_upstream_root),
            "condition_seed": seed,
            "first_pass_audio": str(first_pass),
            "reference_audio": str(reference),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    retrieve = phases.add_parser("retrieve")
    retrieve.add_argument("--index-root", type=Path, required=True)
    retrieve.add_argument("--model", required=True)
    retrieve.add_argument("--device", default="cuda:0")
    retrieve.add_argument("--audio", type=Path, required=True)
    retrieve.add_argument("--output", type=Path, required=True)
    synthesize = phases.add_parser("synthesize")
    synthesize.add_argument("--checkpoint", type=Path, required=True)
    synthesize.add_argument("--upstream-root", type=Path, required=True)
    synthesize.add_argument("--source-audio", required=True)
    synthesize.add_argument("--donor-audio", required=True)
    synthesize.add_argument("--text", required=True)
    synthesize.add_argument("--seed", type=int, required=True)
    synthesize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "retrieve":
        records = [
            json.loads(line)
            for line in (args.index_root / "metadata.jsonl").read_text().splitlines()
            if line.strip()
        ]
        status = json.loads((args.index_root / "status.json").read_text())
        if status["completed"] != len(records):
            raise ValueError("Donor index is incomplete; finish building it first")
        if Path(status["model"]).resolve() != Path(args.model).resolve():
            raise ValueError("Emotion2Vec checkpoint differs from the donor index")
        candidates = np.load(args.index_root / "embeddings.npy", mmap_mode="r")
        if candidates.shape[0] != len(records):
            raise ValueError("Donor metadata and embedding counts differ")
        model = load_embedding_model(args.model, args.device)
        query = np.stack(list(utterance_embeddings(model, [str(args.audio)], 1)))
        indices, scores = nearest_donors(query, candidates)
        args.output.write_text(
            json.dumps(
                {**records[int(indices[0])], "cosine_similarity": float(scores[0])},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        from .adapters.common import seed_all
        from .adapters.indextts2 import IndexEditor

        editor = IndexEditor(args.checkpoint, args.upstream_root, 25, 0.7)
        seed_all(args.seed)
        result = editor.tts.infer(
            spk_audio_prompt=args.source_audio,
            emo_audio_prompt=args.donor_audio,
            emo_alpha=1.0,
            text=args.text,
            output_path=str(args.output),
            use_random=False,
            verbose=False,
            max_text_tokens_per_segment=int(editor.tts.cfg.gpt.max_text_tokens) - 1,
        )
        if result != str(args.output) or sf.info(args.output).frames == 0:
            raise ValueError(f"IndexTTS2 did not write {args.output}")


if __name__ == "__main__":
    main()
