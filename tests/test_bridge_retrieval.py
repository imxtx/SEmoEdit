import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import scripts.bridge_retrieval as retrieval


class DonorIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus"
        (self.corpus / "nested").mkdir(parents=True)
        for name in ("z.wAv", "nested/B.WAV", "a.wav", "ignored.mp3"):
            (self.corpus / name).touch()
        self.args = SimpleNamespace(
            corpus_root=self.corpus,
            donor_manifest=None,
            index_root=self.root / "index",
            model=str(self.root / "emotion_model"),
            device="cpu",
            batch_size=1,
        )

    def test_scan_is_recursive_case_insensitive_and_sorted(self):
        records = retrieval.donor_records(self.corpus)
        self.assertEqual(
            records,
            [
                {"audio": str(self.corpus / name)}
                for name in ("a.wav", "nested/B.WAV", "z.wAv")
            ],
        )

    def test_manifest_preserves_order_and_optional_metadata(self):
        manifest = self.root / "donors.jsonl"
        records = [
            {"audio": "z.wAv", "emotion": "happy"},
            {"audio": "a.wav"},
        ]
        retrieval.write_jsonl(manifest, records)
        self.assertEqual(
            retrieval.donor_records(self.corpus, manifest),
            [
                {"audio": str(self.corpus / "z.wAv"), "emotion": "happy"},
                {"audio": str(self.corpus / "a.wav")},
            ],
        )

    def test_empty_directory_fails_before_model_loading(self):
        empty = self.root / "empty"
        empty.mkdir()
        self.args.corpus_root = empty
        with patch.object(retrieval, "load_embedding_model") as load:
            with self.assertRaisesRegex(ValueError, "No donor audio"):
                retrieval.build(self.args)
            load.assert_not_called()
        self.assertFalse(self.args.index_root.exists())

    def test_build_resumes_and_rejects_changed_file_list(self):
        vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

        def interrupted(model, paths, batch_size):
            yield vectors[0]
            raise RuntimeError("embedding extraction interrupted")

        with (
            patch.object(retrieval, "load_embedding_model"),
            patch.object(retrieval, "utterance_embeddings", side_effect=interrupted),
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                retrieval.build(self.args)
        status = json.loads((self.args.index_root / "status.json").read_text())
        self.assertEqual(status["completed"], 1)
        with (
            patch.object(retrieval, "load_embedding_model"),
            patch.object(
                retrieval, "utterance_embeddings", return_value=iter(vectors[1:])
            ) as embeddings,
        ):
            retrieval.build(self.args)
            self.assertEqual(
                embeddings.call_args.args[1],
                [str(self.corpus / "nested/B.WAV"), str(self.corpus / "z.wAv")],
            )
        np.testing.assert_array_equal(
            np.load(self.args.index_root / "embeddings.npy"), vectors
        )
        with patch.object(retrieval, "load_embedding_model") as load:
            retrieval.build(self.args)
            load.assert_not_called()
            (self.corpus / "new.wav").touch()
            with self.assertRaisesRegex(ValueError, "Donor audio list changed"):
                retrieval.build(self.args)
            load.assert_not_called()

    def test_search_accepts_audio_only_and_preserves_available_metadata(self):
        records = [
            {"audio": str(self.corpus / "a.wav")},
            {
                "audio": str(self.corpus / "z.wAv"),
                "emotion": "happy",
                "speaker_id": "speaker",
                "transcript": "Optional text.",
            },
        ]
        retrieval.write_jsonl(self.args.index_root / "metadata.jsonl", records)
        np.save(self.args.index_root / "embeddings.npy", np.eye(2, dtype=np.float32))
        self.args.jobs = self.root / "jobs.jsonl"
        self.args.matches = self.root / "matches.jsonl"
        retrieval.write_jsonl(
            self.args.jobs,
            [
                {"job_id": str(index), "raw_audio": f"query{index}.wav"}
                for index in range(2)
            ],
        )
        with (
            patch.object(retrieval, "load_embedding_model"),
            patch.object(
                retrieval, "utterance_embeddings", return_value=iter(np.eye(2))
            ),
        ):
            retrieval.search(self.args)
        first, second = retrieval.read_jsonl(self.args.matches)
        self.assertEqual(first["donor_audio"], records[0]["audio"])
        self.assertNotIn("donor_emotion", first)
        self.assertNotIn("donor_speaker_id", first)
        self.assertNotIn("donor_transcript", first)
        self.assertEqual(second["donor_emotion"], "happy")
        self.assertEqual(second["donor_speaker_id"], "speaker")
        self.assertEqual(second["donor_transcript"], "Optional text.")

    def test_build_cli_accepts_directory_without_manifest(self):
        arguments = [
            "bridge_retrieval.py",
            "build",
            "--corpus-root",
            str(self.corpus),
            "--index-root",
            str(self.args.index_root),
            "--model",
            self.args.model,
        ]
        with patch("sys.argv", arguments), patch.object(retrieval, "build") as build:
            retrieval.main()
        self.assertIsNone(build.call_args.args[0].donor_manifest)


if __name__ == "__main__":
    unittest.main()
