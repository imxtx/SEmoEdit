import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import scripts.edit as cli


class EditCliTests(unittest.TestCase):
    def test_bridging_switch_and_explicit_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "source.wav"
            sf.write(audio, np.zeros(240), 24000)
            python = root / "python"
            python.symlink_to("/usr/bin/python3")
            output = root / "edited.wav"
            arguments = [
                "edit.py",
                "--model",
                "cosyvoice2",
                "--source-audio",
                str(audio),
                "--text",
                "Source text.",
                "--target-reference-audio",
                str(audio),
                "--target-reference-text",
                "Target text.",
                "--strength",
                "0.5",
                "--output",
                str(output),
                "--donor-index",
                str(root / "index"),
                "--emotion-model",
                str(root / "emotion"),
                "--emotion-python",
                str(python),
                "--bridge-python",
                str(python),
                "--bridge-checkpoint",
                str(root / "checkpoint"),
                "--bridge-upstream-root",
                str(root / "upstream"),
            ]
            for enabled in (True, False):
                with (
                    self.subTest(enabled=enabled),
                    patch(
                        "sys.argv",
                        arguments + ([] if enabled else ["--no-emotion-bridging"]),
                    ),
                    patch.object(
                        cli,
                        "load_editor",
                        return_value=SimpleNamespace(sample_rate=24000),
                    ),
                    patch.object(cli, "EmotionBridge") as bridge,
                    patch.object(
                        cli, "edit", return_value=(torch.zeros(240), {"first_pass": []})
                    ) as edit,
                ):
                    cli.main()
                    self.assertEqual(edit.call_args.kwargs["emotion_bridging"], enabled)
                    metadata = json.loads(output.with_suffix(".json").read_text())
                    self.assertEqual(metadata["emotion_bridging"], enabled)
                    if enabled:
                        self.assertEqual(
                            edit.call_args.kwargs["bridging"], bridge.return_value
                        )
                        paths = bridge.call_args.kwargs
                        self.assertEqual(paths["index_root"], root / "index")
                        self.assertEqual(paths["emotion_python"], python)
                        self.assertEqual(paths["index_python"], python)
                        self.assertEqual(paths["index_checkpoint"], root / "checkpoint")
                    else:
                        bridge.assert_not_called()
                        self.assertIsNone(edit.call_args.kwargs["bridging"])


if __name__ == "__main__":
    unittest.main()
