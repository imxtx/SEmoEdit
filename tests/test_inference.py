import unittest
from pathlib import Path

import torch

from semoedit.inference import edit
from semoedit.transport import derived_seed


class Editor:
    sample_rate = 24000
    times = torch.tensor([0.0, 0.5, 1.0])

    def __init__(self):
        self.prepared = []

    def prepare(self, case, seed):
        self.prepared.append((case.copy(), seed))
        prefix = torch.zeros(1, 1, 1)
        return (
            torch.zeros(1, 1, 2),
            {"prefix": prefix, "velocity": 0},
            {
                "prefix": prefix,
                "velocity": 3 if case["target_reference_audio"] == "bridge.wav" else 2,
            },
        )

    def velocity(self, query, time, branch):
        return torch.full_like(query, branch["velocity"])

    def decode(self, state, seed):
        return state


class Bridge:
    def __init__(self):
        self.calls = []

    def reference(self, waveform, sample_rate, case, seed, source_index, strength):
        self.calls.append((waveform.clone(), seed, source_index, strength))
        return Path("bridge.wav"), {"donor": "selected"}


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.editor = Editor()
        self.bridge = Bridge()
        self.case = {
            "source_audio": "source.wav",
            "text": "Source text.",
            "target_reference_audio": "target.wav",
            "target_reference_text": "Different reference text.",
        }

    def test_intermediate_strength_runs_two_passes_with_shared_seeds(self):
        waveform, diagnostics = edit(
            self.editor, self.case, strength=0.5, source_index=7, bridging=self.bridge
        )
        torch.testing.assert_close(waveform, torch.full_like(waveform, 3))
        first_pass, condition_seed, source_index, strength = self.bridge.calls[0]
        torch.testing.assert_close(first_pass, torch.ones_like(first_pass))
        self.assertEqual(
            (condition_seed, source_index, strength), (derived_seed(42, 7, 0), 7, 0.5)
        )
        first_case, first_seed = self.editor.prepared[0]
        second_case, second_seed = self.editor.prepared[1]
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(first_case, self.case)
        self.assertEqual(second_case["source_audio"], self.case["source_audio"])
        self.assertEqual(second_case["target_reference_audio"], "bridge.wav")
        self.assertEqual(second_case["target_reference_text"], self.case["text"])
        self.assertEqual(set(diagnostics), {"first_pass", "bridging", "second_pass"})

    def test_opt_out_uses_direct_transport_without_bridge_dependencies(self):
        waveform, diagnostics = edit(
            self.editor, self.case, strength=0.5, emotion_bridging=False
        )
        torch.testing.assert_close(waveform, torch.ones_like(waveform))
        self.assertEqual(len(self.editor.prepared), 1)
        self.assertEqual(set(diagnostics), {"first_pass"})

    def test_endpoints_skip_bridging(self):
        for strength in (0, 1):
            with self.subTest(strength=strength):
                waveform, diagnostics = edit(
                    self.editor, self.case, strength=strength, bridging=self.bridge
                )
                torch.testing.assert_close(
                    waveform, torch.full_like(waveform, 2 * strength)
                )
                self.assertEqual(set(diagnostics), {"first_pass"})
        self.assertEqual(self.bridge.calls, [])

    def test_missing_bridge_fails_before_model_inference(self):
        with self.assertRaisesRegex(ValueError, "EmotionBridge is required"):
            edit(self.editor, self.case, strength=0.5)
        self.assertEqual(self.editor.prepared, [])

    def test_out_of_range_strength_fails_before_model_inference(self):
        for strength in (-0.1, 1.1, float("nan")):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                edit(self.editor, self.case, strength=strength)
        self.assertEqual(self.editor.prepared, [])


if __name__ == "__main__":
    unittest.main()
