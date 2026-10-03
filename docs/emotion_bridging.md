# Emotion bridging

Run all commands below from the SEmoEdit repository root.

For intermediate strengths (`0 < strength < 1`), single-sample inference uses Emotion2Vec donor retrieval, native IndexTTS2 reference synthesis, and a second edit at strength `1`. Strengths `0` and `1` use direct transport. Pass `--no-emotion-bridging` on the CLI or `emotion_bridging=False` in Python to use direct transport at intermediate strengths.

## Environment and donor index

Prepare the IndexTTS2 environment, its main and auxiliary weights, and `checkpoints/emotion2vec_plus_large` using the [installation guide](../README.md#installation) and [fixed download instructions](../third_party/README.md#fixed-model-downloads), even when editing with F5-TTS or CosyVoice2. Install the separate Emotion2Vec environment and build a donor index once. For paper reproduction, use the fixed ESD list below (replace `/path/to/ESD` with the complete original ESD corpus, containing all 35,000 utterances); for other datasets, follow [custom donor data](#custom-donor-data):

```bash
uv sync --project SEmoEditBench --locked
SEmoEditBench/.venv/bin/python scripts/bridge_retrieval.py build \
  --donor-manifest configs/bridging_donors.jsonl \
  --corpus-root /path/to/ESD \
  --index-root SEmoEditBench/outputs/_emotion2vec_esd_index \
  --model checkpoints/emotion2vec_plus_large \
  --device cuda:0 \
  --batch-size 32
```

Keep the original donor audio available after indexing: IndexTTS2 reads the selected donor WAV when synthesizing a bridge reference. Index construction resumes after interruption. Reuse the index for all three backbones; keep its Emotion2Vec checkpoint and corpus location fixed, or build a new index directory when changing them. Single-sample bridging does not require preparing the 600 benchmark cases.

The editing model stays in its backbone environment. Emotion2Vec runs in `SEmoEditBench/.venv`, and bridge synthesis runs in `third_party/IndexTTS/.venv`; both subprocesses inherit `CUDA_VISIBLE_DEVICES`. Select a GPU with enough free memory for the editing model and a bridge worker together. No Emotion2Vec or IndexTTS2 bridging dependencies are needed when bridging is disabled or strength is `0` or `1` (IndexTTS2 editing still requires its own backbone environment and weights).

## Custom donor data

Emotion bridging accepts any directory of speech WAV files. Donors need neither emotion labels, speaker IDs, nor transcripts: retrieval uses Emotion2Vec embeddings, and IndexTTS2 reads the selected donor WAV as its emotion reference. To build an index from your own dataset, omit `--donor-manifest`:

```bash
SEmoEditBench/.venv/bin/python scripts/bridge_retrieval.py build \
  --corpus-root /path/to/donor_speech \
  --index-root SEmoEditBench/outputs/_emotion2vec_custom_index \
  --model checkpoints/emotion2vec_plus_large \
  --device cuda:0 \
  --batch-size 32
```

The builder recursively scans files with a case-insensitive `.wav` extension, sorts their relative paths, and saves the resulting list in `metadata.jsonl` alongside `embeddings.npy` and `status.json`. Repeating the command resumes the same index; if the donor file list, audio content, corpus location, or Emotion2Vec checkpoint changes, use a new index directory. An optional JSONL manifest can restrict the donor set or fix its order; each line needs only an `audio` path relative to `--corpus-root`. Additional metadata is optional.

Use the custom index by passing `--donor-index SEmoEditBench/outputs/_emotion2vec_custom_index` to the single-sample CLI, or the same path as `EmotionBridge(index_root=...)` in Python.

## Single-sample CLI

After building the donor index, run an intermediate-strength edit with explicit bridging paths. This F5-TTS example uses the same sample inputs as the [main CLI guide](../README.md#single-sample-cli):

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n f5-tts python scripts/edit.py \
  --model f5_tts \
  --source-audio source.wav \
  --text "I never said she stole my money." \
  --target-reference-audio happy_reference.wav \
  --target-reference-text "What a wonderful day." \
  --upstream-root third_party/F5-TTS \
  --checkpoint checkpoints/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors \
  --vocoder checkpoints/vocos-mel-24khz \
  --steps 32 \
  --cfg 2.0 \
  --sway -1.0 \
  --seed 42 \
  --tau 0 \
  --strength 0.5 \
  --donor-index SEmoEditBench/outputs/_emotion2vec_esd_index \
  --emotion-model checkpoints/emotion2vec_plus_large \
  --emotion-python SEmoEditBench/.venv/bin/python \
  --bridge-python third_party/IndexTTS/.venv/bin/python \
  --bridge-checkpoint checkpoints/IndexTTS/IndexTTS-2 \
  --bridge-upstream-root third_party/IndexTTS \
  --output outputs/f5_tts.wav
```

Use the same bridging options with the CosyVoice2 and IndexTTS2 commands in the main README, changing `--strength` to `0.5`. Select a custom dataset with `--donor-index` pointing to its completed index. Add `--no-emotion-bridging` to use direct transport at intermediate strengths.

| Option | Description | Default |
| --- | --- | --- |
| `--donor-index` | Complete donor index directory | Bridging configuration |
| `--emotion-model` | Local Emotion2Vec checkpoint used to build the index | Bridging configuration |
| `--emotion-python` | Emotion2Vec environment executable | `runtime.benchmark_python` |
| `--bridge-python` | IndexTTS2 environment executable | IndexTTS2 configuration |
| `--bridge-checkpoint`, `--bridge-upstream-root` | IndexTTS2 bridge weights and source directory | IndexTTS2 configuration |

Command-line values override configuration defaults. Paths are relative to the current working directory. Bridged runs retain the first-pass WAV, donor metadata, and synthesized reference under `outputs/<name>_bridging/`, alongside the output WAV and its JSON diagnostics sidecar.

## Python API and batches

Reuse `editor` and `case` from the [Python API example](../README.md#python-api), construct a bridge, and replace the direct `edit()` call with:

```python
from pathlib import Path

from semoedit.bridging import EmotionBridge
from semoedit.inference import edit

bridge = EmotionBridge(
    index_root=Path("SEmoEditBench/outputs/_emotion2vec_esd_index"),
    emotion2vec_model=Path("checkpoints/emotion2vec_plus_large"),
    emotion_python=Path("SEmoEditBench/.venv/bin/python"),
    index_python=Path("third_party/IndexTTS/.venv/bin/python"),
    index_checkpoint=Path("checkpoints/IndexTTS/IndexTTS-2"),
    index_upstream_root=Path("third_party/IndexTTS"),
    work_dir=Path("outputs/python_bridging"),
)
waveform, diagnostics = edit(editor, case, strength=0.5, seed=42, bridging=bridge)
```

Save `waveform` using the main example's `sf.write()` call. `diagnostics` contains `first_pass`, `bridging`, and `second_pass`. Pass `emotion_bridging=False` to run direct transport without constructing a bridge. Strengths `0` and `1` do not use the bridge. The bridge workers run in their separate environments for each sample.

For [batch inference](../README.md#batch-inference), reuse the bridge and call `edit(editor, case, strength=0.5, seed=42, source_index=index, bridging=bridge)` inside the loop. Bridge artifacts are separated by `source_index` and strength under `bridge.work_dir`; use a different work directory for a different dataset.

## Benchmark configuration

For paper reproduction, set `bridging.donor_corpus` in `configs/local.yaml` to the complete ESD root, matching `raw_dataset_roots.ESD` in `SEmoEditBench/configs/benchmark.yaml`, and retain the fixed donor manifest.

For a custom donor dataset, update these fields in `configs/local.yaml`:

```yaml
bridging:
  donor_corpus: /path/to/donor_speech
  donor_manifest: null
  index_root: ../SEmoEditBench/outputs/_emotion2vec_custom_index
```

Keep the other bridging fields from the default configuration. The benchmark launcher builds or resumes this index automatically. Donor quality and emotion coverage affect retrieval and editing results; the default 35,000-utterance ESD manifest remains the configuration for reproducing the paper.

The benchmark intensity protocol bridges every nonzero strength, including `1`. See the [benchmark guide](../SEmoEditBench/README.md#run-inference) for the full protocol and launch commands.
