# SEmoEdit

[![Static Badge](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2609.34648) [![Demo](https://img.shields.io/badge/Demo-blue)](https://semoedit.pages.dev/) [![SEmoEditBench](https://img.shields.io/badge/SEmoEditBench-orange)](SEmoEditBench/README.md)

Official implementation of **SEmoEdit: Probing and Harnessing the Editability of Pre-trained Speech Flows**.

SEmoEdit edits the emotion of generated speech using pre-trained speech flows. It supports **F5-TTS, CosyVoice2, and IndexTTS2**, with a shared Python API, single-sample CLI, and the 600-case SEmoEditBench pipeline for emotion replacement, erasure, and intensity editing.

[Installation](#installation) · [Single-sample CLI](#single-sample-cli) · [Python API](#python-api) · [Batch inference](#batch-inference) · [Benchmark](#benchmark) · [Citation](#citation)

## Installation

First, clone this repo:

```bash
git clone https://github.com/imxtx/SEmoEdit.git
```

Then run all commands below from this repository's root.

### 1. Prepare backbones and weights

Follow [the setup guide](third_party/README.md#fixed-upstream-source) to clone the three backbones at their pinned commits. [Download the model weights](third_party/README.md#fixed-model-downloads) into `checkpoints/`.

### 2. Install environments

Each backbone uses its own environment:

```bash
# F5-TTS
conda create -n f5-tts python=3.11 pip -y
conda run -n f5-tts python -m pip install \
  -e third_party/F5-TTS \
  -c third_party/environments/f5_tts.constraints.txt
conda run -n f5-tts python -m pip install --no-deps -e .

# CosyVoice2
conda create -n cosyvoice2 python=3.10 pip -y
conda run -n cosyvoice2 python -m pip install "pip>=25.3"
conda run -n cosyvoice2 python -m pip install \
  -r third_party/environments/cosyvoice2.requirements.txt \
  -c third_party/environments/cosyvoice2.constraints.txt \
  --build-constraint third_party/environments/cosyvoice2.build-constraints.txt
conda run -n cosyvoice2 python -m pip install --no-deps -e .

# IndexTTS2
uv sync --project third_party/IndexTTS --locked
uv pip install --python third_party/IndexTTS/.venv/bin/python --no-deps -e .
```

If the backbone environments are already installed, run only the SEmoEdit installation command for each environment. System prerequisites and dependency versions are documented in the [setup guide](third_party/README.md#environments).

### 3. Configure benchmark paths (optional)

```bash
cp configs/inference.yaml configs/local.yaml
conda run -n f5-tts python -c 'import sys; print(sys.executable)'
conda run -n cosyvoice2 python -c 'import sys; print(sys.executable)'
```

Single-sample inference accepts model paths directly on the command line. For benchmark runs, in `configs/local.yaml`, set `models.f5_tts.python` and `models.cosyvoice2.python` to the printed executable paths. Check the `upstream_root`, `checkpoint`, and F5-TTS `vocoder` paths if you use a different directory layout. YAML paths are relative to the configuration file's directory; absolute paths are also accepted.

Configure benchmark donor data using the [emotion bridging guide](docs/emotion_bridging.md#benchmark-configuration).

| Backbone | Model argument | Environment |
| --- | --- | --- |
| F5-TTS | `f5_tts` | Conda `f5-tts` |
| CosyVoice2 | `cosyvoice2` | Conda `cosyvoice2` |
| IndexTTS2 | `indextts2` | `third_party/IndexTTS/.venv` |

## Emotion bridging

Single-sample CLI and Python inference use emotion bridging by default for `0 < strength < 1`. Strengths `0` and `1` use direct transport. Pass `--no-emotion-bridging` or `emotion_bridging=False` to disable bridging at intermediate strengths.

See the [emotion bridging guide](docs/emotion_bridging.md) for environment setup, donor indexing, custom WAV datasets, CLI/Python examples, and benchmark configuration.

## Single-sample CLI

Provide the source audio, its transcript, the target reference audio, and its transcript directly on the command line. The source reference must speak `--text`; the target reference can speak different text. SEmoEdit first synthesizes source speech with the source reference, then edits that generated speech toward the target reference's emotion.

```bash
# F5-TTS
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
  --strength 1 \
  --output outputs/f5_tts.wav

# CosyVoice2
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n cosyvoice2 python scripts/edit.py \
  --model cosyvoice2 \
  --source-audio source.wav \
  --text "I never said she stole my money." \
  --target-reference-audio happy_reference.wav \
  --target-reference-text "What a wonderful day." \
  --upstream-root third_party/CosyVoice \
  --checkpoint checkpoints/CosyVoice/CosyVoice2-0.5B \
  --steps 10 \
  --cfg 0.7 \
  --seed 42 \
  --tau 0 \
  --strength 1 \
  --output outputs/cosyvoice2.wav

# IndexTTS2
CUDA_VISIBLE_DEVICES=0 third_party/IndexTTS/.venv/bin/python scripts/edit.py \
  --model indextts2 \
  --source-audio source.wav \
  --text "I never said she stole my money." \
  --target-reference-audio happy_reference.wav \
  --target-reference-text "What a wonderful day." \
  --upstream-root third_party/IndexTTS \
  --checkpoint checkpoints/IndexTTS/IndexTTS-2 \
  --steps 25 \
  --cfg 0.7 \
  --seed 42 \
  --tau 0 \
  --strength 1 \
  --output outputs/indextts2.wav
```

Each command writes a WAV and a JSON sidecar containing the inputs, settings, and solver diagnostics. No input JSON file is required. Command-line paths are relative to the current working directory.

| Option | Description | Default |
| --- | --- | --- |
| `--model` | `f5_tts`, `cosyvoice2`, or `indextts2` | Required |
| `--source-audio` | Source reference WAV | Required |
| `--text` | Source transcript and synthesis text | Required |
| `--target-reference-audio` | Target emotion reference WAV | Required |
| `--target-reference-text` | Target reference transcript | Required |
| `--output` | Output WAV path | Required |
| `--checkpoint`, `--upstream-root` | Backbone weights and source directory | Model configuration |
| `--vocoder` | F5-TTS vocoder directory | Model configuration |
| `--steps`, `--cfg`, `--sway` | Solver settings; `--sway` applies to F5-TTS | Model configuration |
| `--seed` | Random seed | `42` |
| `--tau` | Start time for transport updates | `0` |
| `--strength` | First-pass editing strength in `[0, 1]`; intermediate values use bridging by default | `1` |
| `--no-emotion-bridging` | Use direct transport at intermediate strengths | Bridging enabled |
| `--config` | Optional override of the default configuration file | `configs/inference.yaml` |

Command-line values override the configuration defaults. You can omit model paths and solver options when they are already configured in `configs/inference.yaml`, or pass `--config configs/local.yaml` to reuse your own settings. For `--strength 0.5`, follow the [bridging CLI guide](docs/emotion_bridging.md#single-sample-cli), or add `--no-emotion-bridging` for direct transport. Use `--strength 0` for the aligned generated source. `--strength 1` performs a direct full edit. Select the GPU with `CUDA_VISIBLE_DEVICES`.

For an ESD example, use `SEmoEditBench/raw_data/ESD/0014/Neutral/0014_000001.wav` as the source and `SEmoEditBench/raw_data/ESD/0014/Happy/0014_000701.wav` as the target reference, with both transcripts set to `"The nine the eggs, I keep."`. Benchmark preparation is not required. Intermediate strengths require the [bridging setup](docs/emotion_bridging.md#environment-and-donor-index) unless `--no-emotion-bridging` is specified.

## Python API

Run the following in the `cosyvoice2` environment. This example loads a model, supplies reference paths and transcripts, edits the sample, and saves the result:

```python
from pathlib import Path

import soundfile as sf
from semoedit.inference import edit, load_editor

editor = load_editor(
    model="cosyvoice2",
    checkpoint=Path("checkpoints/CosyVoice/CosyVoice2-0.5B"),
    upstream_root=Path("third_party/CosyVoice"),
    steps=10,
    cfg=0.7,
)

case = {
    "source_audio": str(Path("source.wav").resolve()),
    "text": "I never said she stole my money.",
    "target_reference_audio": str(Path("happy_reference.wav").resolve()),
    "target_reference_text": "What a wonderful day.",
}

waveform, diagnostics = edit(editor, case, strength=1, seed=42)
output = Path("outputs/edited.wav")
output.parent.mkdir(parents=True, exist_ok=True)
sf.write(
    output,
    waveform.detach().float().cpu().numpy().reshape(-1),
    editor.sample_rate,
    subtype="FLOAT",
)
```

`edit()` returns a waveform tensor and a diagnostics dictionary with `first_pass`. Reuse the editor for subsequent samples. For intermediate strengths, follow the [bridging Python guide](docs/emotion_bridging.md#python-api-and-batches), or pass `emotion_bridging=False` for direct transport. To load another backbone, use its model name, checkpoint, and upstream directory with these settings:

| Model | `steps` | `cfg` | Additional `load_editor()` arguments |
| --- | --- | --- | --- |
| `f5_tts` | 32 | 2.0 | `vocoder=Path("checkpoints/vocos-mel-24khz"), sway=-1.0` |
| `cosyvoice2` | 10 | 0.7 | — |
| `indextts2` | 25 | 0.7 | — |

F5-TTS's `checkpoint` points to `model_1250000.safetensors`; the other two checkpoints are directories. See [the default configuration](configs/inference.yaml) for their paths.

## Batch inference

For your own dataset, create `samples.jsonl` with one sample object per line, using the same four fields as the Python `case` dictionary. Resolve audio paths relative to the JSONL file, and reuse the editor from the Python example:

```python
import json

manifest = Path("samples.jsonl").resolve()
output_dir = Path("outputs/batch")
output_dir.mkdir(parents=True, exist_ok=True)

for index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
    case = json.loads(line)
    for field in ("source_audio", "target_reference_audio"):
        case[field] = str((manifest.parent / case[field]).resolve())

    waveform, diagnostics = edit(editor, case, strength=1, seed=42, source_index=index)
    sf.write(
        output_dir / f"{index:05d}.wav",
        waveform.detach().float().cpu().numpy().reshape(-1),
        editor.sample_rate,
        subtype="FLOAT",
    )
```

This loads the model once and writes numbered WAVs. `source_index` assigns each sample an independent random stream; keep the manifest order fixed for repeatable runs. For the fixed SEmoEditBench manifests, use the batch CLI below.

For intermediate-strength batches, follow the [bridging batch guide](docs/emotion_bridging.md#python-api-and-batches), or pass `emotion_bridging=False` for direct transport.

## Benchmark

SEmoEditBench provides **600 fixed cases: 320 replacement, 152 erasure, and 128 intensity**. See the [SEmoEditBench README](SEmoEditBench/README.md) for dataset setup, inference options, output layout, and evaluation details.

After configuring dataset locations in `SEmoEditBench/configs/benchmark.yaml` and model paths in `configs/local.yaml`, prepare the audio and run each backbone:

```bash
uv sync --project SEmoEditBench --locked
uv run --project SEmoEditBench emoedit prepare \
  --config SEmoEditBench/configs/benchmark.yaml
export SEMOEDIT_PYTHON="$PWD/SEmoEditBench/.venv/bin/python"
bash scripts/run_f5_tts.sh --config configs/local.yaml
bash scripts/run_cosyvoice2.sh --config configs/local.yaml
bash scripts/run_indextts2.sh --config configs/local.yaml
```

Evaluate the generated audio following the [evaluation instructions](SEmoEditBench/README.md#evaluate).

## Project layout

```text
semoedit/            # Transport, inference API, bridging and model adapters
scripts/             # Single-sample CLI and benchmark orchestration
configs/             # Inference settings and fixed donor manifest
docs/                # Bridging and development guides
SEmoEditBench/       # Benchmark manifests, preparation and evaluation
third_party/         # Upstream setup and version records
```


## Citation

If you find this project helpful, please consider citing the following paper:

```bibtex
@article{xie2026semoedit,
  title={SEmoEdit: Probing and Harnessing the Editability of Pre-trained Speech Flows},
  author={Xie, Tianxin and Zhang, Pengfei and Jiang, Kai and Zhao, Zelin and Liu, Li},
  journal={arXiv preprint arXiv:2609.34648},
  year={2026}
}
```

## License

SEmoEdit is released under the [MIT License](LICENSE). Backbones, model weights, and datasets retain their respective licenses.

## Star History

<a href="https://www.star-history.com/?repos=imxtx%2Fsemoedit&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=imxtx/semoedit&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=imxtx/semoedit&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=imxtx/semoedit&type=date&legend=top-left" />
 </picture>
</a>


## Development

[Development guide](docs/development.md)