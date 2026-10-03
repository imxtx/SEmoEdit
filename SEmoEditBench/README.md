# SEmoEditBench (600 cases)

This directory contains the nine fixed manifests used for the paper's 600 benchmark cases (320 replacement, 152 erasure, and 128 intensity), the audio preparation code, and the objective evaluator. It has its own Python project and lockfile. Run all commands below from the SEmoEdit repository root.

## Environment

Use Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for the model-based preparation and evaluation steps:

```bash
uv sync --project SEmoEditBench --locked
```

`uv` creates `SEmoEditBench/.venv` from its `pyproject.toml` and `uv.lock`. Prepare backbone and Emotion2Vec weights using [the fixed download instructions](../third_party/README.md#fixed-model-downloads). The ASR, speaker, and quality models download their weights when first used.

## Data and configuration

Obtain ESD, IEMOCAP, RAVDESS, and CREMA-D from their providers. Set their original locations in `raw_dataset_roots` in `SEmoEditBench/configs/benchmark.yaml`. Relative paths in this file are resolved from `SEmoEditBench/configs/`.

The corpus roots should contain these directories:

| Corpus | Layout under its root |
| --- | --- |
| ESD | `0001/`–`0020/`, including each speaker's transcript file |
| IEMOCAP | `Session1/`–`Session5/`, including dialogue transcripts |
| RAVDESS | `Actor_01/`–`Actor_24/` |
| CREMA-D | `AudioWAV/` |

If a synthesized CREMA-D reference is missing from both `SEmoEditBench/data/` and the raw corpus, preparation requires a separate [IndexTTS2](https://github.com/index-tts/index-tts) checkout with its own `.venv` and checkpoint. Set its paths under `indextts2` in the config. IndexTTS2 is not installed into the benchmark environment.

The fixed manifests refer to audio under `SEmoEditBench/data/`. Keep `dataset_roots` aligned with the output of preparation when evaluating.

## Prepare

Copy the audio referenced by `SEmoEditBench/manifests600/` into `SEmoEditBench/data/`, synthesizing only missing CREMA-D references:

```bash
uv run --project SEmoEditBench emoedit prepare --config SEmoEditBench/configs/benchmark.yaml
```

To see how many files are already present, need copying, or need synthesis without writing files:

```bash
uv run --project SEmoEditBench emoedit prepare \
  --config SEmoEditBench/configs/benchmark.yaml \
  --dry-run
```

Preparation does not sample cases or write new manifests. The nine checked-in manifests are the evaluation specification. Existing audio is reused, and generated references already present in the raw corpus are copied instead of synthesized.

New CREMA-D references use `indextts2.seed` (default 42), reset for each reference so partial reruns produce the same result. Existing prepared references are reused; to regenerate one with the fixed seed, remove only that generated reference from `SEmoEditBench/data/` before preparation.

## Evaluate

Place edited WAV files under `SEmoEditBench/outputs/<system>/<split>/<case_id>.wav`, where `<split>` is a filename stem in `SEmoEditBench/manifests600/`. For intensity cases, also provide `SEmoEditBench/outputs/<system>/<split>/<case_id>/<strength>.wav` for strengths `0`, `0.25`, `0.5`, `0.75`, and `1`. The evaluator checks output coverage before loading models.

Preflight one system and then run the metrics (replace `<system>` with the system directory name):

```bash
uv run --project SEmoEditBench emoedit evaluate \
  --config SEmoEditBench/configs/benchmark.yaml \
  --manifest-dir SEmoEditBench/manifests600 \
  --output-root SEmoEditBench/outputs/<system> \
  --emotion-model checkpoints/emotion2vec_plus_large \
  --results-root SEmoEditBench/results600 \
  --device cuda:0 \
  --dry-run

# Remove --dry-run to compute all metrics.
```

To evaluate every matching system, pass the parent output directory:

```bash
uv run --project SEmoEditBench emoedit evaluate \
  --config SEmoEditBench/configs/benchmark.yaml \
  --manifest-dir SEmoEditBench/manifests600 \
  --output-root SEmoEditBench/outputs \
  --emotion-model checkpoints/emotion2vec_plus_large \
  --results-root SEmoEditBench/results600 \
  --device cuda:0
```

By default, this selects system directories ending in `600cases`. Results are written to `SEmoEditBench/results600/`. See [evaluation metrics](docs/evaluation_metrics.md) for metric definitions and supported-case rules.

Preparation does not load an ASR model. Evaluation uses ASR to compute the text-preservation metric ΔWER.
