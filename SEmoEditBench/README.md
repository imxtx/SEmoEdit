# SEmoEditBench (600 cases)

This directory contains the nine fixed manifests used for the paper's 600 benchmark cases (320 replacement, 152 erasure, and 128 intensity), the audio preparation code, and the objective evaluator. It has its own Python project and lockfile, so run the following commands from this directory.

## Environment

Use Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for the model-based preparation and evaluation steps:

```bash
cd SEmoEditBench
uv sync
```

`uv` creates this directory's `.venv` from `pyproject.toml` and `uv.lock`. Model weights are downloaded by their respective libraries when first used.

## Data and configuration

Obtain ESD, IEMOCAP, RAVDESS, and CREMA-D from their providers. Set their original locations in `raw_dataset_roots` in `configs/benchmark.yaml`. Relative paths in this file are resolved from `configs/`.

If a synthesized CREMA-D reference is missing from both `data/` and the raw corpus, preparation requires a separate [IndexTTS2](https://github.com/index-tts/index-tts) checkout with its own `.venv` and checkpoint. Set its paths under `indextts2` in the config. IndexTTS2 is not installed into the benchmark environment.

The fixed manifests refer to audio under `data/`. Keep `dataset_roots` aligned with the output of preparation when evaluating.

## Prepare

Copy the audio referenced by `manifests600/` into `data/`, synthesizing only missing CREMA-D references:

```bash
uv run emoedit prepare --config configs/benchmark.yaml
```

To see how many files are already present, need copying, or need synthesis without writing files:

```bash
uv run emoedit prepare --config configs/benchmark.yaml --dry-run
```

Preparation does not sample cases or write new manifests. The nine checked-in manifests are the evaluation specification. Existing audio is reused, and generated references already present in the raw corpus are copied instead of synthesized.

## Evaluate

Place edited WAV files under `outputs/<system>/<split>/<case_id>.wav`, where `<split>` is a filename stem in `manifests600/`. For intensity cases, also provide `outputs/<system>/<split>/<case_id>/<strength>.wav` for strengths `0`, `0.25`, `0.5`, `0.75`, and `1`. The evaluator checks output coverage before loading models.

From this directory, preflight one system and then run the metrics:

```bash
uv run emoedit evaluate --manifest-dir manifests600 --output-root outputs/<system> --results-root results600 --device cuda:0 --dry-run
uv run emoedit evaluate --manifest-dir manifests600 --output-root outputs/<system> --results-root results600 --device cuda:0
```

To evaluate every matching system under `outputs/`, use `uv run emoedit evaluate --device cuda:0`; by default it selects directories ending in `600cases`. Results are written to `results600/`. See [evaluation metrics](docs/evaluation_metrics.md) for metric definitions and supported-case rules.

Preparation does not load an ASR model. Evaluation uses ASR to compute the text-preservation metric ΔWER.
