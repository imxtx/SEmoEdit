# Backbones and weights

Run these commands from the SEmoEdit repository root. Use Linux x86_64, a CUDA GPU, Conda for F5-TTS and CosyVoice2, and uv for IndexTTS2. Source and dependency versions are listed in [upstream.json](upstream.json) and [environments/](environments/). The environments are intentionally separate. The pinned PyTorch builds use CUDA 13.0 for F5-TTS and the benchmark, CUDA 12.1 for CosyVoice2, and CUDA 12.8 for IndexTTS2. Install an NVIDIA driver supporting CUDA 13.0; the Python packages supply the CUDA runtime libraries.

## Fixed upstream source

```bash
git clone https://github.com/SWivid/F5-TTS.git third_party/F5-TTS
git -C third_party/F5-TTS checkout 9c614e9657089213efc6a7421b30630be138a3f5
git clone https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
git -C third_party/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C third_party/CosyVoice submodule update --init --recursive
git clone https://github.com/index-tts/index-tts.git third_party/IndexTTS
git -C third_party/IndexTTS checkout d9e41aac89fd00b3d71497fddb287b7f24613712
```

CosyVoice's pinned Matcha-TTS submodule is `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`.

## Environments

```bash
conda create -n f5-tts python=3.11 pip -y
conda run --no-capture-output -n f5-tts python -m pip install \
  -e third_party/F5-TTS \
  -c third_party/environments/f5_tts.constraints.txt
conda run --no-capture-output -n f5-tts python -m pip install --no-deps -e .

conda create -n cosyvoice2 python=3.10 pip -y
conda run --no-capture-output -n cosyvoice2 python -m pip install "pip>=25.3"
conda run --no-capture-output -n cosyvoice2 python -m pip install \
  -r third_party/environments/cosyvoice2.requirements.txt \
  -c third_party/environments/cosyvoice2.constraints.txt \
  --build-constraint third_party/environments/cosyvoice2.build-constraints.txt
conda run --no-capture-output -n cosyvoice2 python -m pip install --no-deps -e .

uv sync --project third_party/IndexTTS --locked
uv pip install --python third_party/IndexTTS/.venv/bin/python --no-deps -e .
uv sync --project SEmoEditBench --locked
```

Install the upstreams' system prerequisites (including ffmpeg and the build tools needed by their Python dependencies) as described in their pinned READMEs. The constraints files pin F5-TTS and CosyVoice2 dependencies. Full dependency lists are available in the JSON environment files. IndexTTS2 and the benchmark use their uv lockfiles. CosyVoice2 also uses a build constraint for its source-package dependencies.

Set `models.f5_tts.python` and `models.cosyvoice2.python` in your inference YAML to the executable paths printed by:

```bash
conda run -n f5-tts python -c 'import sys; print(sys.executable)'
conda run -n cosyvoice2 python -c 'import sys; print(sys.executable)'
```

The IndexTTS2 executable defaults to `third_party/IndexTTS/.venv/bin/python`. Keep the virtual-environment executable path; do not replace it with its symlink target. Model source always comes from the configured `upstream_root`.

## Fixed model downloads

Install the current [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) in a download environment. Model revisions are pinned in [weights.json](weights.json).

```bash
hf download SWivid/F5-TTS \
  F5TTS_v1_Base/model_1250000.safetensors \
  F5TTS_v1_Base/vocab.txt \
  --revision 84e5a410d9cead4de2f847e7c9369a6440bdfaca \
  --local-dir checkpoints/F5-TTS
hf download charactr/vocos-mel-24khz \
  --revision 0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21 \
  --local-dir checkpoints/vocos-mel-24khz
hf download FunAudioLLM/CosyVoice2-0.5B \
  --revision eec1ae6c79877dbd9379285cf8789c9e0879293d \
  --local-dir checkpoints/CosyVoice/CosyVoice2-0.5B
hf download IndexTeam/IndexTTS-2 \
  --revision 740dcaff396282ffb241903d150ac011cd4b1ede \
  --local-dir checkpoints/IndexTTS/IndexTTS-2
hf download facebook/w2v-bert-2.0 \
  --revision da985ba0987f70aaeb84a80f2851cfac8c697a7b \
  --local-dir checkpoints/IndexTTS/IndexTTS-2/hf_cache/w2v-bert-2.0
hf download amphion/MaskGCT \
  semantic_codec/model.safetensors \
  --revision 265c6cef07625665d0c28d2faafb1415562379dc \
  --local-dir checkpoints/IndexTTS/IndexTTS-2/hf_cache
hf download funasr/campplus \
  campplus_cn_common.bin \
  --revision e4b6ede7ce16997aff4ae69fbca1f0175e2afede \
  --local-dir checkpoints/IndexTTS/IndexTTS-2/hf_cache
hf download nvidia/bigvgan_v2_22khz_80band_256x \
  --revision 633ff708ed5b74903e86ff1298cf4a98e921c513 \
  --local-dir checkpoints/IndexTTS/IndexTTS-2/hf_cache/bigvgan
hf download emotion2vec/emotion2vec_plus_large \
  --revision 6c303ba987b86b93193de93e34bb2b077a6bedc4 \
  --local-dir checkpoints/emotion2vec_plus_large
```

The IndexTTS2 auxiliary layout is required by the adapter. CosyVoice's complete snapshot includes `CosyVoice-BlankEN/`, its tokenizer and speaker encoder. CosyVoice2 also downloads its WeText normalization assets from ModelScope on first use, so that first run requires network access. Emotion2Vec and the IndexTTS2 main and auxiliary checkpoints are needed for emotion bridging on every backbone. Complete the [donor index setup](../docs/emotion_bridging.md#environment-and-donor-index) before using intermediate strengths. Direct transport with bridging disabled does not load Emotion2Vec or the IndexTTS2 bridge synthesizer. Model and corpus access and licenses follow their respective providers.
