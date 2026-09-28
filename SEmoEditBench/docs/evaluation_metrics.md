# Evaluation Metrics

This document defines the metrics produced by `emoedit evaluate`. Metrics are computed per case and then averaged over each selected manifest split. Missing values are excluded from the corresponding mean.

The CLI defaults to the 600-case manifests and evaluates all `outputs/*600cases` systems. It requires complete output coverage before loading models. Each model is loaded once for the batch; source ASR is cached separately from immutable manifests. Per-system results and a combined comparison report are written under `results600/`. Optional paired-target metrics are excluded from completion requirements when a case has no paired target.

For `cocoemo_cosyvoice2_baseline_activation_steering_600cases` and `cocoemo_indextts2_baseline_activation_steering_600cases`, evaluation follows the inference scripts: only `replacement` and `intensity` cases with `source_emotion == "neutral"` and target emotion `angry`, `happy`, `sad`, or `surprise` are included. Preflight checks, all metrics, and report counts use this subset; empty splits (including erasure) are reported as unsupported/N/A. All five intensity outputs remain required for included cases. Filtered manifests are saved under each system's results directory in `_manifests/`; the original manifests and other systems' evaluation scope are unchanged.

## Notation

- $x_s$: source waveform (`source_audio`)
- $x_e$: edited waveform (`outputs/<system>/<split>/<case_id>.wav`)
- $x_t$: optional paired evaluation target (`target_audio`)
- $y$: manifest transcript (`text`)
- $A(x)$: ASR transcript of waveform $x$
- $P(e \mid x)$: Emotion2Vec probability for emotion $e$
- $f_E(x)$: Emotion2Vec embedding
- $f_S(x)$: ECAPA-TDNN speaker embedding
- $Q(x)$: UTMOSv2 quality prediction

## Emotion conversion

For `replacement`, let $e_s$ be the source emotion and $e_t$ the requested target emotion.

**Target-emotion probability (TEP)**

$$
\mathrm{TEP} = P(e_t \mid x_e)
$$

**Source-emotion suppression (SES)**

$$
\mathrm{SES} = P(e_s \mid x_s) - P(e_s \mid x_e)
$$

Higher values indicate stronger target emotion and greater suppression of the source emotion.

## Emotion erasure

For `erasure`, the requested output emotion is neutral.

**Neutral probability (NP)**

$$
\mathrm{NP} = P(\mathrm{neutral} \mid x_e)
$$

SES is also reported, using the non-neutral source emotion as $e_s$.

## Intensity control

Intensity uses five edited waveforms at strengths $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$, stored as `outputs/<system>/<split>/<case_id>/<strength>.wav`. The ordinary case file is retained as the $\alpha=1$ waveform for shared ASR, speaker, and quality metrics.

For target emotion $e_t$, the evaluator records

$$
p_i = P\left(e_t \mid x_e^{(\alpha_i)}\right)
$$

for each strength. No `target_audio` is used.

**Intensity monotonicity**

An inversion is a pair $(i,j)$ with $i<j$ and $p_i>p_j$. With five strengths there are $\binom{5}{2}=10$ pairs:

$$
\mathrm{inversions} = \sum_{i<j} \mathbb{1}[p_i > p_j]
$$

$$
\mathrm{intensity\_monotonicity}
= 1 - \frac{\mathrm{inversions}}{10}
$$

Equal probabilities do not count as inversions. A score of $1.0$ is fully non-decreasing and $0.0$ reverses every pair. The complete curve is stored in `intensity_probabilities`; split summaries use its macro mean.

**Effective intensity control (EIC, experimental)**

The CLI additionally computes `effective_intensity_control` from the saved five
probabilities: the mean over all ten pairs of
`sign(d) * clip((abs(d) - epsilon) / (tau - epsilon), 0, 1)`, where
`d = p_j - p_i`, `epsilon = 0`, and `tau = 0.2`. These thresholds are uncalibrated.
Constant curves score zero; negative scores indicate decreasing probabilities.
The existing monotonicity metric is unchanged.

`emoedit evaluate` writes EIC and its parameters to each existing
`results600/<system>/<split>/intensity.jsonl`, then includes its mean in the
existing summaries, comparison report, and WebUI. Cached `emotion.jsonl` files
are sufficient: adding EIC does not invalidate model inference caches or require
`--force`. Unsupported systems retain null/N/A. Older results without EIC remain
readable and display a missing value until evaluated again.

**Reference progress and embedding-based EIC (EIC-Emb)**

For intensity cases that have a paired target audio $x_t$ (same speaker, same
text as the source), Emotion2Vec embeddings define a per-case reference
direction:

$$
a_i = \frac{(z_i - z_s)^\top (z_t - z_s)}{\lVert z_t - z_s \rVert^2}
$$

where $z_s$, $z_t$, and $z_i$ are the embeddings of the neutral source, the
paired target, and the five edited outputs. $a_i = 0$ means no progress beyond
the source along the target direction, $a_i = 1$ reaches the target's
projection, negative values move away from it, and values above 1 overshoot.
The generation-time `target_reference_audio` is never used for this anchor.

`reference_progress` stores the five $a_i$ and `reference_progress_range` is
the directed range $a_1 - a_0$.
`effective_intensity_control_embedding` applies the same pairwise EIC formula
to the five $a_i$ values with `epsilon = 0` and `tau = 1`. Unlike cosine
similarity, the projection keeps the magnitude of the change.

**Common intensity subset**

All intensity summary metrics (monotonicity, EIC, EIC-Emb, reference progress
range, ΔWER, S-SIM, UTMOS) are averaged over the same cases for every system:
neutral source with a target emotion from the activation-steering-supported
set {angry, happy, sad, surprise}. The 600-case manifests contain 30 such
cases per split (angry/happy/sad, ten each), 90 in total. Case-level
`intensity.jsonl`/`emotion.jsonl` files still retain all 128 manifest cases
for diagnosis; EIC-Emb is null for non-common cases or cases without a paired
target. The benchmark manifest itself still defines 128 intensity cases
(44/44/40).

## Text preservation

**Delta WER (ΔWER)**

$$
\Delta\mathrm{WER}
= \mathrm{WER}\left(y, A(x_e)\right)
- \mathrm{WER}\left(y, A(x_s)\right)
$$

Lower is better. WER uses the manifest transcript as reference. Text is normalized before scoring; Chinese is tokenized by character, making this effectively CER.

## Speaker preservation

**Speaker similarity (S-SIM)**

$$
\mathrm{S\text{-}SIM}
= \operatorname{cos}\left(f_S(x_s), f_S(x_e)\right)
$$

where

$$
\operatorname{cos}(u,v)
= \frac{u^\top v}{\lVert u\rVert_2\lVert v\rVert_2}
$$

Higher values indicate better preservation of the source speaker.

## Audio quality

**UTMOS**

$$
\mathrm{UTMOS} = Q(x_e)
$$

$Q$ is the pretrained UTMOSv2 predictor applied to the edited waveform. Higher values indicate better predicted quality.

## Paired-target metrics

For replacement and erasure cases with `target_audio`, the evaluator additionally computes:

**Emotion similarity (E-SIM)**

$$
\mathrm{E\text{-}SIM}
= \operatorname{cos}\left(f_E(x_e), f_E(x_t)\right)
$$

**Directional editing score (DES)**

$$
d_e = f_E(x_e) - f_E(x_s),
\qquad
d_t = f_E(x_t) - f_E(x_s)
$$

$$
\mathrm{DES} = \operatorname{cos}(d_e, d_t)
$$

These metrics are omitted when no paired target exists. `target_reference_audio` is a generation condition and is never used as $x_t$.

## Aggregation

Case metrics are averaged independently within each manifest split. Intensity split summaries use the common 30-case subset (90 across the three splits) for every system so that systems are directly comparable; case-level files retain all manifest cases. Intensity results are also grouped by the manifest's `target_intensity` label for reporting, but that label is not used in the monotonicity calculation. The benchmark does not produce a combined weighted score.
