# Evaluation Metrics

## Notation

- $x_s$, $x_e$, $x_t$: source, edited, and paired-target waveforms.
- $e_s$, $e_t$: source and target emotion labels.
- $y$: reference transcript; $A(x)$: transcript of waveform $x$.
- $P(e\mid x)$: probability of emotion $e$ given waveform $x$.
- $f_E(x)$, $f_S(x)$: emotion and speaker embeddings of waveform $x$.
- $Q(x)$: predicted perceptual quality of waveform $x$.
- $\cos(u,v)=u^\top v/(\lVert u\rVert_2\lVert v\rVert_2)$.

## Editing success

**Target emotion probability (TEP)** measures the probability of the requested emotion in a replacement edit. Higher is better.

$$
\mathrm{TEP}=P(e_t\mid x_e)
$$

**Neutral probability (NP)** measures the probability of neutral emotion after erasure. Higher is better.

$$
\mathrm{NP}=P(\mathrm{neutral}\mid x_e)
$$

**Source emotion suppression (SES)** measures the decrease in the source emotion probability. It applies to replacement and erasure. Higher is better.

$$
\mathrm{SES}=P(e_s\mid x_s)-P(e_s\mid x_e)
$$

**Embedding-based effective intensity control (EIC-Emb)** measures whether stronger edits move farther toward a paired target in emotion-embedding space. For five outputs $x_e^{(\alpha_i)}$ at increasing strengths $\alpha_i\in\{0,0.25,0.5,0.75,1\}$, let $z_s=f_E(x_s)$, $z_t=f_E(x_t)$, and $z_i=f_E(x_e^{(\alpha_i)})$. Their progress along the source-to-target direction is

$$
a_i=\frac{(z_i-z_s)^\top(z_t-z_s)}{\lVert z_t-z_s\rVert^2}.
$$

EIC-Emb averages the signed progress over all ten pairs $i<j$, with $d_{ij}=a_j-a_i$ and $\tau=1$:

$$
\mathrm{EIC-Emb}=\frac{1}{10}\sum_{i\lt j}\mathrm{sgn}(d_{ij})\min(1,\max(0,|d_{ij}|/\tau))
$$

It applies to intensity cases with a paired target. Higher is better; negative values indicate decreasing progress as strength increases.

SEmoEdit receives per-case results for all 128 intensity cases. EIC-Emb is reported for the 90 shared cases with a neutral source and an angry, happy, sad, or surprise target; the other 38 records retain their probability and reference-progress metrics with EIC-Emb set to `null`. Intensity summaries and comparison reports aggregate the 90 shared cases.

**Emotion similarity (E-SIM)** measures the cosine similarity of edited and paired-target emotion embeddings. It applies when a paired target exists. Higher is better.

$$
\mathrm{E-SIM}=\cos(f_E(x_e),f_E(x_t))
$$

**Directional editing score (DES)** measures whether the edit moves in the source-to-target emotion direction. It applies when a paired target exists. Higher is better. A zero edited displacement scores zero; a zero target displacement is invalid.

$$
\mathrm{DES}=\cos(f_E(x_e)-f_E(x_s),f_E(x_t)-f_E(x_s))
$$

## Preservation

**Relative word error rate ($\Delta$ WER)** measures the change in transcription error against the same reference transcript. For Chinese, character tokenization makes this effectively $\Delta$CER. It applies to all tasks. Lower is better.

$$
\Delta\mathrm{WER}=\mathrm{WER}(y,A(x_e))-\mathrm{WER}(y,A(x_s))
$$

**Speaker similarity (S-SIM)** measures preservation of the source speaker's identity. It applies to all tasks. Higher is better.

$$
\mathrm{S-SIM}=\cos(f_S(x_s),f_S(x_e))
$$

## Quality

**UTMOS** estimates the perceptual quality of the edited waveform. It applies to all tasks. Higher is better.

$$
\mathrm{UTMOS}=Q(x_e)
$$
