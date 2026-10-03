"""Model loading and single-sample SEmoEdit inference."""

from .transport import derived_seed, integrate


def load_editor(
    model,
    checkpoint,
    upstream_root,
    steps,
    cfg,
    vocoder=None,
    sway=None,
    instruct=False,
):
    if model == "f5_tts":
        from .adapters.f5_tts import F5Editor

        return F5Editor(checkpoint, vocoder, upstream_root, steps, cfg, sway)
    if model == "cosyvoice2":
        from .adapters.cosyvoice2 import CosyEditor

        return CosyEditor(checkpoint, upstream_root, steps, cfg, instruct=instruct)
    if model == "indextts2":
        from .adapters.indextts2 import IndexEditor

        return IndexEditor(checkpoint, upstream_root, steps, cfg, instruct=instruct)
    raise ValueError(f"Unknown model: {model}")


def edit_prepared(
    editor,
    source,
    source_branch,
    target_branch,
    seed,
    strength=1.0,
    tau=0.0,
    vocoder_seed=42,
):
    """Edit aligned model states and decode using the model's native vocoder."""
    edited, diagnostics = integrate(
        source,
        source_branch,
        target_branch,
        editor.times,
        editor.velocity,
        seed,
        strength,
        tau,
    )
    return editor.decode(edited, vocoder_seed), diagnostics


def edit(
    editor,
    case,
    strength=1.0,
    seed=42,
    tau=0.0,
    source_index=0,
    bridging=None,
    emotion_bridging=True,
):
    """Generate source speech from its reference, then edit toward the target.

    case contains source_audio, text, target_reference_audio and
    target_reference_text. The source reference must speak case['text'].
    For 0 < strength < 1, bridging must be an EmotionBridge. Retrieve a donor
    from the first-pass edit, synthesize a reference, then edit at strength 1.
    Strengths 0 and 1 return the direct result without bridging.
    Set emotion_bridging=False to use direct transport at any strength.
    Returns (waveform tensor, diagnostics with first_pass and optional
    bridging / second_pass entries).
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    use_bridging = emotion_bridging and 0 < strength < 1
    if use_bridging and bridging is None:
        raise ValueError("An EmotionBridge is required for 0 < strength < 1")
    condition_seed = derived_seed(seed, source_index, 0)
    edit_seed = derived_seed(seed, source_index, 1)
    vocoder_seed = derived_seed(seed, source_index, 2)
    source, source_branch, target_branch = editor.prepare(case, condition_seed)
    waveform, steps = edit_prepared(
        editor,
        source,
        source_branch,
        target_branch,
        edit_seed,
        strength,
        tau,
        vocoder_seed,
    )
    diagnostics = {"first_pass": steps}
    if use_bridging:
        reference, diagnostics["bridging"] = bridging.reference(
            waveform, editor.sample_rate, case, condition_seed, source_index, strength
        )
        bridge_case = {
            **case,
            "target_reference_audio": str(reference),
            "target_reference_text": case["text"],
        }
        source, source_branch, target_branch = editor.prepare(
            bridge_case, condition_seed
        )
        waveform, diagnostics["second_pass"] = edit_prepared(
            editor,
            source,
            source_branch,
            target_branch,
            edit_seed,
            1.0,
            tau,
            vocoder_seed,
        )
    return waveform, diagnostics
