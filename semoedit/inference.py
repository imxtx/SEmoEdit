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


def edit(editor, case, strength=1.0, seed=42, tau=0.0, source_index=0):
    """Generate source speech from its reference, then edit toward the target.

    case contains source_audio, text, target_reference_audio and
    target_reference_text. The source reference must speak case['text'].
    strength controls direct transport; benchmark emotion bridging is a
    separate two-pass procedure orchestrated by scripts/run.py.
    Returns (waveform tensor, per-step diagnostics).
    """
    source, source_branch, target_branch = editor.prepare(
        case, derived_seed(seed, source_index, 0)
    )
    return edit_prepared(
        editor,
        source,
        source_branch,
        target_branch,
        derived_seed(seed, source_index, 1),
        strength,
        tau,
        derived_seed(seed, source_index, 2),
    )
