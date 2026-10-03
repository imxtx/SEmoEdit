import sys

import soundfile as sf
import torch

from .common import aligned_inputs, seed_all


class F5Editor:
    def __init__(self, checkpoint, vocoder_dir, upstream_root, steps, cfg, sway):
        """Load the F5 backbone and configure editing."""
        self.upstream_root = upstream_root
        self.steps = steps
        self.cfg = cfg
        self.sway = sway
        sys.path.insert(0, str(upstream_root / "src"))
        from omegaconf import OmegaConf
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT

        config = OmegaConf.load(
            self.upstream_root / "src/f5_tts/configs/F5TTS_v1_Base.yaml"
        )
        self.model = load_model(
            DiT,
            config.model.arch,
            str(checkpoint),
            vocab_file=str(checkpoint.with_name("vocab.txt")),
            device="cuda",
        )
        self.vocoder = load_vocoder(
            is_local=True, local_path=str(vocoder_dir), device="cuda"
        )
        for module in (self.model, self.vocoder):
            module.eval().requires_grad_(False)
        self.sample_rate = 24000
        self.dtype = next(self.model.parameters()).dtype
        grid = torch.linspace(0, 1, steps + 1, device="cuda", dtype=self.dtype)
        self.times = grid + sway * (torch.cos(torch.pi / 2 * grid) - 1 + grid)

    @torch.inference_mode()
    def capture(self, path, prompt_text, text, seed):
        """Capture native F5-TTS synthesis state for one condition."""
        import f5_tts.model.cfm as cfm_module
        from f5_tts.infer.utils_infer import infer_batch_process
        from f5_tts.model.utils import list_str_to_idx

        wave, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(wave.mean(1)).unsqueeze(0)
        captured = {"gain": min(1.0, float(audio.square().mean().sqrt()) / 0.1)}
        sample, odeint = self.model.sample, cfm_module.odeint

        def capture_sample(*args, **kwargs):
            """Record F5-TTS generation and its conditioning."""
            kwargs["seed"] = seed
            output, trajectory = sample(*args, **kwargs)
            captured.update(
                full=output.transpose(1, 2).clone(),
                cond_audio=kwargs["cond"].clone(),
                text=kwargs["text"],
            )
            return output, trajectory

        def capture_odeint(function, state, times, **kwargs):
            """Record the native integration time grid."""
            captured["times"] = times.clone()
            return odeint(function, state, times, **kwargs)

        # Capture the actual native trajectory and conditioning used for synthesis.
        self.model.sample, cfm_module.odeint = capture_sample, capture_odeint
        seed_all(seed)
        try:
            outputs = list(
                infer_batch_process(
                    (audio, sample_rate),
                    prompt_text,
                    [text],
                    self.model,
                    self.vocoder,
                    nfe_step=self.steps,
                    cfg_strength=self.cfg,
                    sway_sampling_coef=self.sway,
                    device="cuda",
                    progress=None,
                )
            )
            if len(outputs) != 1:
                raise ValueError("F5-TTS requires one native generation per branch")
        finally:
            self.model.sample, cfm_module.odeint = sample, odeint
        captured["condition"] = self.model.mel_spec(captured.pop("cond_audio")).to(
            self.dtype
        )
        captured["start"] = captured["condition"].shape[-1]
        captured["text_tokens"] = list_str_to_idx(
            captured.pop("text"), self.model.vocab_char_map
        ).cuda()
        return captured

    @torch.inference_mode()
    def prepare(self, case, seed):
        """Capture source and target branches with aligned generated lengths."""
        self.source = self.capture(
            case["source_audio"], case["text"], case["text"], seed
        )
        self.target = self.capture(
            case["target_reference_audio"],
            case["target_reference_text"],
            case["text"],
            seed,
        )
        torch.testing.assert_close(
            self.source["times"], self.target["times"], atol=0, rtol=0
        )
        self.times = self.source["times"]
        self.gain = self.source["gain"]
        return aligned_inputs(self.source, self.target)

    def velocity(self, query, time, branch):
        """Evaluate the frozen model's guided velocity field for one branch."""
        prediction = self.model.transformer(
            x=query.transpose(1, 2),
            cond=branch["cond"].transpose(1, 2),
            text=branch["text"],
            time=time,
            mask=None,
            cfg_infer=True,
            cache=False,
        )
        conditional, unconditional = prediction.chunk(2, dim=0)
        return (conditional + self.cfg * (conditional - unconditional)).transpose(1, 2)

    @torch.inference_mode()
    def decode(self, mel, seed):
        """Decode an edited acoustic representation with the native vocoder."""
        return self.vocoder.decode(mel.float()) * self.gain
