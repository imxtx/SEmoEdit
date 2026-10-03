import sys

import torch

from .common import aligned_inputs, seed_all


class CosyEditor:
    def __init__(self, checkpoint, upstream_root, steps, cfg, instruct=False):
        """Load the Cosy backbone and configure editing."""
        self.instruct = instruct
        sys.path.insert(0, str(upstream_root))
        sys.path.insert(0, str(upstream_root / "third_party/Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.cosy = CosyVoice2(
            str(checkpoint), load_jit=False, load_trt=False, load_vllm=False, fp16=False
        )
        for module in (self.cosy.model.llm, self.cosy.model.flow, self.cosy.model.hift):
            module.eval().requires_grad_(False)
        self.decoder = self.cosy.model.flow.decoder
        if steps != 10:
            raise ValueError("CosyVoice2 native synthesis uses 10 steps")
        self.decoder.inference_cfg_rate = cfg
        self.sample_rate = self.cosy.sample_rate
        self.dtype = torch.float32
        self.times = 1 - torch.cos(
            torch.linspace(0, 1, 11, device="cuda") * 0.5 * torch.pi
        )

    @torch.inference_mode()
    def capture(self, path, prompt_text, text, seed):
        """Capture native CosyVoice2 synthesis state for one condition."""
        captured = {}
        forward = self.decoder.forward
        flow = self.cosy.model.flow.inference
        hift = self.cosy.model.hift.inference

        def capture_forward(*args, **kwargs):
            """Record decoder inputs and output during native synthesis."""
            captured.update(
                {
                    key: value.clone()
                    for key, value in kwargs.items()
                    if torch.is_tensor(value)
                }
            )
            result = forward(*args, **kwargs)
            captured["full"] = result[0].clone()
            return result

        def capture_flow(*args, **kwargs):
            """Record the native prompt boundary and token sequence."""
            captured.update(
                start=kwargs["prompt_feat"].shape[1], token=kwargs["token"].clone()
            )
            return flow(*args, **kwargs)

        def capture_hift(*args, **kwargs):
            """Capture the vocoder random state before waveform synthesis."""
            captured.update(
                rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()
            )
            return hift(*args, **kwargs)

        # Hooks expose the prompt boundary, flow inputs, and vocoder RNG state.
        self.decoder.forward = capture_forward
        self.cosy.model.flow.inference = capture_flow
        self.cosy.model.hift.inference = capture_hift
        seed_all(seed)
        try:
            if self.instruct:
                instruction = (
                    prompt_text
                    if prompt_text.endswith("<|endofprompt|>")
                    else prompt_text + "<|endofprompt|>"
                )
                captured["instruction"] = instruction
                outputs = list(
                    self.cosy.inference_instruct2(text, instruction, path, stream=False)
                )
            else:
                outputs = list(
                    self.cosy.inference_zero_shot(text, prompt_text, path, stream=False)
                )
            if len(outputs) != 1:
                raise ValueError(
                    "CosyVoice2 requires one native generation per branch; text was split"
                )
            captured["wave"] = outputs[0]["tts_speech"].clone()
        finally:
            self.decoder.forward = forward
            self.cosy.model.flow.inference = flow
            self.cosy.model.hift.inference = hift
        return captured

    @torch.inference_mode()
    def prepare(self, case, seed):
        """Capture source and target branches with aligned generated lengths."""
        if self.instruct:
            self.source = self.capture(
                case["source_audio"], case["source_instruction"], case["text"], seed
            )
            self.target = self.capture(
                case["source_audio"], case["target_instruction"], case["text"], seed
            )
        else:
            self.source = self.capture(
                case["source_audio"], case["text"], case["text"], seed
            )
            self.target = self.capture(
                case["target_reference_audio"],
                case["target_reference_text"],
                case["text"],
                seed,
            )
        return aligned_inputs(self.source, self.target)

    def velocity(self, query, time, branch):
        """Evaluate the frozen model's guided velocity field for one branch."""
        query_pair = torch.zeros(
            (2, 80, query.shape[-1]), device=query.device, dtype=branch["spks"].dtype
        )
        query_pair[:] = query
        prediction = self.decoder.forward_estimator(
            query_pair,
            branch["mask"].repeat(2, 1, 1),
            torch.cat((branch["mu"], torch.zeros_like(branch["mu"]))),
            time.expand(2),
            torch.cat((branch["spks"], torch.zeros_like(branch["spks"]))),
            torch.cat((branch["cond"], torch.zeros_like(branch["cond"]))),
            streaming=False,
        )
        conditional, unconditional = prediction.chunk(2, dim=0)
        return (
            1 + self.decoder.inference_cfg_rate
        ) * conditional - self.decoder.inference_cfg_rate * unconditional

    @torch.inference_mode()
    def decode(self, mel, seed):
        # Replay the native vocoder state so comparisons share its stochastic path.
        """Decode an edited acoustic representation with the native vocoder."""
        torch.set_rng_state(self.source["rng_cpu"])
        torch.cuda.set_rng_state(self.source["rng_cuda"])
        return self.cosy.model.hift.inference(
            speech_feat=mel, cache_source=torch.zeros(1, 1, 0)
        )[0]
