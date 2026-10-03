"""IndexTTS2 native s2mel capture and SEmoEdit adapter."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml

from .common import resize, seed_all


def index_branch(captured, length):
    """Resize only the generated semantic region, preserving the prompt prefix."""
    start = captured["start"]
    semantic = captured["mu"].transpose(1, 2)
    semantic = torch.cat(
        (semantic[..., :start], resize(semantic[..., start:], length)), -1
    ).transpose(1, 2)
    return {
        "prefix": captured["full"][..., :start].clone(),
        "prompt": functional.pad(captured["prompt"], (0, length)),
        "mu": semantic,
        "style": captured["style"],
        "x_lens": captured["x_lens"].new_tensor([start + length]),
        "cfg": captured["cfg"],
    }


class IndexEditor:
    def __init__(self, checkpoint, upstream_root, steps, cfg, instruct=False):
        """Load the Index backbone and configure editing."""
        self.instruct = instruct
        if steps != 25 or cfg != 0.7:
            raise ValueError("IndexTTS2 native synthesis uses 25 steps and CFG=0.7")
        sys.path.insert(0, str(upstream_root))
        from indextts.infer_v2 import IndexTTS2

        if instruct:
            config = yaml.safe_load((checkpoint / "config.yaml").read_text())
            emotion_path = checkpoint / config["qwen_emo_path"]
            if not emotion_path.is_dir():
                raise FileNotFoundError(
                    f"Missing local IndexTTS emotion parser: {emotion_path}"
                )
        cache = checkpoint / "hf_cache"
        aux_paths = {
            "w2v_bert": str(cache / "w2v-bert-2.0"),
            "semantic_codec": str(cache / "semantic_codec/model.safetensors"),
            "campplus": str(cache / "campplus_cn_common.bin"),
            "bigvgan": str(cache / "bigvgan"),
        }
        for path in aux_paths.values():
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Missing local IndexTTS auxiliary asset: {path}"
                )
        self.tts = IndexTTS2(
            cfg_path=str(checkpoint / "config.yaml"),
            model_dir=str(checkpoint),
            use_fp16=False,
            device="cuda:0",
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            use_qwen_emo=instruct,
            aux_paths=aux_paths,
        )
        for module in (
            self.tts.gpt,
            self.tts.s2mel,
            self.tts.bigvgan,
            self.tts.semantic_model,
            self.tts.semantic_codec,
            self.tts.campplus_model,
        ):
            module.eval().requires_grad_(False)
        if instruct:
            self.tts.qwen_emo.model.eval().requires_grad_(False)
        self.cfm = self.tts.s2mel.models["cfm"]
        self.sample_rate = int(self.tts.cfg.s2mel.preprocess_params.sr)
        self.dtype = torch.float32
        self.times = torch.linspace(0, 1, 26, device="cuda")

    @torch.inference_mode()
    def capture(self, path, text, seed, instruction=None):
        """Capture native IndexTTS2 synthesis state for one condition."""
        captures = []
        original = self.cfm.inference
        emotion = {}
        if self.instruct:
            original_emotion = self.tts.qwen_emo.inference

            def capture_emotion(description):
                """Record the emotion vector returned by IndexTTS2."""
                result = original_emotion(description)
                emotion.update(result)
                return result

            self.tts.qwen_emo.inference = capture_emotion

        def capture_flow(
            mu,
            x_lens,
            prompt,
            style,
            f0,
            n_timesteps,
            temperature=1.0,
            inference_cfg_rate=0.5,
        ):
            """Record the native flow inputs and generated state."""
            if n_timesteps != 25 or inference_cfg_rate != 0.7 or f0 is not None:
                raise ValueError(
                    "Expected IndexTTS2 native 25-step, CFG=0.7, no-F0 synthesis"
                )
            if captures:
                raise ValueError(
                    "IndexTTS2 split this case into multiple segments; one native region is required"
                )
            result = original(
                mu,
                x_lens,
                prompt,
                style,
                f0,
                n_timesteps,
                temperature=temperature,
                inference_cfg_rate=inference_cfg_rate,
            )
            captures.append(
                {
                    "full": result.clone(),
                    "start": prompt.shape[-1],
                    "mu": mu.clone(),
                    "prompt": prompt.clone(),
                    "style": style.clone(),
                    "x_lens": x_lens.clone(),
                    "cfg": inference_cfg_rate,
                }
            )
            return result

        # Intercept one native s2mel call without changing the model's synthesis path.
        seed_all(seed)
        self.cfm.inference = capture_flow
        try:
            self.tts.infer(
                spk_audio_prompt=path,
                emo_audio_prompt=None if self.instruct else path,
                emo_alpha=1.0,
                text=text,
                output_path=None,
                use_random=False,
                use_emo_text=self.instruct,
                emo_text=instruction,
                max_text_tokens_per_segment=int(self.tts.cfg.gpt.max_text_tokens) - 1,
            )
        finally:
            self.cfm.inference = original
            if self.instruct:
                self.tts.qwen_emo.inference = original_emotion
        if len(captures) != 1:
            raise ValueError("IndexTTS2 did not produce exactly one native mel region")
        if self.instruct:
            captures[0].update(instruction=instruction, emotion_vector=emotion)
        return captures[0]

    @torch.inference_mode()
    def prepare(self, case, seed):
        """Capture source and target branches with aligned generated lengths."""
        if self.instruct:
            self.source = self.capture(
                case["source_audio"], case["text"], seed, case["source_instruction"]
            )
            self.target = self.capture(
                case["source_audio"], case["text"], seed, case["target_instruction"]
            )
        else:
            self.source = self.capture(case["source_audio"], case["text"], seed)
            self.target = self.capture(
                case["target_reference_audio"], case["text"], seed
            )
        length = self.target["full"].shape[-1] - self.target["start"]
        source = resize(self.source["full"][..., self.source["start"] :], length)
        return (
            source,
            index_branch(self.source, length),
            index_branch(self.target, length),
        )

    def velocity(self, query, time, branch):
        """Evaluate the frozen model's guided velocity field for one branch."""
        query = query.clone()
        # The native estimator receives the prompt separately, not as query state.
        query[..., : branch["prefix"].shape[-1]] = 0
        prediction = self.cfm.estimator(
            torch.cat((query, query)),
            torch.cat((branch["prompt"], torch.zeros_like(branch["prompt"]))),
            branch["x_lens"],
            time.expand(2),
            torch.cat((branch["style"], torch.zeros_like(branch["style"]))),
            torch.cat((branch["mu"], torch.zeros_like(branch["mu"]))),
        )
        conditional, unconditional = prediction.chunk(2)
        return (1 + branch["cfg"]) * conditional - branch["cfg"] * unconditional

    @torch.inference_mode()
    def decode(self, mel, seed):
        """Decode an edited acoustic representation with the native vocoder."""
        return self.tts.bigvgan(mel.float())
