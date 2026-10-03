import random

import numpy as np
import torch
import torch.nn.functional as functional


def seed_all(seed):
    """Seed the random generators used by the model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize(tensor, length):
    """Linearly resize an acoustic region to the requested length."""
    if tensor.shape[-1] == length:
        return tensor.clone()
    return functional.interpolate(
        tensor, size=length, mode="linear", align_corners=False
    )


def aligned_inputs(source, target):
    """Match generated lengths while retaining each condition's native prefix."""
    length = target["full"].shape[-1] - target["start"]
    edited_source = resize(source["full"][..., source["start"] :], length)
    branches = []
    for captured in (source, target):
        start = captured["start"]
        branch = {"prefix": captured["full"][..., :start].clone()}
        if "mu" in captured:
            branch.update(
                mu=torch.cat(
                    (
                        captured["mu"][..., :start],
                        resize(captured["mu"][..., start:], length),
                    ),
                    -1,
                ),
                cond=functional.pad(captured["cond"][..., :start], (0, length)),
                spks=captured["spks"],
                mask=torch.ones(
                    (1, 1, start + length),
                    device=edited_source.device,
                    dtype=edited_source.dtype,
                ),
            )
        else:
            branch.update(
                cond=functional.pad(captured["condition"], (0, length)),
                text=captured["text_tokens"],
            )
        branches.append(branch)
    return edited_source, *branches
