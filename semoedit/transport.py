"""Source-anchored Euler transport with independent reference prefixes."""

import numpy as np
import torch


def derived_seed(seed, source_index, stream):
    """Keep conditioning, editing, and vocoder random streams independent."""
    return int(
        np.random.SeedSequence([seed, source_index, stream]).generate_state(1)[0]
    )


@torch.inference_mode()
def integrate(
    source, source_branch, target_branch, times, velocity, seed, strength, tau=0.0
):
    """Integrate the source-to-target velocity difference on the native time grid."""
    edited = source.clone()
    prefix_length = max(
        source_branch["prefix"].shape[-1], target_branch["prefix"].shape[-1]
    )
    generator = torch.Generator(device=source.device)
    diagnostics = []
    for step, (time, next_time) in enumerate(zip(times[:-1], times[1:])):
        generator.manual_seed(derived_seed(seed, step, 0))
        noise = torch.randn(
            source.shape, generator=generator, device=source.device, dtype=source.dtype
        )
        prefix_noise = torch.randn(
            (1, source.shape[1], prefix_length),
            generator=generator,
            device=source.device,
            dtype=source.dtype,
        )
        # Both queries receive the same perturbation; only the target state evolves.
        source_query = (1 - time) * noise + time * source
        target_query = source_query + (edited - source)
        velocities = []
        for branch, query in (
            (source_branch, source_query),
            (target_branch, target_query),
        ):
            prefix = branch["prefix"]
            length = prefix.shape[-1]
            # A shared suffix couples prefixes even when their native lengths differ.
            noisy_prefix = (1 - time) * prefix_noise[..., -length:] + time * prefix
            full_query = torch.cat((noisy_prefix, query), dim=-1)
            velocities.append(velocity(full_query, time, branch)[..., length:])
        difference = velocities[1] - velocities[0]
        # tau excludes early solver intervals without changing their time grid.
        if time >= tau:
            edited = edited + strength * (next_time - time) * difference
        diagnostics.append(
            {
                "step": step,
                "time": float(time),
                "dt": float(next_time - time),
                "difference_rms": float(difference.float().square().mean().sqrt()),
            }
        )
    return edited, diagnostics
