import logging
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

logger = logging.getLogger("dinov3")


def load_pretrained_backbone(model, checkpoint_path):
    """Strictly load an official backbone or an exported SSL teacher backbone.

    Convert against each destination's mesh and placements, including aliased
    parameters (ConvNeXt norm/norms.3). Projection heads are not restored.
    """
    path = Path(checkpoint_path).expanduser()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "teacher" in state:
        state = state["teacher"]
    if any(key.startswith("backbone.") for key in state):
        state = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
    target_state = model.state_dict()
    missing = target_state.keys() - state.keys()
    unexpected = state.keys() - target_state.keys()
    if missing or unexpected:
        raise RuntimeError(f"Backbone checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, tensor in state.items():
        target = target_state[key]
        if tensor.shape != target.shape:
            raise RuntimeError(f"Backbone shape mismatch for {key}: {tuple(tensor.shape)} vs {tuple(target.shape)}")
        if isinstance(target, DTensor):
            state[key] = distribute_tensor(
                tensor.to(device=target.device, dtype=target.dtype),
                device_mesh=target.device_mesh,
                placements=target.placements,
                src_data_rank=None,
            )
    result = model.load_state_dict(state, strict=True)
    logger.info("Loaded pretrained backbone strictly from %s: %s", path, result)
    return result
