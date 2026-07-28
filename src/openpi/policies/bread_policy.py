import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BreadInputs(transforms.DataTransformFn):
    """UMI bread-to-toaster: two wrist cams only (the training-time top view is
    a human-worn egocam that does not exist at deployment, so base_0_rgb is
    zeroed and masked). State is 20D relative_to_first, actions are per-step
    14D chained deltas -- both baked into the pack; padding to the model's 32D
    happens in the model transforms, not here.

    See SARM2-bread-UMI docs/R1_OPENPI_CONFIG_DRAFT.md for the full config
    rationale.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        left = _parse_image(data["left_wrist_image"])
        right = _parse_image(data["right_wrist_image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": np.zeros_like(left),
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "ra_bc_weight" in data:
            # RA-BC per-sample weight (raw dP at the chunk's start frame);
            # consumed by the loss weighting patch in the train step.
            inputs["ra_bc_weight"] = np.asarray(data["ra_bc_weight"], dtype=np.float32)
        return inputs


@dataclasses.dataclass(frozen=True)
class BreadOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 14 real action dims: [dxyz, axis-angle, abs-grip] x 2 arms.
        return {"actions": np.asarray(data["actions"][..., :14])}
