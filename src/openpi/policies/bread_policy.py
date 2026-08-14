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
    """UMI bread-to-toaster: two wrist cams, plus optionally the high/ego view
    in base_0_rgb. State is 20D relative_to_first, actions are per-step
    14D chained deltas -- both baked into the pack; padding to the model's 32D
    happens in the model transforms, not here.

    use_high_cam=False (R1/R2 behaviour): base_0_rgb is zeroed and masked --
    the training-time top view is a human-worn egocam that does not exist at
    deployment. use_high_cam=True (r2_hicam configs): the pack's cam_high
    feeds base_0_rgb with mask True. KNOWN DOMAIN GAP: training high cam is
    human-worn (no extrinsic); deployment must supply the robot's fixed top
    RealSense as "high_image" -- every consumer (offline eval, serving
    bridge) has to send it or the model sees an OOD zero image.

    See SARM2-bread-UMI docs/R1_OPENPI_CONFIG_DRAFT.md for the full config
    rationale.
    """

    model_type: _model.ModelType
    use_high_cam: bool = False

    def __call__(self, data: dict) -> dict:
        left = _parse_image(data["left_wrist_image"])
        right = _parse_image(data["right_wrist_image"])

        if self.use_high_cam:
            high = _parse_image(data["high_image"])
            base_image, base_mask = high, np.True_
        else:
            base_image = np.zeros_like(left)
            base_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": base_mask,
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


def _rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: (3,) rotation vector -> (3,3) rotation matrix."""
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues: (3,3) rotation matrix -> (3,) rotation vector."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3)
    if theta > np.pi - 1e-6:
        # near-pi fallback via the symmetric part (chunk-relative rotations
        # in a 1.3 s window never get here in practice)
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diagonal(A), 0))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(theta)) * theta


@dataclasses.dataclass(frozen=True)
class BreadChunkRelativeActions(transforms.DataTransformFn):
    """r3 action target: convert the horizon chunk of per-step chained deltas
    (as baked in the r-packs: row t holds inv(T_t)@T_{t+1}) into
    relative-to-chunk-start targets sharing ONE anchor (the current frame):

        T_rel[k] = inv(T_t) @ T_{t+k+1} = d_t . d_{t+1} ... d_{t+k}

    encoded [xyz, rotvec] per arm; absolute grippers pass through. This is the
    AgRobotics/Zhengmao convention -- the executor composes every target with
    the same cached anchor pose and never accumulates predictions. Episode-end
    padding rows hold identity deltas, so padded targets freeze at the last
    real pose (consistent with "hold position").

    No-op at inference time (no "actions" key): the policy then OUTPUTS
    chunk-relative targets which the serving bridge decodes against its anchor.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        acts = np.asarray(data["actions"])
        out = np.array(acts, dtype=np.float64, copy=True)
        for off in (0, 7):
            T = np.eye(4)
            for k in range(acts.shape[0]):
                d = np.eye(4)
                d[:3, :3] = _rotvec_to_matrix(acts[k, off + 3:off + 6].astype(np.float64))
                d[:3, 3] = acts[k, off:off + 3]
                T = T @ d
                out[k, off:off + 3] = T[:3, 3]
                out[k, off + 3:off + 6] = _matrix_to_rotvec(T[:3, :3])
        data["actions"] = out.astype(acts.dtype)
        return data


@dataclasses.dataclass(frozen=True)
class BreadOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 14 real action dims: [dxyz, axis-angle, abs-grip] x 2 arms.
        # (For chunk_relative configs these are chunk-start-relative targets,
        # decoded against the anchor by the serving bridge.)
        return {"actions": np.asarray(data["actions"][..., :14])}
