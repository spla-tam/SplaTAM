import copy
import warnings
from collections import OrderedDict
from typing import List, Union

import numpy as np
import torch

__all__ = [
    "normalize_image",
    "channels_first",
    "scale_intrinsics",
    "pointquaternion_to_homogeneous",
    "poses_to_transforms",
    "create_label_image",
]


def normalize_image(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Normalizes RGB image values from :math:`[0, 255]` range to :math:`[0, 1]` range.

    Args:
        rgb (torch.Tensor or numpy.ndarray): RGB image in range :math:`[0, 255]`

    Returns:
        torch.Tensor or numpy.ndarray: Normalized RGB image in range :math:`[0, 1]`

    Shape:
        - rgb: :math:`(*)` (any shape)
        - Output: Same shape as input :math:`(*)`
    """
    if torch.is_tensor(rgb):
        return rgb.float() / 255
    elif isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255
    else:
        raise TypeError("Unsupported input rgb type: %r" % type(rgb))


def channels_first(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Converts from channels last representation :math:`(*, H, W, C)` to channels first representation
    :math:`(*, C, H, W)`

    Args:
        rgb (torch.Tensor or numpy.ndarray): :math:`(*, H, W, C)` ordering `(*, height, width, channels)`

    Returns:
        torch.Tensor or numpy.ndarray: :math:`(*, C, H, W)` ordering

    Shape:
        - rgb: :math:`(*, H, W, C)`
        - Output: :math:`(*, C, H, W)`
    """
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):
        raise TypeError("Unsupported input rgb type {}".format(type(rgb)))

    if rgb.ndim < 3:
        raise ValueError(
            "Input rgb must contain atleast 3 dims, but had {} dims.".format(rgb.ndim)
        )
    if rgb.shape[-3] < rgb.shape[-1]:
        msg = "Are you sure that the input is correct? Number of channels exceeds height of image: %r > %r"
        warnings.warn(msg % (rgb.shape[-1], rgb.shape[-3]))
    ordering = list(range(rgb.ndim))
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]

    if isinstance(rgb, np.ndarray):
        return np.ascontiguousarray(rgb.transpose(*ordering))
    elif torch.is_tensor(rgb):
        return rgb.permute(*ordering).contiguous()


def scale_intrinsics(
    intrinsics: Union[np.ndarray, torch.Tensor],
    h_ratio: Union[float, int],
    w_ratio: Union[float, int],
):
    r"""Scales the intrinsics appropriately for resized frames where
    :math:`h_\text{ratio} = h_\text{new} / h_\text{old}` and :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Args:
        intrinsics (numpy.ndarray or torch.Tensor): Intrinsics matrix of original frame
        h_ratio (float or int): Ratio of new frame's height to old frame's height
            :math:`h_\text{ratio} = h_\text{new} / h_\text{old}`
        w_ratio (float or int): Ratio of new frame's width to old frame's width
            :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Returns:
        numpy.ndarray or torch.Tensor: Intrinsics matrix scaled approprately for new frame size

    Shape:
        - intrinsics: :math:`(*, 3, 3)` or :math:`(*, 4, 4)`
        - Output: Matches `intrinsics` shape, :math:`(*, 3, 3)` or :math:`(*, 4, 4)`

    """
    if isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = intrinsics.astype(np.float32).copy()
    elif torch.is_tensor(intrinsics):
        scaled_intrinsics = intrinsics.to(torch.float).clone()
    else:
        raise TypeError("Unsupported input intrinsics type {}".format(type(intrinsics)))
    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
        raise ValueError(
            "intrinsics must have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(
                intrinsics.shape
            )
        )
    if (intrinsics[..., -1, -1] != 1).any() or (intrinsics[..., 2, 2] != 1).any():
        warnings.warn(
            "Incorrect intrinsics: intrinsics[..., -1, -1] and intrinsics[..., 2, 2] should be 1."
        )

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
    return scaled_intrinsics


def pointquaternion_to_homogeneous(
    pointquaternions: Union[np.ndarray, torch.Tensor], eps: float = 1e-12
):
    r"""Converts 3D point and unit quaternions :math:`(t_x, t_y, t_z, q_x, q_y, q_z, q_w)` to
    homogeneous transformations [R | t] where :math:`R` denotes the :math:`(3, 3)` rotation matrix and :math:`T`
    denotes the :math:`(3, 1)` translation matrix:

    .. math::

        \left[\begin{array}{@{}c:c@{}}
        R & T \\ \hdashline
        \begin{array}{@{}ccc@{}}
            0 & 0 & 0
        \end{array}  & 1
        \end{array}\right]

    Args:
        pointquaternions (numpy.ndarray or torch.Tensor): 3D point positions and unit quaternions
            :math:`(tx, ty, tz, qx, qy, qz, qw)` where :math:`(tx, ty, tz)` is the 3D position and
            :math:`(qx, qy, qz, qw)` is the unit quaternion.
        eps (float): Small value, to avoid division by zero. Default: 1e-12

    Returns:
        numpy.ndarray or torch.Tensor: Homogeneous transformation matrices.

    Shape:
        - pointquaternions: :math:`(*, 7)`
        - Output: :math:`(*, 4, 4)`

    """
    if not (
        isinstance(pointquaternions, np.ndarray) or torch.is_tensor(pointquaternions)
    ):
        raise TypeError(
            '"pointquaternions" must be of type "np.ndarray" or "torch.Tensor". Got {0}'.format(
                type(pointquaternions)
            )
        )
    if not isinstance(eps, float):
        raise TypeError('"eps" must be of type "float". Got {0}.'.format(type(eps)))
    if pointquaternions.shape[-1] != 7:
        raise ValueError(
            '"pointquaternions" must be of shape (*, 7). Got {0}.'.format(
                pointquaternions.shape
            )
        )

    output_shape = (*pointquaternions.shape[:-1], 4, 4)
    if isinstance(pointquaternions, np.ndarray):
        t = pointquaternions[..., :3].astype(np.float32)
        q = pointquaternions[..., 3:7].astype(np.float32)
        transform = np.zeros(output_shape, dtype=np.float32)
    else:
        t = pointquaternions[..., :3].float()
        q = pointquaternions[..., 3:7].float()
        transform = torch.zeros(
            output_shape, dtype=torch.float, device=pointquaternions.device
        )

    q_norm = (0.5 * (q ** 2).sum(-1)[..., None]) ** 0.5
    q /= (
        torch.max(q_norm, torch.tensor(eps))
        if torch.is_tensor(q_norm)
        else np.maximum(q_norm, eps)
    )

    if isinstance(q, np.ndarray):
        q = np.matmul(q[..., None], q[..., None, :])
    else:
        q = torch.matmul(q.unsqueeze(-1), q.unsqueeze(-2))

    txx = q[..., 0, 0]
    tyy = q[..., 1, 1]
    tzz = q[..., 2, 2]
    txy = q[..., 0, 1]
    txz = q[..., 0, 2]
    tyz = q[..., 1, 2]
    twx = q[..., 0, 3]
    twy = q[..., 1, 3]
    twz = q[..., 2, 3]
    transform[..., 0, 0] = 1.0
    transform[..., 1, 1] = 1.0
    transform[..., 2, 2] = 1.0
    transform[..., 3, 3] = 1.0
    transform[..., 0, 0] -= tyy + tzz
    transform[..., 0, 1] = txy - twz
    transform[..., 0, 2] = txz + twy
    transform[..., 1, 0] = txy + twz
    transform[..., 1, 1] -= txx + tzz
    transform[..., 1, 2] = tyz - twx
    transform[..., 2, 0] = txz - twy
    transform[..., 2, 1] = tyz + twx
    transform[..., 2, 2] -= txx + tyy
    transform[..., :3, 3] = t

    return transform


def poses_to_transforms(poses: Union[np.ndarray, List[np.ndarray]]):
    r"""Converts poses to transformations w.r.t. the first frame in the sequence having identity pose

    Args:
        poses (numpy.ndarray or list of numpy.ndarray): Sequence of poses in `numpy.ndarray` format.

    Returns:
        numpy.ndarray or list of numpy.ndarray: Sequence of frame to frame transformations where initial
            frame is transformed to have identity pose.

    Shape:
        - poses: Could be `numpy.ndarray` of shape :math:`(N, 4, 4)`, or list of `numpy.ndarray`s of shape
          :math:`(4, 4)`
        - Output: Of same shape as input `poses`
    """
    transformations = copy.deepcopy(poses)
    for i in range(len(poses)):
        if i == 0:
            transformations[i] = np.eye(4)
        else:
            transformations[i] = np.linalg.inv(poses[i - 1]).dot(poses[i])
    return transformations


def create_label_image(prediction: np.ndarray, color_palette: OrderedDict):
    r"""Creates a label image, given a network prediction (each pixel contains class index) and a color palette.

    Args:
        prediction (numpy.ndarray): Predicted image where each pixel contains an integer,
            corresponding to its class label.
        color_palette (OrderedDict): Contains RGB colors (`uint8`) for each class.

    Returns:
        numpy.ndarray: Label image with the given color palette

    Shape:
        - prediction: :math:`(H, W)`
        - Output: :math:`(H, W)`
    """

    label_image = np.zeros(
        (prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8
    )
    for idx, color in enumerate(color_palette):
        label_image[prediction == idx] = color
    return label_image
