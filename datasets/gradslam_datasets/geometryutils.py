"""
Projective geometry utility functions.
"""

from typing import Optional

import torch
from kornia.geometry.linalg import compose_transformations, inverse_transformation


def homogenize_points(pts: torch.Tensor):
    r"""Convert a set of points to homogeneous coordinates.

    Args:
        pts (torch.Tensor): Tensor containing points to be homogenized.

    Shape:
        pts: N x 3 (N-points, and (usually) 3 dimensions)
        (returns): N x 4

    Returns:
        (torch.Tensor): Homogeneous coordinates of pts

    """
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    return torch.nn.functional.pad(pts, (0, 1), "constant", 1.0)


def unhomogenize_points(pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert a set of points from homogeneous coordinates to Euclidean
    coordinates.

    This is usually done by taking each point (x, y, z, w) and dividing it by
    the last coordinate (w).

    Args:
        pts (torch.Tensor): Tensor containing points to be unhomogenized.

    Shape:
        pts: N x 4 (N-points, and usually 4 dimensions per point)
        (returns): N x 3

    Returns:
        (torch.Tensor): 'Unhomogenized' points

    """
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    # Get points with the last coordinate (scale) as 0 (points at infinity)
    w: torch.Tensor = pts[..., -1:]
    # Determine the scale factor each point needs to be multiplied by
    # For points at infinity, use a scale factor of 1 (used by OpenCV
    # and by kornia)
    # https://github.com/opencv/opencv/pull/14411/files
    scale: torch.Tensor = torch.where(torch.abs(w) > eps, 1.0 / w, torch.ones_like(w))

    return scale * pts[..., :-1]


def quaternion_to_axisangle(quat: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to an axis angle.

    Args:
        quat (torch.Tensor): Quaternion (qx, qy, qz, qw) (shape:
            :math:`* \times 4`)

    Return:
        axisangle (torch.Tensor): Axis-angle representation. (shape:
            :math:`* \times 3`)

    """
    if not torch.is_tensor(quat):
        raise TypeError(
            "Expected input quat to be of type torch.Tensor."
            " Got {} instead.".format(type(quat))
        )
    if not quat.shape[-1] == 4:
        raise ValueError(
            "Last dim of input quat must be of shape 4. "
            "Got {} instead.".format(quat.shape[-1])
        )

    # Unpack quat
    qx: torch.Tensor = quat[..., 0]
    qy: torch.Tensor = quat[..., 1]
    qz: torch.Tensor = quat[..., 2]
    sin_sq_theta: torch.Tensor = qx * qx + qy * qy + qz * qz
    sin_theta: torch.Tensor = torch.sqrt(sin_sq_theta)
    cos_theta: torch.Tensor = quat[..., 3]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_sq_theta > 0.0, k_pos, k_neg)

    axisangle: torch.Tensor = torch.zeros_like(quat)[..., :3]
    axisangle[..., 0] = qx * k
    axisangle[..., 1] = qy * k
    axisangle[..., 2] = qz * k

    return axisangle


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12):
    r"""Normalize a quaternion. The quaternion should be in (x, y, z, w)
    format.

    Args:
        quaternion (torch.Tensor): Quaternion to be normalized
            (shape: (*, 4))
        eps (Optional[bool]): Small value, to avoid division by zero
            (default: 1e-12).

    Returns:
        (torch.Tensor): Normalized quaternion (shape: (*, 4))
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}.".format(quaternion.shape)
        )
    return torch.nn.functional.normalize(quaternion, p=2, dim=-1, eps=eps)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix. The quaternion should
    be in (x, y, z, w) format.

    Args:
        quaternion (torch.Tensor): Quaternion to be converted (shape: (*, 4))

    Return:
        (torch.Tensor): Rotation matrix (shape: (*, 3, 3))

    """
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape)
        )

    # Normalize the input quaternion
    quaternion_norm = normalize_quaternion(quaternion)

    # Unpack the components of the normalized quaternion
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # Compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        [
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def inverse_transfom_3d(trans: torch.Tensor):
    r"""Inverts a 4 x 4 3D transformation matrix.

    Args:
        trans (torch.Tensor): transformation matrix (shape:
            :math:`* \times 4 \times 4`)

    Returns:
        trans_inv (torch.Tensor): inverse of `trans`

    """
    if not torch.is_tensor(trans):
        raise TypeError(
            "Expected input trans of type torch.Tensor. Got {} instead.".format(
                type(trans)
            )
        )
    if not trans.dim() in (2, 3) and trans.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(trans.shape)
        )

    # Unpack tensor into rotation and tranlation components
    rmat: torch.Tensor = trans[..., :3, :3]
    tvec: torch.Tensor = trans[..., :3, 3]

    # Compute the inverse
    rmat_inv: torch.Tensor = torch.transpose(rmat, -1, -2)
    tvec_inv: torch.Tensor = torch.matmul(-rmat_inv, tvec)

    # Pack the inverse rotation and translation into tensor
    trans_inv: torch.Tensor = torch.zeros_like(trans)
    trans_inv[..., :3, :3] = rmat_inv
    trans_inv[..., :3, 3] = tvec_inv
    trans_inv[..., -1, -1] = 1.0

    return trans_inv


def compose_transforms_3d(trans1: torch.Tensor, trans2: torch.Tensor) -> torch.Tensor:
    r"""Compose two homogeneous 3D transforms.

    Args:
        trans1 (torch.Tensor): first transformation (shape:
            :math:`* \times 4 \times 4`)
        trans2 (torch.Tensor): second transformation (shape:
            :math:`* \times 4 \times 4`)

    Returns:
        trans_cat (torch.Tensor): composed transformation matrix.

    """
    if not torch.is_tensor(trans1):
        raise TypeError(
            "Expected input trans1 of type torch.Tensor. Got {} instead.".format(
                type(trans1)
            )
        )
    if not trans1.dim() in (2, 3) and trans1.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(
                trans1.shape
            )
        )
    if not torch.is_tensor(trans2):
        raise TypeError(
            "Expected input trans2 of type torch.Tensor. Got {} instead.".format(
                type(trans2)
            )
        )
    if not trans2.dim() in (2, 3) and trans2.shape[-2, :] == (4, 4):
        raise ValueError(
            "Input size must be N x 4 x 4 or 4 x 4. Got {} instead.".format(
                trans2.shape
            )
        )
    assert (
        trans1.shape == trans2.shape
    ), "Both input transformations must have the same shape."

    # Unpack into rmat, tvec
    rmat1: torch.Tensor = trans1[..., :3, :3]
    rmat2: torch.Tensor = trans2[..., :3, :3]
    tvec1: torch.Tensor = trans1[..., :3, 3]
    tvec2: torch.Tensor = trans2[..., :3, 3]

    # Compute the composition
    rmat_cat: torch.Tensor = torch.matmul(rmat1, rmat2)
    tvec_cat: torch.Tensor = torch.matmul(rmat1, tvec2) + tvec1

    # Pack into output tensor
    trans_cat: torch.Tensor = torch.zeros_like(trans1)
    trans_cat[..., :3, :3] = rmat_cat
    trans_cat[..., :3, 3] = tvec_cat
    trans_cat[..., -1, -1] = 1.0

    return trans_cat


def transform_pts_3d(pts_b: torch.Tensor, t_ab: torch.Tensor) -> torch.Tensor:
    r"""Transforms a set of points `pts_b` from frame `b` to frame `a`, given an SE(3)
    transformation matrix `t_ab`

    Args:
        pts_b (torch.Tensor): points to be transformed (shape: :math:`N \times 3`)
        t_ab (torch.Tensor): homogenous 3D transformation matrix (shape: :math:`4 \times 4`)

    Returns:
        pts_a (torch.Tensor): `pts_b` transformed to the coordinate frame `a`
            (shape: :math:`N \times 3`)

    """
    if not torch.is_tensor(pts_b):
        raise TypeError(
            "Expected input pts_b of type torch.Tensor. Got {} instead.".format(
                type(pts_b)
            )
        )
    if not torch.is_tensor(t_ab):
        raise TypeError(
            "Expected input t_ab of type torch.Tensor. Got {} instead.".format(
                type(t_ab)
            )
        )
    if pts_b.dim() < 2:
        raise ValueError(
            "Expected pts_b to have at least 2 dimensions. Got {} instead.".format(
                pts_b.dim()
            )
        )
    if t_ab.dim() != 2:
        raise ValueError(
            "Expected t_ab to have 2 dimensions. Got {} instead.".format(t_ab.dim())
        )
    if t_ab.shape[0] != 4 or t_ab.shape[1] != 4:
        raise ValueError(
            "Expected t_ab to have shape (4, 4). Got {} instead.".format(t_ab.shape)
        )

    # Determine if we need to homogenize the points
    if pts_b.shape[-1] == 3:
        pts_b = homogenize_points(pts_b)

    # Apply the transformation

    if pts_b.dim() == 4:
        pts_a_homo = torch.matmul(
            t_ab.unsqueeze(0).unsqueeze(0), pts_b.unsqueeze(-1)
        ).squeeze(-1)
    else:
        pts_a_homo = torch.matmul(t_ab.unsqueeze(0), pts_b.unsqueeze(-1))
    pts_a = unhomogenize_points(pts_a_homo)

    return pts_a[..., :3]


def transform_pts_nd_KF(pts, tform):
    r"""Applies a transform to a set of points.

    Args:
        pts (torch.Tensor): Points to be transformed (shape: B x N x D)
            (N points, D dimensions per point; B -> batchsize)
        tform (torch.Tensor): Transformation to be applied
            (shape: B x D+1 x D+1)

    Returns:
        (torch.Tensor): Transformed points (B, N, D)

    """
    if not pts.shape[0] == tform.shape[0]:
        raise ValueError("Input batchsize must be the same for both  tensors")
    if not pts.shape[-1] + 1 == tform.shape[-1]:
        raise ValueError(
            "Last input dims must differ by one, i.e., "
            "pts.shape[-1] + 1 should be equal to tform.shape[-1]."
        )

    # Homogenize
    pts_homo = homogenize_points(pts)

    # Transform
    pts_homo_tformed = torch.matmul(tform.unsqueeze(1), pts_homo.unsqueeze(-1))
    pts_homo_tformed = pts_homo_tformed.squeeze(-1)

    # Unhomogenize
    return unhomogenize_points(pts_homo_tformed)


def relative_transform_3d(
    trans_01: torch.Tensor, trans_02: torch.Tensor
) -> torch.Tensor:
    r"""Given two 3D homogeneous transforms `trans_01` and `trans_02`
    in the global frame '0', this function returns a relative
    transform `trans_12`.

    Args:
        trans_01 (torch.Tensor): first transformation (shape:
            :math:`* \times 4 \times 4`)
        trans_02 (torch.Tensor): second transformation (shape:
            :math:`* \times 4 \times 4`)

    Returns:
        trans_12 (torch.Tensor): composed transformation matrix.

    """
    return compose_transforms_3d(inverse_transfom_3d(trans_01), trans_02)


def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    r"""Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    .. note:: Works with imperfect (non-orthogonal) rotation matrices as well.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        orthogonal_rotations (bool): If True, will invert `trans_01` assuming `trans_01[:, :3, :3]` are
            orthogonal rotation matrices (more efficient). Default: False

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = gradslam.geometry.geometryutils.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_01.shape)
        )
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )
    trans_10: torch.Tensor = (
        inverse_transformation(trans_01)
        if orthogonal_rotations
        else torch.inverse(trans_01)
    )
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12


def normalize_pixel_coords(
    pixel_coords: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    r"""Normalizes pixel coordinates, so that each dimension (x, y) is now
    in the range [-1, 1].

    x coordinates get mapped from [0, height-1] to [-1, 1]
    y coordinates get mapped from [0, width-1] to [-1, 1]

    Args:
        pixel_coords (torch.Tensor): pixel coordinates of a grid
            (shape: :math:`* \times 2`)
        height (int): height of the image (x-direction)
        width (int): width of the image (y-direction)

    Returns:
        (torch.Tensor): normalized pixel coordinates (same shape
            as `pixel_coords`.)

    """
    if not torch.is_tensor(pixel_coords):
        raise TypeError(
            "Expected pixel_coords to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords)
            )
        )
    if pixel_coords.shape[-1] != 2:
        raise ValueError(
            "Expected last dimension of pixel_coords to be of size 2. Got {} instead.".format(
                pixel_coords.shape[-1]
            )
        )

    assert type(height) == int, "Height must be an integer."
    assert type(width) == int, "Width must be an integer."

    dtype = pixel_coords.dtype
    device = pixel_coords.device

    height = torch.tensor([height]).type(dtype).to(device)
    width = torch.tensor([width]).type(dtype).to(device)

    # Compute normalization factor along each axis
    wh: torch.Tensor = torch.stack([height, width]).type(dtype).to(device)

    norm: torch.Tensor = 2.0 / (wh - 1)

    return norm[:, 0] * pixel_coords - 1


def unnormalize_pixel_coords(
    pixel_coords_norm: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    r"""Unnormalizes pixel coordinates from the range [-1, 1], [-1, 1]
    to [0, `height`-1] and [0, `width`-1] for x and y respectively.

    Args:
        pixel_coords_norm (torch.Tensor): Normalized pixel coordinates
            (shape: :math:`* \times 2`)
        height (int): Height of the image
        width (int): Width of the image

    Returns:
        (torch.Tensor): Unnormalized pixel coordinates

    """
    if not torch.is_tensor(pixel_coords_norm):
        raise TypeError(
            "Expected pixel_coords_norm to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords_norm)
            )
        )
    if pixel_coords_norm.shape[-1] != 2:
        raise ValueError(
            "Expected last dim of pixel_coords_norm to be of shape 2. Got {} instead.".format(
                pixel_coords_norm.shape[-1]
            )
        )

    assert type(height) == int, "Height must be an integer."
    assert type(width) == int, "Width must be an integer."

    dtype = pixel_coords_norm.dtype
    device = pixel_coords_norm.device

    height = torch.tensor([height]).type(dtype).to(device)
    width = torch.tensor([width]).type(dtype).to(device)

    # Compute normalization factor along each axis
    wh: torch.Tensor = torch.stack([height, width]).type(dtype).to(device)

    norm: torch.Tensor = 2.0 / (wh - 1)
    return 1.0 / norm[:, 0] * (pixel_coords_norm + 1)


def create_meshgrid(
    height: int, width: int, normalized_coords: Optional[bool] = True
) -> torch.Tensor:
    r"""Generates a coordinate grid for an image.

    When `normalized_coords` is set to True, the grid is normalized to
    be in the range [-1, 1] (to be consistent with the pytorch function
    `grid_sample`.)

    https://kornia.readthedocs.io/en/latest/utils.html#kornia.utils.create_meshgrid

    Args:
        height (int): Height of the image (number of rows).
        width (int): Width of the image (number of columns).
        normalized_coords (optional, bool): whether or not to
            normalize the coordinates to be in the range [-1, 1].

    Returns:
        (torch.Tensor): grid tensor (shape: :math:`1 \times H \times W \times 2`).

    """

    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coords:
        xs = torch.linspace(-1, 1, height)
        ys = torch.linspace(-1, 1, width)
    else:
        xs = torch.linspace(0, height - 1, height)
        ys = torch.linspace(0, width - 1, width)
    # Generate grid (2 x H x W)
    base_grid: torch.Tensor = torch.stack((torch.meshgrid([xs, ys])))
    return base_grid.permute(1, 2, 0).unsqueeze(0)  # 1 xH x W x 2


def cam2pixel(
    cam_coords_src: torch.Tensor,
    dst_proj_src: torch.Tensor,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    r"""Transforms coordinates from the camera frame to the pixel frame.

    # based on
    # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43

    Args:
        cam_coords_src (torch.Tensor): pixel coordinates (defined in the
            frame of the first camera). (shape: :math:`H \times W \times 3`)
        dst_proj_src (torch.Tensor): projection matrix between the reference
            and the non-reference camera frame. (shape: :math:`4 \times 4`)

    Returns:
        (torch.Tensor): array of [-1, 1] coordinates (shape:
            :math:`H \times W \times 2`)

    """
    assert torch.is_tensor(
        cam_coords_src
    ), "cam_coords_src must be of type torch.Tensor."
    assert cam_coords_src.dim() in (3, 4), "cam_coords_src must have 3 or 4 dimensions."
    assert cam_coords_src.shape[-1] == 3
    assert torch.is_tensor(dst_proj_src), "dst_proj_src must be of type torch.Tensor."
    assert (
        dst_proj_src.dim() == 2
        and dst_proj_src.shape[0] == 4
        and dst_proj_src.shape[0] == 4
    )

    _, h, w, _ = cam_coords_src.shape
    pts: torch.Tensor = transform_pts_3d(cam_coords_src, dst_proj_src)
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))

    return torch.stack([u, v], dim=-1)


def pixel2cam(
    depth: torch.Tensor, intrinsics_inv: torch.Tensor, pixel_coords: torch.Tensor
) -> torch.Tensor:
    r"""Transforms points from the pixel frame to the camera frame.

    Args:
        depth (torch.Tensor): the source depth maps (shape:
            :math:`H \times W`)
        intrinsics_inv (torch.Tensor): the inverse of the intrinsics
            (shape: :math:`4 \times 4`)
        pixel_coords (torch.Tensor): the grid of homogeneous camera
            coordinates (shape: :math:`H \times W \times 3`)

    Returns:
        (torch.Tensor): camera coordinates (shape: :math:`H \times W \times 3`)

    """
    if not torch.is_tensor(depth):
        raise TypeError(
            "Expected depth to be of type torch.Tensor. Got {} instead.".format(
                type(depth)
            )
        )
    if not torch.is_tensor(intrinsics_inv):
        raise TypeError(
            "Expected intrinsics_inv to be of type torch.Tensor. Got {} instead.".format(
                type(intrinsics_inv)
            )
        )
    if not torch.is_tensor(pixel_coords):
        raise TypeError(
            "Expected pixel_coords to be of type torch.Tensor. Got {} instead.".format(
                type(pixel_coords)
            )
        )
    assert (
        intrinsics_inv.shape[0] == 4
        and intrinsics_inv.shape[1] == 4
        and intrinsics_inv.dim() == 2
    )

    cam_coords: torch.Tensor = transform_pts_3d(
        pixel_coords, intrinsics_inv
    )  # .permute(0, 3, 1, 2)

    return cam_coords * depth.permute(0, 2, 3, 1)


def cam2pixel_KF(
    cam_coords_src: torch.Tensor, P: torch.Tensor, eps: Optional[float] = 1e-6
) -> torch.Tensor:
    r"""Projects camera coordinates onto the image.

    Args:
        cam_coords_src (torch.Tensor): camera coordinates (defined in the
            frame of the first camera). (shape: :math:`H \times W \times 3`)
        P (torch.Tensor): projection matrix between the reference and the
            non-reference camera frame. (shape: :math:`4 \times 4`)

    Returns:
        (torch.Tensor): array of [-1, 1] coordinates (shape:
            :math:`H \times W \times 2`)

    """
    assert torch.is_tensor(
        cam_coords_src
    ), "cam_coords_src must be of type torch.Tensor."
    # assert cam_coords_src.dim() > 3, 'cam_coords_src must have > 3 dimensions.'
    assert cam_coords_src.shape[-1] == 3
    assert torch.is_tensor(P), "dst_proj_src must be of type torch.Tensor."
    assert P.dim() >= 2 and P.shape[-1] == 4 and P.shape[-2] == 4

    pts: torch.Tensor = transform_pts_nd_KF(cam_coords_src, P)
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))

    return torch.stack([u, v], dim=-1)


def transform_pointcloud(pointcloud: torch.Tensor, transform: torch.Tensor):
    r"""Applies a rigid-body transformation to a pointcloud.

    Args:
        pointcloud (torch.Tensor): Pointcloud to be transformed
                                   (shape: numpts x 3)
        transform (torch.Tensor): An SE(3) rigid-body transform matrix
                                  (shape: 4 x 4)

    Returns:
        transformed_pointcloud (torch.Tensor): Rotated and translated cloud
                                               (shape: numpts x 3)

    """
    if not torch.is_tensor(pointcloud):
        raise TypeError(
            "pointcloud should be tensor, but was %r instead" % type(pointcloud)
        )

    if not torch.is_tensor(transform):
        raise TypeError(
            "transform should be tensor, but was %r instead" % type(transform)
        )

    if not pointcloud.ndim == 2:
        raise ValueError(
            "pointcloud should have ndim of 2, but had {} instead.".format(
                pointcloud.ndim
            )
        )
    if not pointcloud.shape[1] == 3:
        raise ValueError(
            "pointcloud.shape[1] should be 3 (x, y, z), but was {} instead.".format(
                pointcloud.shape[1]
            )
        )
    if not transform.shape[-2:] == (4, 4):
        raise ValueError(
            "transform should be of shape (4, 4), but was {} instead.".format(
                transform.shape
            )
        )

    # Rotation matrix
    rmat = transform[:3, :3]
    # Translation vector
    tvec = transform[:3, 3]

    # Transpose the pointcloud (to enable broadcast of rotation to each point)
    transposed_pointcloud = torch.transpose(pointcloud, 0, 1)
    # Rotate and translate cloud
    transformed_pointcloud = torch.matmul(rmat, transposed_pointcloud) + tvec.unsqueeze(
        1
    )
    # Transpose the transformed cloud to original dimensions
    transformed_pointcloud = torch.transpose(transformed_pointcloud, 0, 1)

    return transformed_pointcloud


def transform_normals(normals: torch.Tensor, transform: torch.Tensor):
    r"""Applies a rotation to a tensor containing point normals.

    Args:
        normals (torch.Tensor): Normal vectors (shape: numpoints x 3)
    """
    if not torch.is_tensor(normals):
        raise TypeError("normals should be tensor, but was %r instead" % type(normals))

    if not torch.is_tensor(transform):
        raise TypeError(
            "transform should be tensor, but was %r instead" % type(transform)
        )

    if not normals.ndim == 2:
        raise ValueError(
            "normals should have ndim of 2, but had {} instead.".format(normals.ndim)
        )
    if not normals.shape[1] == 3:
        raise ValueError(
            "normals.shape[1] should be 3 (x, y, z), but was {} instead.".format(
                normals.shape[1]
            )
        )
    if not transform.shape[-2:] == (4, 4):
        raise ValueError(
            "transform should be of shape (4, 4), but was {} instead.".format(
                transform.shape
            )
        )

    # Rotation
    R = transform[:3, :3]

    # apply transpose to normals
    transposed_normals = torch.transpose(normals, 0, 1)

    # transpose after transform
    transformed_normals = torch.transpose(torch.matmul(R, transposed_normals), 0, 1)

    return transformed_normals


if __name__ == "__main__":

    # pts = torch.randn(20, 10, 3)
    # homo = homogenize_points(pts)
    # homo[0:3,0:3,3] = torch.zeros(3)
    # # print(homo)
    # unhomo = unhomogenize_points(homo)
    # # print(unhomo)

    # tf = 2 * torch.eye(4)
    # pts_a = transform_pts_3d(unhomo, tf)
    # # print(pts_a)

    # grid = create_meshgrid(480, 640, False)
    # # # print(grid)
    # grid_norm = normalize_pixel_coords(grid, 480, 640)
    # # # print(grid_norm)
    # # grid_unnorm = unnormalize_pixel_coords(grid_norm, 480, 640)
    # # # print(grid_unnorm)

    # from PinholeCamera import PinholeCamera
    # cam = PinholeCamera.from_params(100, 101, 20, 21, 480, 640)
    # pixels = cam2pixel(pts, cam.extrinsics)
    # depth = torch.randn(20, 10, 1)
    # pxl = torch.randn(20, 10, 3)
    # cam_pts = pixel2cam(depth, cam.intrinsics_inverse(), pxl)
    # print(pixels)

    """
    Testing all functions
    """
    h, w = 32, 32
    f, cx, cy = 5, 16, 16
    depth_src = torch.ones(1, 1, h, w)
    img_dst = torch.rand(1, 3, h, w)
    from PinholeCamera import PinholeCamera

    cam = PinholeCamera.from_params(f, f, cx, cy, h, w, 1.0, 2.0, 3.0)
    grid = create_meshgrid(h, w, False)
    grid_homo = homogenize_points(grid)
    px2cm = pixel2cam(depth_src, cam.intrinsics_inverse(), grid_homo)
    print(px2cm.shape, cam.intrinsics.shape)
    cm2px = cam2pixel(px2cm, cam.intrinsics)
    print(cm2px.shape)