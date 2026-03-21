import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


SH_C0 = 0.28209479177387814
SH_C1_x = 0.4886025119029199
SH_C1_y = 0.4886025119029199
SH_C1_z = 0.4886025119029199
SH_C2_xy = 1.0925484305920792
SH_C2_xz = 1.0925484305920792
SH_C2_yz = 1.0925484305920792
SH_C2_zz = 0.31539156525252005
SH_C2_xx_yy = 0.5462742152960396
SH_C3_yxx_yyy = 0.5900435899266435
SH_C3_xyz = 2.890611442640554
SH_C3_yzz_yxx_yyy = 0.4570457994644658
SH_C3_zzz_zxx_zyy = 0.3731763325901154
SH_C3_xzz_xxx_xyy = 0.4570457994644658
SH_C3_zxx_zyy = 1.445305721320277
SH_C3_xxx_xyy = 0.5900435899266435


def evaluate_spherical_harmonics(sh_coeffs_dc, sh_coeffs_rest, view_dirs):
    # sh_coeffs_dc shape: (N, 3) for degree 0
    # sh_coeffs_rest shape: (N, 15, 3) for degree 1,2,3
    # view_dirs shape: (N, 3)

    x, y, z = view_dirs.unbind(dim=1)  # (N,)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z

    sh_basis_rest = torch.stack(
        [
            -SH_C1_y * y,  # l=1, m=-1
            SH_C1_z * z,  # l=1, m=0
            -SH_C1_x * x,  # l=1, m=1
            SH_C2_xy * xy,  # l=2, m=-2
            SH_C2_yz * yz,  # l=2, m=-1
            SH_C2_zz * (3 * zz - 1),  # l=2, m=0
            SH_C2_xz * xz,  # l=2, m=1
            SH_C2_xx_yy * (xx - yy),  # l=2, m=2
            SH_C3_yxx_yyy * y * (xx - yy),  # l=3, m=-3
            SH_C3_xyz * (x * y * z),  # l=3, m=-2
            SH_C3_yzz_yxx_yyy * y * (4 * zz - xx - yy),  # l=3, m=-1
            SH_C3_zzz_zxx_zyy * z * (2 * zz - 3 * xx - 3 * yy),  # l=3, m=0
            SH_C3_xzz_xxx_xyy * x * (4 * zz - xx - yy),  # l=3, m=1
            SH_C3_zxx_zyy * z * (xx - yy),  # l=3, m=2
            SH_C3_xxx_xyy * x * (xx - 3 * yy),  # l=3, m=3
        ],
        dim=1,
    )  # (N, 15)

    sh_dc = sh_coeffs_dc * SH_C0  # (N, 3)
    sh_rest = (sh_coeffs_rest * sh_basis_rest.unsqueeze(-1)).sum(dim=1)  # (N, 3)

    # the test data has SH coefficients in logit space
    return torch.sigmoid(sh_dc + sh_rest)  # (N, 3)

    # in the cuda implementation, the SH coefficients are not in logit space
    # return sh_dc + sh_rest  # (N, 3)


def quaternion_to_rotation_matrix(quaternions):
    # quaternions shape: (N, 4) in (x, y, z, w) format
    x, y, z, w = quaternions.unbind(dim=1)

    R = torch.zeros((quaternions.shape[0], 3, 3), device=quaternions.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R


def downsample_point_cloud(
    points: np.ndarray,
    rgbs: np.ndarray,
    max_points: int,
    seed: int = 42,
):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, rgbs

    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep_idx], rgbs[keep_idx]


def initialize(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    init_scale: float = 1.0,
    init_opacity: float = 0.1,
    sh_degree: int = 3,
    device: str = "cuda",
):
    # points shape: (N, 3)
    # rgbs shape: (N, 3) in [0, 1] range
    N = points.shape[0]

    # initalize gaussians size to be the average distance of the 3 nearest neighbors
    pts = points.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=4).fit(pts)  # 3 closest neighbors + the point itself
    distances, _ = knn.kneighbors(pts)  # distances shape: (N, 4)
    dist = torch.from_numpy(distances[:, 1:]).float()  # 0th column is distance to itself
    avg_dist = torch.sqrt(torch.square(dist).mean(axis=1))  # avg_dist shape: (N,)
    # in log space, after exp the values will be positive
    scales = torch.log(avg_dist * init_scale).unsqueeze(-1).repeat(1, 3)

    quaternions = torch.rand(N, 4)
    # in logit space, after sigmoid the values will be constrained in (0, 1)
    opacities = torch.logit(torch.full((N,), init_opacity))

    C0 = 0.28209479177387814  # DC component of SH basis
    # normalize RGB colors to [-0.5, 0.5] and scale by C0 and initialize sh_coeffs_dc
    sh_coeffs_dc = (rgbs - 0.5) / C0  # (N, 3) for the DC component of SH
    # dc and rest of the SH coefficients are separated to apply for different learning rates
    sh_coeffs_rest = torch.zeros(
        (N, (sh_degree + 1) ** 2 - 1, 3)
    )  # (N, 15, 3) for degree 3 excluding DC

    learnable_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quaternions": torch.nn.Parameter(quaternions),
            "opacities": torch.nn.Parameter(opacities),
            "sh_coeffs_dc": torch.nn.Parameter(sh_coeffs_dc),
            "sh_coeffs_rest": torch.nn.Parameter(sh_coeffs_rest),
        }
    ).to(device)

    return learnable_params
