import cuda_rasterizer


def render(
    world_to_camera,
    intrinsic,
    width,
    height,
    means,
    scales,
    quaternions,
    opacities,
    sh_coeffs_dc,
    sh_coeffs_rest,
    near_plane=0.01,
    far_plane=100.0,
    min_opacity=1 / 255,
    min_radius=0.5,
    max_radius=128.0,
    alpha_threshold=1e-4,
    transmittance_threshold=1e-4,
    chi_squared_threshold=9.21,  # 99% confidence interval for 2 DOF
):
    points_img, depth, cov_inv_img, radii, mask = cuda_rasterizer.project_points(
        means,
        scales,
        quaternions,
        opacities,
        world_to_camera,
        intrinsic,
        near_plane,
        far_plane,
        min_opacity,
        min_radius,
        max_radius,
        width,
        height,
    )

    camera_pos = -world_to_camera[:3, :3].t() @ world_to_camera[:3, 3]
    colors = cuda_rasterizer.evaluate_spherical_harmonics(
        camera_pos, means, sh_coeffs_dc, sh_coeffs_rest, mask
    )
