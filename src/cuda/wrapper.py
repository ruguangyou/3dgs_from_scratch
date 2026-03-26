import cuda_rasterizer
import torch


class ProjectPointsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
    ):
        points_img, depths, cov_img, cov_inv_img, radii, mask = cuda_rasterizer.project_points(
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
        ctx.save_for_backward(
            means, scales, quaternions, opacities, world_to_camera, intrinsic, cov_img, mask
        )
        return points_img, depths, cov_inv_img, radii, mask

    @staticmethod
    def backward(ctx, grad_points_img, grad_depths, grad_cov_inv_img, grad_radii, grad_mask):
        means, scales, quaternions, opacities, world_to_camera, intrinsic, cov_img, mask = (
            ctx.saved_tensors
        )
        grad_means, grad_scales, grad_quaternions = cuda_rasterizer.project_points_backward(
            grad_points_img,
            grad_cov_inv_img,
            means,
            scales,
            quaternions,
            opacities,
            world_to_camera,
            intrinsic,
            cov_img,
            mask,
        )
        return (
            grad_means,
            grad_scales,
            grad_quaternions,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SphericalHarmonicsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, camera_pos, means, sh_coeffs_dc, sh_coeffs_rest, mask):
        colors = cuda_rasterizer.evaluate_spherical_harmonics(
            camera_pos, means, sh_coeffs_dc, sh_coeffs_rest, mask
        )
        ctx.save_for_backward(camera_pos, means, sh_coeffs_dc, sh_coeffs_rest, colors, mask)
        return colors

    @staticmethod
    def backward(ctx, grad_colors):
        camera_pos, means, sh_coeffs_dc, sh_coeffs_rest, colors, mask = ctx.saved_tensors
        grad_means, grad_sh_coeffs_dc, grad_sh_coeffs_rest = (
            cuda_rasterizer.evaluate_spherical_harmonics_backward(
                grad_colors,
                camera_pos,
                means,
                sh_coeffs_dc,
                sh_coeffs_rest,
                colors,
                mask,
            )
        )
        return None, grad_means, grad_sh_coeffs_dc, grad_sh_coeffs_rest, None


class RasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indexing_offset,
        gaussian_ids_sorted,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    ):
        rendered_image = cuda_rasterizer.rasterize(
            indexing_offset,
            gaussian_ids_sorted,
            points_img,
            cov_inv_img,
            opacities,
            colors,
            width,
            height,
            tile_size,
            alpha_threshold,
            transmittance_threshold,
            chi_squared_threshold,
        )
        ctx.save_for_backward(
            indexing_offset,
            gaussian_ids_sorted,
            points_img,
            cov_inv_img,
            opacities,
            colors,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.alpha_threshold = alpha_threshold
        ctx.transmittance_threshold = transmittance_threshold
        ctx.chi_squared_threshold = chi_squared_threshold
        return rendered_image

    @staticmethod
    def backward(ctx, grad_rendered_image):
        (
            indexing_offset,
            gaussian_ids_sorted,
            points_img,
            cov_inv_img,
            opacities,
            colors,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        alpha_threshold = ctx.alpha_threshold
        transmittance_threshold = ctx.transmittance_threshold
        chi_squared_threshold = ctx.chi_squared_threshold

        (
            grad_points_img,
            grad_cov_inv_img,
            grad_opacities,
            grad_colors,
        ) = cuda_rasterizer.rasterize_backward(
            grad_rendered_image,
            indexing_offset,
            gaussian_ids_sorted,
            points_img,
            cov_inv_img,
            opacities,
            colors,
            width,
            height,
            tile_size,
            alpha_threshold,
            transmittance_threshold,
            chi_squared_threshold,
        )
        return (
            None,
            None,
            grad_points_img,
            grad_cov_inv_img,
            grad_opacities,
            grad_colors,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    points_img, depths, cov_inv_img, radii, mask = ProjectPointsFunction.apply(
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
    colors = SphericalHarmonicsFunction.apply(camera_pos, means, sh_coeffs_dc, sh_coeffs_rest, mask)

    tile_size = 16
    indexing_offset, gaussian_ids_sorted = cuda_rasterizer.compute_tile_intersection(
        points_img, radii, depths, mask, width, height, tile_size
    )

    rendered_image = RasterizeFunction.apply(
        indexing_offset,
        gaussian_ids_sorted,
        points_img,
        cov_inv_img,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
    )

    return rendered_image * 255  # scale to [0, 255]
