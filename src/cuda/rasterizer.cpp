#include <torch/extension.h>
#include <tuple>

void launch_project_points_kernel(
    const torch::Tensor points_world,
    const torch::Tensor scales,
    const torch::Tensor quaternions,
    const torch::Tensor opacities,
    const torch::Tensor world_to_camera,
    const torch::Tensor intrinsic,
    const float near_plane,
    const float far_plane,
    const float min_opacity,
    const float min_radius,
    const float max_radius,
    const uint32_t width,
    const uint32_t height,
    torch::Tensor points_image,
    torch::Tensor depth,
    torch::Tensor cov_inv_image,
    torch::Tensor radii,
    torch::Tensor mask
);

void launch_evaluate_spherical_harmonics_kernel(
    const torch::Tensor world_to_camera,
    const torch::Tensor world_points,
    const torch::Tensor sh_coeffs_dc,
    const torch::Tensor sh_coeffs_rest,
    const torch::Tensor mask,
    torch::Tensor colors
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
project_points(
    const torch::Tensor points_world,  // (N, 3)
    const torch::Tensor scales,  // (N, 3)
    const torch::Tensor quaternions,  // (N, 4)
    const torch::Tensor opacities,  // (N,)
    const torch::Tensor world_to_camera,  // (4, 4)
    const torch::Tensor intrinsic,  // (3, 3)
    const float near_plane,
    const float far_plane,
    const float min_opacity,
    const float min_radius,
    const float max_radius,
    const uint32_t width,
    const uint32_t height
) {
    uint32_t N = points_world.size(0);
    torch::Tensor points_image = torch::zeros({N, 2}, torch::kFloat32);  // (N, 2)
    torch::Tensor depth = torch::zeros({N}, torch::kFloat32);  // (N,)
    torch::Tensor cov_inv_image = torch::zeros({N, 3}, torch::kFloat32);  // (N, 3)
    torch::Tensor radii = torch::zeros({N, 2}, torch::kFloat32);  // (N, 2)
    torch::Tensor mask = torch::ones({N}, torch::kBool);  // (N,)

    launch_project_points_kernel(
        points_world,
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
        points_image,
        depth,
        cov_inv_image,
        radii,
        mask
    );

    return std::make_tuple(points_image, depth, cov_inv_image, radii, mask);
}

torch::Tensor evaluate_spherical_harmonics(
    const torch::Tensor camera_pos,  // (3,)
    const torch::Tensor world_points,  // (N, 3)
    const torch::Tensor sh_coeffs_dc,  // (N, 3)
    const torch::Tensor sh_coeffs_rest,  // (N, 15, 3)
    const torch::Tensor mask  // (N,)
) {
    int N = world_points.size(0);
    torch::Tensor colors = torch::zeros({N, 3}, torch::kFloat32);  // (N, 3)

    launch_evaluate_spherical_harmonics_kernel(
        camera_pos,
        world_points,
        sh_coeffs_dc,
        sh_coeffs_rest,
        mask,
        colors
    );

    return colors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_points",
          &project_points,
          "Project world points to image plane");
    m.def("evaluate_spherical_harmonics",
          &evaluate_spherical_harmonics,
          "Compute spherical harmonics lighting");
}
