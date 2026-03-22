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
    torch::Tensor depths,
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

void launch_count_tiles_kernel(
    const torch::Tensor points_image,
    const torch::Tensor radii,
    const torch::Tensor mask,
    const uint32_t width,
    const uint32_t height,
    const uint32_t tile_size,
    torch::Tensor num_tiles
);

void launch_compute_tile_intersection_kernel(
    const torch::Tensor points_image,
    const torch::Tensor radii,
    const torch::Tensor depth,
    const torch::Tensor cum_num_tiles,
    const torch::Tensor mask,
    const uint32_t width,
    const uint32_t height,
    const uint32_t tile_size,
    torch::Tensor tile_indices_encoded_depth,
    torch::Tensor gaussian_indices
);

void radix_sort_double_buffer(
    const int64_t num_items,
    const torch::Tensor keys_in,
    const torch::Tensor values_in,
    torch::Tensor keys_out,
    torch::Tensor values_out
);

void launch_shift_out_depth_kernel(
    const int64_t total_tiles,
    const torch::Tensor tile_ids_encoded_depth,
    torch::Tensor tile_ids_sorted
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
    torch::Tensor depths = torch::zeros({N}, torch::kFloat32);  // (N,)
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
        depths,
        cov_inv_image,
        radii,
        mask
    );

    return std::make_tuple(points_image, depths, cov_inv_image, radii, mask);
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

std::tuple<torch::Tensor, torch::Tensor> compute_tile_intersection(
    const torch::Tensor points_image,  // (N, 2)
    const torch::Tensor radii,  // (N, 2)
    const torch::Tensor depths,  // (N,)
    const torch::Tensor mask,  // (N,)
    const uint32_t width,
    const uint32_t height,
    const uint32_t tile_size = 16
) {
    int N = points_image.size(0);
    torch::Tensor num_tiles = torch::zeros({N}, torch::kInt32);  // (N,)
    torch::Tensor cum_num_tiles = torch::zeros({N}, torch::kInt64);  // (N,)

    // compute number of tiles intersected by each projected gaussian
    launch_count_tiles_kernel(
        points_image,
        radii,
        mask,
        width,
        height,
        tile_size,
        num_tiles
    );

    // compute cumulative sum to get starting index of gaussian in flattened tile list
    cum_num_tiles = torch::cumsum(num_tiles, 0);
    
    // compute tile ids encoded with depth (for sorting) and gaussian ids
    int64_t total_tiles = cum_num_tiles[-1].item<int64_t>();
    torch::Tensor tile_ids_encoded_depth = torch::zeros({total_tiles}, torch::kInt64);
    torch::Tensor gaussian_ids = torch::zeros({total_tiles}, torch::kInt32);
    launch_compute_tile_intersection_kernel(
        points_image,
        radii,
        depths,
        cum_num_tiles,
        mask,
        width,
        height,
        tile_size,
        tile_ids_encoded_depth,
        gaussian_ids
    );

    // sort by tile id (upper 32 bits) and depth (lower 32 bits)
    torch::Tensor tile_ids_encoded_depth_sorted = torch::zeros({total_tiles}, torch::kInt64);
    torch::Tensor gaussian_ids_sorted = torch::zeros({total_tiles}, torch::kInt32);
    radix_sort_double_buffer(
        total_tiles,
        tile_ids_encoded_depth,
        gaussian_ids,
        tile_ids_encoded_depth_sorted,
        gaussian_ids_sorted
    );

    // shift out depth
    torch::Tensor tile_ids_sorted = torch::zeros({total_tiles}, torch::kInt32);
    launch_shift_out_depth_kernel(
        total_tiles,
        tile_ids_encoded_depth_sorted,
        tile_ids_sorted
    );

    return std::make_tuple(tile_ids_sorted, gaussian_ids_sorted);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_points",
          &project_points,
          "Project world points to image plane");
    m.def("evaluate_spherical_harmonics",
          &evaluate_spherical_harmonics,
          "Evaluate spherical harmonics to get colors in given viewing direction");
    m.def("compute_tile_intersection",
          &compute_tile_intersection,
          "Compute number of tiles intersected by each projected gaussian");
}
