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
    const int32_t width,
    const int32_t height,
    torch::Tensor points_image,
    torch::Tensor depths,
    torch::Tensor cov_image,
    torch::Tensor cov_inv_image,
    torch::Tensor radii,
    torch::Tensor mask
);

void launch_project_points_backward_kernel(
    const torch::Tensor grad_points_image,
    const torch::Tensor grad_cov_inv_image,
    const torch::Tensor points_world,
    const torch::Tensor scales,
    const torch::Tensor quaternions,
    const torch::Tensor opacities,
    const torch::Tensor world_to_camera,
    const torch::Tensor intrinsic,
    const torch::Tensor cov_image,
    const torch::Tensor mask,
    torch::Tensor grad_points_world,
    torch::Tensor grad_scales,
    torch::Tensor grad_quaternions
);

void launch_evaluate_spherical_harmonics_kernel(
    const torch::Tensor world_to_camera,
    const torch::Tensor points_world,
    const torch::Tensor sh_coeffs_dc,
    const torch::Tensor sh_coeffs_rest,
    const torch::Tensor mask,
    torch::Tensor colors
);

void launch_evaluate_spherical_harmonics_backward_kernel(
    const torch::Tensor grad_colors,
    const torch::Tensor world_to_camera,
    const torch::Tensor points_world,
    const torch::Tensor sh_coeffs_dc,
    const torch::Tensor sh_coeffs_rest,
    const torch::Tensor mask,
    torch::Tensor grad_points_world,
    torch::Tensor grad_sh_coeffs_dc,
    torch::Tensor grad_sh_coeffs_rest
);

void launch_count_tiles_kernel(
    const torch::Tensor points_image,
    const torch::Tensor radii,
    const torch::Tensor mask,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    torch::Tensor num_tiles
);

void launch_compute_tile_intersection_kernel(
    const torch::Tensor points_image,
    const torch::Tensor radii,
    const torch::Tensor depth,
    const torch::Tensor cum_num_tiles,
    const torch::Tensor mask,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
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

void launch_compute_indexing_offset_kernel(
    const torch::Tensor tile_ids_encoded_depth,
    torch::Tensor indexing_offset
);

void launch_rasterize_kernel(
    const torch::Tensor indexing_offset,
    const torch::Tensor gaussian_ids_sorted,
    const torch::Tensor points_image,
    const torch::Tensor cov_inv_image,
    const torch::Tensor opacities,
    const torch::Tensor colors,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    const float alpha_threshold,
    const float transmittance_threshold,
    const float chi_squared_threshold,
    torch::Tensor rendered_image
);

void launch_rasterize_backward_kernel(
    const torch::Tensor grad_rendered_image,
    const torch::Tensor indexing_offset,
    const torch::Tensor gaussian_ids_sorted,
    const torch::Tensor points_image,
    const torch::Tensor cov_inv_image,
    const torch::Tensor opacities,
    const torch::Tensor colors,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    const float alpha_threshold,
    const float transmittance_threshold,
    const float chi_squared_threshold,
    torch::Tensor grad_points_image,
    torch::Tensor grad_cov_inv_image,
    torch::Tensor grad_opacities,
    torch::Tensor grad_colors
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    const int32_t width,
    const int32_t height
) {
    int N = points_world.size(0);
    
    // use the options of input for output tensors, make sure device is the same
    auto float_options = points_world.options().dtype(torch::kFloat32);
    auto bool_options = points_world.options().dtype(torch::kBool);
    torch::Tensor points_image = torch::zeros({N, 2}, float_options);  // (N, 2)
    torch::Tensor depths = torch::zeros({N}, float_options);  // (N,)
    torch::Tensor cov_image = torch::zeros({N, 3}, float_options);  // (N, 3)
    torch::Tensor cov_inv_image = torch::zeros({N, 3}, float_options);  // (N, 3)
    torch::Tensor radii = torch::zeros({N, 2}, float_options);  // (N, 2)
    torch::Tensor mask = torch::ones({N}, bool_options);  // (N,)

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
        cov_image,
        cov_inv_image,
        radii,
        mask
    );

    return std::make_tuple(points_image, depths, cov_image, cov_inv_image, radii, mask);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project_points_backward(
    const torch::Tensor grad_points_image,  // (N, 2)
    const torch::Tensor grad_cov_inv_image,  // (N, 3)
    const torch::Tensor points_world,  // (N, 3)
    const torch::Tensor scales,  // (N, 3)
    const torch::Tensor quaternions,  // (N, 4)
    const torch::Tensor opacities,  // (N,)
    const torch::Tensor world_to_camera,  // (4, 4)
    const torch::Tensor intrinsic,  // (3, 3)
    const torch::Tensor cov_image,  // (N, 3)
    const torch::Tensor mask  // (N,)
) {
    int N = points_world.size(0);
    auto float_options = points_world.options().dtype(torch::kFloat32);
    torch::Tensor grad_points_world = torch::zeros({N, 3}, float_options);
    torch::Tensor grad_scales = torch::zeros({N, 3}, float_options);
    torch::Tensor grad_quaternions = torch::zeros({N, 4}, float_options);

    launch_project_points_backward_kernel(
        grad_points_image,
        grad_cov_inv_image,
        points_world,
        scales,
        quaternions,
        opacities,
        world_to_camera,
        intrinsic,
        cov_image,
        mask,
        grad_points_world,
        grad_scales,
        grad_quaternions
    );

    return std::make_tuple(grad_points_world, grad_scales, grad_quaternions);
}

torch::Tensor evaluate_spherical_harmonics(
    const torch::Tensor camera_pos,  // (3,)
    const torch::Tensor points_world,  // (N, 3)
    const torch::Tensor sh_coeffs_dc,  // (N, 3)
    const torch::Tensor sh_coeffs_rest,  // (N, 15, 3)
    const torch::Tensor mask  // (N,)
) {
    int N = points_world.size(0);
    auto float_options = points_world.options().dtype(torch::kFloat32);
    torch::Tensor colors = torch::zeros({N, 3}, float_options);

    launch_evaluate_spherical_harmonics_kernel(
        camera_pos,
        points_world,
        sh_coeffs_dc,
        sh_coeffs_rest,
        mask,
        colors
    );

    return colors;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> evaluate_spherical_harmonics_backward(
    const torch::Tensor grad_colors,  // (N, 3)
    const torch::Tensor camera_pos,  // (3,)
    const torch::Tensor points_world,  // (N, 3)
    const torch::Tensor sh_coeffs_dc,  // (N, 3)
    const torch::Tensor sh_coeffs_rest,  // (N, 15, 3)
    const torch::Tensor mask  // (N,)
) {
    int N = points_world.size(0);
    auto float_options = grad_colors.options().dtype(torch::kFloat32);
    torch::Tensor grad_points_world = torch::zeros({N, 3}, float_options);
    torch::Tensor grad_sh_coeffs_dc = torch::zeros({N, 3}, float_options);
    torch::Tensor grad_sh_coeffs_rest = torch::zeros({N, 15, 3}, float_options);

    launch_evaluate_spherical_harmonics_backward_kernel(
        grad_colors,
        camera_pos,
        points_world,
        sh_coeffs_dc,
        sh_coeffs_rest,
        mask,
        grad_points_world,
        grad_sh_coeffs_dc,
        grad_sh_coeffs_rest
    );

    return std::make_tuple(grad_points_world, grad_sh_coeffs_dc, grad_sh_coeffs_rest);
}

std::tuple<torch::Tensor, torch::Tensor> compute_tile_intersection(
    const torch::Tensor points_image,  // (N, 2)
    const torch::Tensor radii,  // (N, 2)
    const torch::Tensor depths,  // (N,)
    const torch::Tensor mask,  // (N,)
    const int32_t width,
    const int32_t height,
    const int32_t tile_size = 16
) {
    int N = points_image.size(0);
    auto int_options = points_image.options().dtype(torch::kInt32);
    auto int64_options = points_image.options().dtype(torch::kInt64);
    torch::Tensor num_tiles = torch::zeros({N}, int_options);  // (N,)
    torch::Tensor cum_num_tiles = torch::zeros({N}, int64_options);  // (N,)

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
    cum_num_tiles = torch::cumsum(num_tiles, 0, torch::kInt64);  // cumsum returns int64 by default

    // compute tile ids encoded with depth (for sorting) and gaussian ids
    int64_t total_tiles = cum_num_tiles[-1].item<int64_t>();
    torch::Tensor tile_ids_encoded_depth = torch::zeros({total_tiles}, int64_options);
    torch::Tensor gaussian_ids = torch::zeros({total_tiles}, int_options);
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
    torch::Tensor tile_ids_encoded_depth_sorted = torch::zeros({total_tiles}, int64_options);
    torch::Tensor gaussian_ids_sorted = torch::zeros({total_tiles}, int_options);
    radix_sort_double_buffer(
        total_tiles,
        tile_ids_encoded_depth,
        gaussian_ids,
        tile_ids_encoded_depth_sorted,
        gaussian_ids_sorted
    );

    // compute tile indexing offset
    int32_t tile_width = (width + tile_size - 1) / tile_size;
    int32_t tile_height = (height + tile_size - 1) / tile_size;
    torch::Tensor indexing_offset = torch::zeros({tile_width * tile_height}, int_options);
    launch_compute_indexing_offset_kernel(
        tile_ids_encoded_depth_sorted,
        indexing_offset
    );

    return std::make_tuple(indexing_offset, gaussian_ids_sorted);
}

torch::Tensor rasterize(
    const torch::Tensor indexing_offset,  // (unique_tiles,)
    const torch::Tensor gaussian_ids_sorted,  // (total_tiles,)
    const torch::Tensor points_image,  // (N, 2)
    const torch::Tensor cov_inv_image,  // (N, 3)
    const torch::Tensor opacities,  // (N,)
    const torch::Tensor colors,  // (N, 3)
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    const float alpha_threshold,
    const float transmittance_threshold,
    const float chi_squared_threshold
) {
    auto float_options = points_image.options().dtype(torch::kFloat32);
    torch::Tensor rendered_image = torch::zeros({height, width, 3}, float_options);

    launch_rasterize_kernel(
        indexing_offset,
        gaussian_ids_sorted,
        points_image,
        cov_inv_image,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
        rendered_image
    );

    return rendered_image;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward(
    const torch::Tensor grad_rendered_image,  // (H, W, 3)
    const torch::Tensor indexing_offset,  // (unique_tiles,)
    const torch::Tensor gaussian_ids_sorted,  // (total_tiles,)
    const torch::Tensor points_image,  // (N, 2)
    const torch::Tensor cov_inv_image,  // (N, 3)
    const torch::Tensor opacities,  // (N,)
    const torch::Tensor colors,  // (N, 3)
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    const float alpha_threshold,
    const float transmittance_threshold,
    const float chi_squared_threshold
) {
    int N = points_image.size(0);
    auto float_options = points_image.options().dtype(torch::kFloat32);
    torch::Tensor grad_points_image = torch::zeros({N, 2}, float_options);
    torch::Tensor grad_cov_inv_image = torch::zeros({N, 3}, float_options);
    torch::Tensor grad_opacities = torch::zeros({N}, float_options);
    torch::Tensor grad_colors = torch::zeros({N, 3}, float_options);

    launch_rasterize_backward_kernel(
        grad_rendered_image,
        indexing_offset,
        gaussian_ids_sorted,
        points_image,
        cov_inv_image,
        opacities,
        colors,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
        grad_points_image,
        grad_cov_inv_image,
        grad_opacities,
        grad_colors
    );

    return std::make_tuple(grad_points_image, grad_cov_inv_image, grad_opacities, grad_colors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_points",
          &project_points,
          "Project world points to image plane");
    m.def("project_points_backward",
          &project_points_backward,
          "Backward pass for point projection");
    m.def("evaluate_spherical_harmonics",
          &evaluate_spherical_harmonics,
          "Evaluate spherical harmonics to get colors in given viewing direction");
    m.def("evaluate_spherical_harmonics_backward",
          &evaluate_spherical_harmonics_backward,
          "Backward pass for spherical harmonics evaluation");
    m.def("compute_tile_intersection",
          &compute_tile_intersection,
          "Compute number of tiles intersected by each projected gaussian");
    m.def("rasterize",
          &rasterize,
          "Rasterize gaussians to get rendered image");
    m.def("rasterize_backward",
          &rasterize_backward,
          "Backward pass for rasterization");
}
