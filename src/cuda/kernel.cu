#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr float SH_C0 = 0.28209479177387814;
constexpr float SH_C1_x = 0.4886025119029199;
constexpr float SH_C1_y = 0.4886025119029199;
constexpr float SH_C1_z = 0.4886025119029199;
constexpr float SH_C2_xy = 1.0925484305920792;
constexpr float SH_C2_xz = 1.0925484305920792;
constexpr float SH_C2_yz = 1.0925484305920792;
constexpr float SH_C2_zz = 0.31539156525252005;
constexpr float SH_C2_xx_yy = 0.5462742152960396;
constexpr float SH_C3_yxx_yyy = 0.5900435899266435;
constexpr float SH_C3_xyz = 2.890611442640554;
constexpr float SH_C3_yzz_yxx_yyy = 0.4570457994644658;
constexpr float SH_C3_zzz_zxx_zyy = 0.3731763325901154;
constexpr float SH_C3_xzz_xxx_xyy = 0.4570457994644658;
constexpr float SH_C3_zxx_zyy = 1.445305721320277;
constexpr float SH_C3_xxx_xyy = 0.5900435899266435;

__device__ void transform_w2c(
    const float *world_to_camera,
    const float *point_world,
    float3 &point_camera
) {
    point_camera.x = world_to_camera[0] * point_world[0] +
                     world_to_camera[1] * point_world[1] +
                     world_to_camera[2] * point_world[2] +
                     world_to_camera[3];
    point_camera.y = world_to_camera[4] * point_world[0] +
                     world_to_camera[5] * point_world[1] +
                     world_to_camera[6] * point_world[2] +
                     world_to_camera[7];
    point_camera.z = world_to_camera[8] * point_world[0] +
                     world_to_camera[9] * point_world[1] +
                     world_to_camera[10] * point_world[2] +
                     world_to_camera[11];
}

__device__ void transform_cov_w2c2i(
    const float *W,  // world_to_camera (4*4,)
    const float *K,  // intrinsic (3*3,)
    const float *scale,
    const float *quaternion,
    const float3 &point_camera,
    float3 &cov_image
) {
    // quaternion to rotation matrix
    float qw = quaternion[0];
    float qx = quaternion[1];
    float qy = quaternion[2];
    float qz = quaternion[3];
    float R[9] = {
        1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy
    };

    // scale^2
    float sx2 = scale[0] * scale[0];
    float sy2 = scale[1] * scale[1];
    float sz2 = scale[2] * scale[2];

    // cov_world = R * S * S * R^T
    float Cw[9] = {
        R[0] * sx2 * R[0] + R[1] * sy2 * R[1] + R[2] * sz2 * R[2],
        R[0] * sx2 * R[3] + R[1] * sy2 * R[4] + R[2] * sz2 * R[5],
        R[0] * sx2 * R[6] + R[1] * sy2 * R[7] + R[2] * sz2 * R[8],
        R[3] * sx2 * R[0] + R[4] * sy2 * R[1] + R[5] * sz2 * R[2],
        R[3] * sx2 * R[3] + R[4] * sy2 * R[4] + R[5] * sz2 * R[5],
        R[3] * sx2 * R[6] + R[4] * sy2 * R[7] + R[5] * sz2 * R[8],
        R[6] * sx2 * R[0] + R[7] * sy2 * R[1] + R[8] * sz2 * R[2],
        R[6] * sx2 * R[3] + R[7] * sy2 * R[4] + R[8] * sz2 * R[5],
        R[6] * sx2 * R[6] + R[7] * sy2 * R[7] + R[8] * sz2 * R[8]
    };

    // cov_camera = W * cov_world * W^T
    float WCw[9] = {  // W * cov_world
        W[0] * Cw[0] + W[1] * Cw[3] + W[2] * Cw[6],
        W[0] * Cw[1] + W[1] * Cw[4] + W[2] * Cw[7],
        W[0] * Cw[2] + W[1] * Cw[5] + W[2] * Cw[8],
        W[4] * Cw[0] + W[5] * Cw[3] + W[6] * Cw[6],
        W[4] * Cw[1] + W[5] * Cw[4] + W[6] * Cw[7],
        W[4] * Cw[2] + W[5] * Cw[5] + W[6] * Cw[8],
        W[8] * Cw[0] + W[9] * Cw[3] + W[10] * Cw[6],
        W[8] * Cw[1] + W[9] * Cw[4] + W[10] * Cw[7],
        W[8] * Cw[2] + W[9] * Cw[5] + W[10] * Cw[8]
    };
    float Cc[9] = {  // W * cov_world * W^T
        WCw[0] * W[0] + WCw[1] * W[1] + WCw[2] * W[2],
        WCw[0] * W[4] + WCw[1] * W[5] + WCw[2] * W[6],
        WCw[0] * W[8] + WCw[1] * W[9] + WCw[2] * W[10],
        WCw[3] * W[0] + WCw[4] * W[1] + WCw[5] * W[2],
        WCw[3] * W[4] + WCw[4] * W[5] + WCw[5] * W[6],
        WCw[3] * W[8] + WCw[4] * W[9] + WCw[5] * W[10],
        WCw[6] * W[0] + WCw[7] * W[1] + WCw[8] * W[2],
        WCw[6] * W[4] + WCw[7] * W[5] + WCw[8] * W[6],
        WCw[6] * W[8] + WCw[7] * W[9] + WCw[8] * W[10]
    };

    // Jacobian of perspective projection
    float x = point_camera.x;
    float y = point_camera.y;
    float z = point_camera.z;
    float inv_z = 1.0f / z;
    float inv_z2 = inv_z * inv_z;
    float J[6] = {
        K[0] * inv_z, 0, -K[0] * x * inv_z2,
        0, K[4] * inv_z, -K[4] * y * inv_z2
    };
    
    // cov_image = J * cov_camera * J^T
    float JCc[6] = {  // J * cov_camera
        J[0] * Cc[0] + J[2] * Cc[6],
        J[0] * Cc[1] + J[2] * Cc[7],
        J[0] * Cc[2] + J[2] * Cc[8],
        J[4] * Cc[3] + J[5] * Cc[6],
        J[4] * Cc[4] + J[5] * Cc[7],
        J[4] * Cc[5] + J[5] * Cc[8]
    };
    float Ci[4] = {  // J * cov_camera * J^T
        JCc[0] * J[0] + JCc[2] * J[2],
        JCc[1] * J[3] + JCc[2] * J[5],
        JCc[3] * J[0] + JCc[5] * J[2],
        JCc[4] * J[3] + JCc[5] * J[5]
    };

    // ensure symmetry
    cov_image.x = Ci[0];
    cov_image.y = 0.5f * (Ci[1] + Ci[2]);
    cov_image.z = Ci[3];
}

__global__ void project_points_kernel(
    const uint32_t N,
    const float *points_world,  // (N*3,) row-major
    const float *scales,  // (N*3,)
    const float *quaternions,  // (N*4,)
    const float *opacities,  // (N,)
    const float *world_to_camera,  // (4*4,)
    const float *intrinsic,  // (3*3,)
    const float near_plane,
    const float far_plane,
    const float min_opacity,
    const float min_radius,
    const float max_radius,
    const uint32_t width,
    const uint32_t height,
    float *points_image,  // (N*2,)
    float *depth,  // (N,)
    float *cov_inv_image,  // (N*3,)
    float *radii,  // (N*2,)
    bool *mask  // (N,)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    // mask out points with low opacity
    if (opacities[idx] < min_opacity) {
        mask[idx] = false;
        return;
    }

    // transform point from world to camera space
    float3 point_camera;
    transform_w2c(world_to_camera, &points_world[idx*3], point_camera);
    if (point_camera.z < near_plane || point_camera.z > far_plane) {
        mask[idx] = false;
        return;
    }

    // perspective projection
    float u = intrinsic[0] * point_camera.x / point_camera.z + intrinsic[2];
    float v = intrinsic[4] * point_camera.y / point_camera.z + intrinsic[5];

    // transform covariance from world to camera to image space
    float3 cov_image;
    transform_cov_w2c2i(
        world_to_camera,
        intrinsic,
        &scales[idx*3],
        &quaternions[idx*4],
        point_camera,
        cov_image
    );

    // compute inverse of covariance in image space
    float det = cov_image.x * cov_image.z - cov_image.y * cov_image.y;
    if (det <= 0) {
        mask[idx] = false;
        return;
    }
    float inv_det = 1.0f / det;

    // compute radius in image space
    float extend = 3.0f;  // 3-sigma
    float radius_u = ceilf(extend * sqrtf(cov_image.x));
    float radius_v = ceilf(extend * sqrtf(cov_image.z));
    float radius = fmaxf(radius_u, radius_v);

    // mask out points with too small or too large radius in image space
    if (radius < min_radius || radius > max_radius) {
        mask[idx] = false;
        return;
    }

    // mask out points outside the image plane
    if (u < -radius_u || u > (float)width + radius_u ||
        v < -radius_v || v > (float)height + radius_v) {
        mask[idx] = false;
        return;
    }

    // output results
    points_image[idx*2] = u;
    points_image[idx*2 + 1] = v;
    depth[idx] = point_camera.z;
    cov_inv_image[idx*3] =  cov_image.z * inv_det;
    cov_inv_image[idx*3 + 1] = -cov_image.y * inv_det;
    cov_inv_image[idx*3 + 2] =  cov_image.x * inv_det;
    radii[idx*2] = radius_u;
    radii[idx*2 + 1] = radius_v;
}

__global__ void evaluate_spherical_harmonics_kernel(
    const uint32_t N,
    const float *camera_pos,  // (3,)
    const float *world_points,  // (N*3,)
    const float *sh_coeffs_dc,  // (N*3,)
    const float *sh_coeffs_rest,  // (N*15*3,)
    const bool *mask,  // (N,)
    float *colors  // (N*3,)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !mask[idx]) {
        return;
    }

    // compute view direction
    float dx = world_points[idx*3] - camera_pos[0];
    float dy = world_points[idx*3 + 1] - camera_pos[1];
    float dz = world_points[idx*3 + 2] - camera_pos[2];
    float norm = sqrtf(dx * dx + dy * dy + dz * dz);
    float inv_norm = 1.0f / (norm + 1e-8f);
    float x = dx * inv_norm;
    float y = dy * inv_norm;
    float z = dz * inv_norm;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;

    // evaluate spherical harmonics basis functions
    float sh_basis[16] = {
        SH_C0,
        -SH_C1_y * y,
        SH_C1_z * z,
        -SH_C1_x * x,
        SH_C2_xy * xy,
        SH_C2_yz * yz,
        SH_C2_zz * (3 * zz - 1),
        SH_C2_xz * xz,
        SH_C2_xx_yy * (xx - yy),
        SH_C3_yxx_yyy * y * (xx - yy),
        SH_C3_xyz * (x * y * z),
        SH_C3_yzz_yxx_yyy * y * (4 * zz - xx - yy),
        SH_C3_zzz_zxx_zyy * z * (2 * zz - 3 * xx - 3 * yy),
        SH_C3_xzz_xxx_xyy * x * (4 * zz - xx - yy),
        SH_C3_zxx_zyy * z * (xx - yy),
        SH_C3_xxx_xyy * x * (xx - 3 * yy),
    };

    colors[idx*3] = sh_basis[0] * sh_coeffs_dc[idx*3];
    colors[idx*3 + 1] = sh_basis[0] * sh_coeffs_dc[idx*3 + 1];
    colors[idx*3 + 2] = sh_basis[0] * sh_coeffs_dc[idx*3 + 2];
    for (int i = 0; i < 15; ++i) {
        colors[idx*3] += sh_basis[i] * sh_coeffs_rest[(idx*15 + i)*3];
        colors[idx*3 + 1] += sh_basis[i] * sh_coeffs_rest[(idx*15 + i)*3 + 1];
        colors[idx*3 + 2] += sh_basis[i] * sh_coeffs_rest[(idx*15 + i)*3 + 2];
    }
}

void launch_project_points_kernel(
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
    const uint32_t height,
    torch::Tensor points_image,  // (N, 2)
    torch::Tensor depth,  // (N,)
    torch::Tensor cov_inv_image,  // (N, 3)
    torch::Tensor radii,  // (N, 2)
    torch::Tensor mask  // (N,)
) {
    int N = points_world.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    project_points_kernel<<<blocks, threads_per_block>>>(
        N,
        points_world.data_ptr<float>(),
        scales.data_ptr<float>(),
        quaternions.data_ptr<float>(),
        opacities.data_ptr<float>(),
        world_to_camera.data_ptr<float>(),
        intrinsic.data_ptr<float>(),
        near_plane,
        far_plane,
        min_opacity,
        min_radius,
        max_radius,
        width,
        height,
        points_image.data_ptr<float>(),
        depth.data_ptr<float>(),
        cov_inv_image.data_ptr<float>(),
        radii.data_ptr<float>(),
        mask.data_ptr<bool>()
    );
}

void launch_evaluate_spherical_harmonics_kernel(
    const torch::Tensor camera_pos,
    const torch::Tensor world_points,
    const torch::Tensor sh_coeffs_dc,
    const torch::Tensor sh_coeffs_rest,
    const torch::Tensor mask,
    torch::Tensor colors
) {
    int N = world_points.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    evaluate_spherical_harmonics_kernel<<<blocks, threads_per_block>>>(
        N,
        camera_pos.data_ptr<float>(),
        world_points.data_ptr<float>(),
        sh_coeffs_dc.data_ptr<float>(),
        sh_coeffs_rest.data_ptr<float>(),
        mask.data_ptr<bool>(),
        colors.data_ptr<float>()
    );
}