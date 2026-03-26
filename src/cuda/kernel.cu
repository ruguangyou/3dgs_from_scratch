#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAException.h>

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
    float *cov_image
) {
    // quaternion to rotation matrix
    float qx = quaternion[0];
    float qy = quaternion[1];
    float qz = quaternion[2];
    float qw = quaternion[3];
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
    //              Cc0 Cc1 Cc2
    //  J0 0  J2    Cc3 Cc4 Cc5
    //  0  J4 J5    Cc6 Cc7 Cc8
    float JCc[6] = {  // J * cov_camera
        J[0] * Cc[0] + J[2] * Cc[6],
        J[0] * Cc[1] + J[2] * Cc[7],
        J[0] * Cc[2] + J[2] * Cc[8],
        J[4] * Cc[3] + J[5] * Cc[6],
        J[4] * Cc[4] + J[5] * Cc[7],
        J[4] * Cc[5] + J[5] * Cc[8]
    };
    //                    J0 0
    //  JCc0 JCc1 JCc2    0  J4
    //  JCc3 JCc4 JCc5    J2 J5
    float Ci[4] = {  // J * cov_camera * J^T
        JCc[0] * J[0] + JCc[2] * J[2],
        JCc[1] * J[4] + JCc[2] * J[5],
        JCc[3] * J[0] + JCc[5] * J[2],
        JCc[4] * J[4] + JCc[5] * J[5]
    };

    // ensure symmetry
    cov_image[0] = Ci[0];
    cov_image[1] = 0.5f * (Ci[1] + Ci[2]);
    cov_image[2] = Ci[3];
}

__device__ void grad_point(
    const float3 &point_camera,
    const float *W,
    const float *K,
    const float *grad_points_image,
    float *grad_points_world
) {
    float x = point_camera.x;
    float y = point_camera.y;
    float z = point_camera.z;
    float inv_z = 1.0f / z;
    float inv_z2 = inv_z * inv_z;

    float grad_u = grad_points_image[0];
    float grad_v = grad_points_image[1];
    grad_points_world[0] =
        (grad_u * K[0] * (W[0] * z - W[8] * x) + grad_v * K[4] * (W[4] * z - W[8] * y)) * inv_z2;
    grad_points_world[1] =
        (grad_u * K[0] * (W[1] * z - W[9] * x) + grad_v * K[4] * (W[5] * z - W[9] * y)) * inv_z2;
    grad_points_world[2] =
        (-grad_u * K[0] * (W[10] * x - W[2] * z) - grad_v * K[4] * (W[10] * y - W[6] * z)) * inv_z2;
}

__device__ void grad_covariance(
    const float3 &point_camera,
    const float *W,
    const float *K,
    const float *scale,
    const float *quaternion,
    const float *cov_image,
    const float *grad_cov_inv_image,
    float *grad_scales,
    float *grad_quaternions
) {
    // E' = J * W * E * W^T * J^T
    //    = (J * W) * ((R * S) * S * R^T) * W^T * J^T
    //    = U * M * M^T * U^T
    
    // chain rule:
    //  dE'_inv/dS = dE'_inv/dE' * dE'/dE * dE/dM * dM/dS
    //  dE'_inv/dq = dE'_inv/dE' * dE'/dE * dE/dM * dM/dq

    // Jacobian of perspective projection
    float inv_z = 1.0f / point_camera.z;
    float inv_z2 = inv_z * inv_z;
    float J[6] = {
        K[0] * inv_z, 0, -K[0] * point_camera.x * inv_z2,
        0, K[4] * inv_z, -K[4] * point_camera.y * inv_z2
    };

    // U = J * W
    //               W0 W1 W2
    //  J0 0  J2     W4 W5 W6
    //  0  J4 J5     W8 W9 W10
    float U[6] = {
        J[0] * W[0] + J[2] * W[8],
        J[0] * W[1] + J[2] * W[9],
        J[0] * W[2] + J[2] * W[10],
        J[4] * W[4] + J[5] * W[8],
        J[4] * W[5] + J[5] * W[9],
        J[4] * W[6] + J[5] * W[10]
    };

    // quaternion to rotation matrix
    float qi = quaternion[0];
    float qj = quaternion[1];
    float qk = quaternion[2];
    float qr = quaternion[3];
    float R[9] = {
        1 - 2*qj*qj - 2*qk*qk, 2*qi*qj - 2*qk*qr, 2*qi*qk + 2*qj*qr,
        2*qi*qj + 2*qk*qr, 1 - 2*qi*qi - 2*qk*qk, 2*qj*qk - 2*qi*qr,
        2*qi*qk - 2*qj*qr, 2*qj*qk + 2*qi*qr, 1 - 2*qi*qi - 2*qj*qj
    };

    // M = R * S
    float sx = scale[0];
    float sy = scale[1];
    float sz = scale[2];
    float M[9] = {
        R[0] * sx, R[1] * sy, R[2] * sz,
        R[3] * sx, R[4] * sy, R[5] * sz,
        R[6] * sx, R[7] * sy, R[8] * sz
    };

    // dM_ij/ds_k = R_ij if j == k else 0
    float dM00_dsx = R[0];
    float dM10_dsx = R[3];
    float dM20_dsx = R[6];
    float dM01_dsy = R[1];
    float dM11_dsy = R[4];
    float dM21_dsy = R[7];
    float dM02_dsz = R[2];
    float dM12_dsz = R[5];
    float dM22_dsz = R[8];

    // dM/dq
    float dM_dqi[9] = {
        0.0f, 2.0f * sy * qj, 2.0f * sz * qk,
        2.0f * sx * qj, -4.0f * sy * qi, -2.0f * sz * qr,
        2.0f * sx * qk, 2.0f * sy * qr, -4.0f * sz * qi
    };
    float dM_dqj[9] = {
        -4.0f * sx * qj, 2.0f * sy * qi, 2.0f * sz * qr,
        2.0f * sx * qi, 0.0f, 2.0f * sz * qk,
        -2.0f * sx * qr, 2.0f * sy * qk, -4.0f * sz * qj
    };
    float dM_dqk[9] = {
        -4.0f * sx * qk, -2.0f * sy * qr, 2.0f * sz * qi,
        2.0f * sx * qr, -4.0f * sy * qk, 2.0f * sz * qj,
        2.0f * sx * qi, 2.0f * sy * qj, 0.0f
    };
    float dM_dqr[9] = {
        0.0f, -2.0f * sy * qk, 2.0f * sz * qj,
        2.0f * sx * qk, 0.0f, -2.0f * sz * qi,
        -2.0f * sx * qj, 2.0f * sy * qi, 0.0f
    };

    // dE/dM = 2 * M^T
    float dE_dM[9][9] = {
        {2.0f * M[0], M[3], M[6], M[3], 0.0f, 0.0f, M[6], 0.0f, 0.0f},
        {2.0f * M[1], M[4], M[7], M[4], 0.0f, 0.0f, M[7], 0.0f, 0.0f},
        {2.0f * M[2], M[5], M[8], M[5], 0.0f, 0.0f, M[8], 0.0f, 0.0f},
        {0.0f, M[0], 0.0f, M[0], 2.0f * M[3], M[6], 0.0f, M[6], 0.0f},
        {0.0f, M[1], 0.0f, M[1], 2.0f * M[4], M[7], 0.0f, M[7], 0.0f},
        {0.0f, M[2], 0.0f, M[2], 2.0f * M[5], M[8], 0.0f, M[8], 0.0f},
        {0.0f, 0.0f, M[0], 0.0f, 0.0f, M[3], M[0], M[3], 2.0f * M[6]},
        {0.0f, 0.0f, M[1], 0.0f, 0.0f, M[4], M[1], M[4], 2.0f * M[7]},
        {0.0f, 0.0f, M[2], 0.0f, 0.0f, M[5], M[2], M[5], 2.0f * M[8]}
    };

    // dE'/dE_ij = [U0i * U0j, U0i * U1j, U0j * U1i, U1i * U1j]
    float dEp_dE[9][4] = {
        {U[0]*U[0], U[0]*U[3], U[0]*U[3], U[3]*U[3]},
        {U[0]*U[1], U[0]*U[4], U[1]*U[3], U[3]*U[4]},
        {U[0]*U[2], U[0]*U[5], U[2]*U[3], U[3]*U[5]},
        {U[1]*U[0], U[1]*U[3], U[0]*U[4], U[4]*U[3]},
        {U[1]*U[1], U[1]*U[4], U[1]*U[4], U[4]*U[4]},
        {U[1]*U[2], U[1]*U[5], U[2]*U[4], U[4]*U[5]},
        {U[2]*U[0], U[2]*U[3], U[0]*U[5], U[5]*U[3]},
        {U[2]*U[1], U[2]*U[4], U[1]*U[5], U[5]*U[4]},
        {U[2]*U[2], U[2]*U[5], U[2]*U[5], U[5]*U[5]}
    };

    // dE'_inv/dE'
    float cov_00 = cov_image[0];
    float cov_01 = cov_image[1];
    float cov_11 = cov_image[2];
    float inv_det = 1.0f / (cov_00 * cov_11 - cov_01 * cov_01);
    float inv_det2 = inv_det * inv_det;
    float trace = cov_00 + cov_11 - 2.0f * cov_01;
    float dEpi_dEp[3] = {
        inv_det - cov_11 * trace * inv_det2,
        2.0f * (-inv_det + cov_01 * trace * inv_det2),
        inv_det - cov_00 * trace * inv_det2,
    };

    float grad_Ep[3] = {
        grad_cov_inv_image[0] * dEpi_dEp[0],
        grad_cov_inv_image[1] * dEpi_dEp[1],
        grad_cov_inv_image[2] * dEpi_dEp[2]
    };

    float grad_E[9];
    for (int i = 0; i < 9; i++) {
        grad_E[i] = (
            grad_Ep[0] * dEp_dE[i][0] + 
            grad_Ep[1] * (dEp_dE[i][1] + dEp_dE[i][2]) + 
            grad_Ep[2] * dEp_dE[i][3]
        );
    }

    float grad_M[9];
    for (int i = 0; i < 9; i++) {
        grad_M[i] = 0.0f;
        for (int j = 0; j < 9; j++) {
            grad_M[i] += grad_E[j] * dE_dM[i][j];
        }
    }

    grad_scales[0] = grad_M[0] * dM00_dsx + grad_M[3] * dM10_dsx + grad_M[6] * dM20_dsx;
    grad_scales[1] = grad_M[1] * dM01_dsy + grad_M[4] * dM11_dsy + grad_M[7] * dM21_dsy;
    grad_scales[2] = grad_M[2] * dM02_dsz + grad_M[5] * dM12_dsz + grad_M[8] * dM22_dsz;
    for (int i = 0; i < 9; i++) {
        grad_quaternions[0] += grad_M[i] * dM_dqi[i];
        grad_quaternions[1] += grad_M[i] * dM_dqj[i];
        grad_quaternions[2] += grad_M[i] * dM_dqk[i];
        grad_quaternions[3] += grad_M[i] * dM_dqr[i];
    }
}

__global__ void project_points_kernel(
    const int32_t N,
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
    const int32_t width,
    const int32_t height,
    float *points_image,  // (N*2,)
    float *depths,  // (N,)
    float *cov_image,  // (N*3,)
    float *cov_inv_image,  // (N*3,)
    float *radii,  // (N*2,)
    bool *mask  // (N,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    transform_cov_w2c2i(
        world_to_camera,
        intrinsic,
        &scales[idx*3],
        &quaternions[idx*4],
        point_camera,
        &cov_image[idx*3]
    );

    // compute inverse of covariance in image space
    float det = cov_image[idx*3] * cov_image[idx*3 + 2] - cov_image[idx*3 + 1] * cov_image[idx*3 + 1];
    if (det <= 0) {
        mask[idx] = false;
        return;
    }
    float inv_det = 1.0f / det;

    // compute radius in image space
    float extend = 3.0f;  // 3-sigma
    float radius_u = ceilf(extend * sqrtf(cov_image[idx*3]));
    float radius_v = ceilf(extend * sqrtf(cov_image[idx*3 + 2]));
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
    depths[idx] = point_camera.z;
    cov_inv_image[idx*3] =  cov_image[idx*3 + 2] * inv_det;
    cov_inv_image[idx*3 + 1] = -cov_image[idx*3 + 1] * inv_det;
    cov_inv_image[idx*3 + 2] = cov_image[idx*3] * inv_det;
    radii[idx*2] = radius_u;
    radii[idx*2 + 1] = radius_v;
}

__global__ void project_points_backward_kernel(
    const int32_t N,
    const float *grad_points_image,  // (N*2,)
    const float *grad_cov_inv_image,  // (N*3,)
    const float *points_world,  // (N*3,)
    const float *scales,  // (N*3,)
    const float *quaternions,  // (N*4,)
    const float *opacities,  // (N,)
    const float *world_to_camera,  // (4*4,)
    const float *intrinsic,  // (3*3,)
    const float *cov_image,  // (N*3,)
    const bool *mask,  // (N,)
    float *grad_points_world,  // (N*3,)
    float *grad_scales,  // (N*3,)
    float *grad_quaternions  // (N*4,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !mask[idx]) {
        return;
    }

    float3 point_camera;
    transform_w2c(world_to_camera, &points_world[idx*3], point_camera);

    // compute gradients w.r.t. point_world
    grad_point(
        point_camera,
        world_to_camera,
        intrinsic,
        &grad_points_image[idx*2],
        &grad_points_world[idx*3]
    );
    
    // compute gradients w.r.t. scale and quaternion
    grad_covariance(
        point_camera,
        world_to_camera,
        intrinsic,
        &scales[idx*3],
        &quaternions[idx*4],
        &cov_image[idx*3],
        &grad_cov_inv_image[idx*3],
        &grad_scales[idx*3],
        &grad_quaternions[idx*4]
    );
}

__global__ void evaluate_spherical_harmonics_kernel(
    const int32_t N,
    const float *camera_pos,  // (3,)
    const float *points_world,  // (N*3,)
    const float *sh_coeffs_dc,  // (N*3,)
    const float *sh_coeffs_rest,  // (N*15*3,)
    const bool *mask,  // (N,)
    float *colors  // (N*3,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !mask[idx]) {
        return;
    }

    // compute view direction
    float dx = points_world[idx*3] - camera_pos[0];
    float dy = points_world[idx*3 + 1] - camera_pos[1];
    float dz = points_world[idx*3 + 2] - camera_pos[2];
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

    float r = sh_basis[0] * sh_coeffs_dc[idx*3];
    float g = sh_basis[0] * sh_coeffs_dc[idx*3 + 1];
    float b = sh_basis[0] * sh_coeffs_dc[idx*3 + 2];
    for (int i = 0; i < 15; ++i) {
        r += sh_basis[i+1] * sh_coeffs_rest[(idx*15 + i)*3];
        g += sh_basis[i+1] * sh_coeffs_rest[(idx*15 + i)*3 + 1];
        b += sh_basis[i+1] * sh_coeffs_rest[(idx*15 + i)*3 + 2];
    }

    // normalize color to [0, 1] with sigmoid
    colors[idx*3] = 1.0 / (1.0 + expf(-r));
    colors[idx*3 + 1] = 1.0 / (1.0 + expf(-g));
    colors[idx*3 + 2] = 1.0 / (1.0 + expf(-b));
}

__global__ void evaluate_spherical_harmonics_backward_kernel(
    const int32_t N,
    const float *grad_colors,  // (N*3,)
    const float *camera_pos,  // (3,)
    const float *points_world,  // (N*3,)
    const float *sh_coeffs_dc,  // (N*3,)
    const float *sh_coeffs_rest,  // (N*15*3,)
    const bool *mask,  // (N,)
    float *grad_points_world,  // (N*3,)
    float *grad_sh_coeffs_dc,  // (N*3,)
    float *grad_sh_coeffs_rest  // (N*15*3,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !mask[idx]) {
        return;
    }

    // compute view direction
    float dx = points_world[idx*3] - camera_pos[0];
    float dy = points_world[idx*3 + 1] - camera_pos[1];
    float dz = points_world[idx*3 + 2] - camera_pos[2];
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

    // compute local gradients w.r.t. point
    float grad_px = (norm * norm - dx * dx) * inv_norm * inv_norm * inv_norm;
    float grad_py = (norm * norm - dy * dy) * inv_norm * inv_norm * inv_norm;
    float grad_pz = (norm * norm - dz * dz) * inv_norm * inv_norm * inv_norm;
    float grad_pxx = 2 * x * grad_px;
    float grad_pyy = 2 * y * grad_py;
    float grad_pzz = 2 * z * grad_pz;
    float grad_sh_rest_px[15] = {
        0.0f,
        0.0f,
        -SH_C1_x * grad_px,
        SH_C2_xy * y * grad_px,
        0.0f,
        0.0f,
        SH_C2_xz * z * grad_px,
        SH_C2_xx_yy * grad_pxx,
        SH_C3_yxx_yyy * y * grad_pxx,
        SH_C3_xyz * y * z * grad_px,
        SH_C3_yzz_yxx_yyy * y * (- grad_pxx),
        SH_C3_zzz_zxx_zyy * z * (-3 * grad_pxx),
        SH_C3_xzz_xxx_xyy * (x * (-grad_pxx) + grad_px * (4 * zz - xx - yy)),
        SH_C3_zxx_zyy * z * grad_pxx,
        SH_C3_xxx_xyy * (x * grad_pxx + grad_px * (xx - 3 * yy)),
    };
    float grad_sh_rest_py[15] = {
        -SH_C1_y * grad_py,
        0.0f,
        0.0f,
        SH_C2_xy * x * grad_py,
        SH_C2_yz * z * grad_py,
        0.0f,
        0.0f,
        SH_C2_xx_yy * (-grad_pyy),
        SH_C3_yxx_yyy * (grad_py * (xx - yy) + y * (-grad_pyy)),
        SH_C3_xyz * x * z * grad_py,
        SH_C3_yzz_yxx_yyy * (grad_py * (4 * zz - xx - yy) + y * (-grad_pyy)),
        SH_C3_zzz_zxx_zyy * z * (-3 * grad_pyy),
        SH_C3_xzz_xxx_xyy * x * (-grad_pyy),
        SH_C3_zxx_zyy * z * (-grad_pyy),
        SH_C3_xxx_xyy * x * (-3 * grad_pyy),
    };
    float grad_sh_rest_pz[15] = {
        0.0f,
        SH_C1_z * grad_pz,
        0.0f,
        0.0f,
        SH_C2_yz * y * grad_pz,
        SH_C2_zz * 3 * grad_pzz,
        SH_C2_xz * x * grad_pz,
        0.0f,
        0.0f,
        SH_C3_xyz * x * y * grad_pz,
        SH_C3_yzz_yxx_yyy * y * (4 * grad_pzz),
        SH_C3_zzz_zxx_zyy * (grad_pz * (2 * zz - 3 * xx - 3 * yy) + z * (2 * grad_pzz)),
        SH_C3_xzz_xxx_xyy * x * (4 * grad_pzz),
        SH_C3_zxx_zyy * grad_pz * (xx - yy),
        0.0f,
    };

    // compute sigmoid gradient
    float grad_r = grad_colors[idx*3] * (1.0f - grad_colors[idx*3]);
    float grad_g = grad_colors[idx*3 + 1] * (1.0f - grad_colors[idx*3 + 1]);
    float grad_b = grad_colors[idx*3 + 2] * (1.0f - grad_colors[idx*3 + 2]);

    // compute gradient w.r.t. sh_coeffs_dc
    grad_sh_coeffs_dc[idx*3] = grad_r * sh_basis[0];
    grad_sh_coeffs_dc[idx*3 + 1] = grad_g * sh_basis[0];
    grad_sh_coeffs_dc[idx*3 + 2] = grad_b * sh_basis[0];

    // compute gradient w.r.t. sh_coeffs_rest
    for (int i = 0; i < 15; ++i) {
        grad_sh_coeffs_rest[(idx*15 + i)*3] = grad_r * sh_basis[i+1];
        grad_sh_coeffs_rest[(idx*15 + i)*3 + 1] = grad_g * sh_basis[i+1];
        grad_sh_coeffs_rest[(idx*15 + i)*3 + 2] = grad_b * sh_basis[i+1];
    }

    // compute gradient w.r.t. points_world
    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_z = 0.0f;
    for (int i = 0; i < 15; ++i) {
        grad_x += grad_r * sh_coeffs_rest[(idx*15 + i)*3] * grad_sh_rest_px[i] +
                  grad_g * sh_coeffs_rest[(idx*15 + i)*3 + 1] * grad_sh_rest_px[i] +
                  grad_b * sh_coeffs_rest[(idx*15 + i)*3 + 2] * grad_sh_rest_px[i];
        grad_y += grad_r * sh_coeffs_rest[(idx*15 + i)*3] * grad_sh_rest_py[i] +
                  grad_g * sh_coeffs_rest[(idx*15 + i)*3 + 1] * grad_sh_rest_py[i] +
                  grad_b * sh_coeffs_rest[(idx*15 + i)*3 + 2] * grad_sh_rest_py[i];
        grad_z += grad_r * sh_coeffs_rest[(idx*15 + i)*3] * grad_sh_rest_pz[i] +
                  grad_g * sh_coeffs_rest[(idx*15 + i)*3 + 1] * grad_sh_rest_pz[i] +
                  grad_b * sh_coeffs_rest[(idx*15 + i)*3 + 2] * grad_sh_rest_pz[i];
    }
    grad_points_world[idx*3] = grad_x;
    grad_points_world[idx*3 + 1] = grad_y;
    grad_points_world[idx*3 + 2] = grad_z;
}

__global__ void count_tiles_kernel(
    const int32_t N,
    const float *points_image,  // (N*2,)
    const float *radii,  // (N*2,)
    const bool *mask,  // (N,)
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    int32_t *num_tiles  // (N,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !mask[idx]) {
        return;
    }

    float u = points_image[idx*2];
    float v = points_image[idx*2 + 1];
    float radius_u = radii[idx*2];
    float radius_v = radii[idx*2 + 1];

    int32_t u_min = min(max(0, (int32_t)floorf((u - radius_u))), width - 1);
    int32_t u_max = min(max(0, (int32_t)floorf((u + radius_u))), width - 1);
    int32_t v_min = min(max(0, (int32_t)floorf((v - radius_v))), height - 1);
    int32_t v_max = min(max(0, (int32_t)floorf((v + radius_v))), height - 1);
    if (u_min > u_max || v_min > v_max) {
        num_tiles[idx] = 0;
        return;
    }

    int32_t tile_u_min = u_min / tile_size;
    int32_t tile_u_max = u_max / tile_size;
    int32_t tile_v_min = v_min / tile_size;
    int32_t tile_v_max = v_max / tile_size;
    num_tiles[idx] = (tile_u_max - tile_u_min + 1) * (tile_v_max - tile_v_min + 1);
}

__global__ void compute_tile_intersection_kernel(
    const int32_t N,
    const float *points_image,  // (N*2,)
    const float *radii,  // (N*2,)
    const float *depths,  // (N,)
    const int64_t *cum_num_tiles,  // (N,)
    const bool *mask,  // (N,)
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    int64_t *tile_ids_encoded_depth,  // (total_tiles,)
    int32_t *gaussian_ids  // (total_tiles,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !mask[idx]) {
        return;
    }

    float u = points_image[idx*2];
    float v = points_image[idx*2 + 1];
    float radius_u = radii[idx*2];
    float radius_v = radii[idx*2 + 1];

    int32_t u_min = min(max(0, (int32_t)floorf((u - radius_u))), width - 1);
    int32_t u_max = min(max(0, (int32_t)floorf((u + radius_u))), width - 1);
    int32_t v_min = min(max(0, (int32_t)floorf((v - radius_v))), height - 1);
    int32_t v_max = min(max(0, (int32_t)floorf((v + radius_v))), height - 1);
    if (u_min > u_max || v_min > v_max) {
        return;
    }
    
    int32_t num_tiles_per_row = (width + tile_size - 1) / tile_size;
    int32_t tile_u_min = u_min / tile_size;
    int32_t tile_u_max = u_max / tile_size;
    int32_t tile_v_min = v_min / tile_size;
    int32_t tile_v_max = v_max / tile_size;

    // reinterpret float depth as int32 in bit-level (keep original bits)
    int32_t depth_i32 = *((int32_t *)(&depths[idx]));
    // instead of directly casting to int64_t which would sign-extend in case of negative depth
    int64_t depth_i64 = (int64_t)depth_i32 & 0xFFFFFFFF;

    // starting index for this gaussian in the output arrays
    int64_t output_idx = idx > 0 ? cum_num_tiles[idx-1] : 0;
    for (int32_t tile_v = tile_v_min; tile_v <= tile_v_max; ++tile_v) {
        for (int32_t tile_u = tile_u_min; tile_u <= tile_u_max; ++tile_u) {
            int64_t tile_id_i64 = (tile_v * num_tiles_per_row) + tile_u;
            // encode tile id (upper 32 bits) and depth (lower 32 bits) for sorting
            tile_ids_encoded_depth[output_idx] = (tile_id_i64 << 32) | depth_i64;
            gaussian_ids[output_idx] = idx;
            ++output_idx;
        }
    }
}

__global__ void compute_indexing_offset_kernel(
    const int32_t total_tiles,
    const int32_t unique_tiles,
    const int64_t *tile_ids_encoded_depth_sorted,  // (total_tiles,)
    int32_t *indexing_offset  // (tile_width * tile_height,)
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tiles) {
        return;
    }

    // decode tile id by shifting right 32 bits
    int32_t tile_id = (int32_t)(tile_ids_encoded_depth_sorted[idx] >> 32);

    // e.g. tile_ids_sorted=[1, 1, 2, 3, 3, 5], total_tiles=6, unique_tiles=8
    // then indexing_offset=[0, 0, 2, 3, 5, 5, 6, 6]
    // indexing_offset[tile_id] <= offset < indexing_offset[tile_id+1] give the range
    if (idx == 0) {
        // set offset of the tiles before the fisrt tile having gaussians to 0 (inclusive)
        for (int32_t i = 0; i <= tile_id; ++i) {
            indexing_offset[i] = 0;
        }
    }
    else if (idx == total_tiles - 1) {
        // set offset of the tiles after the last tile having gaussians to total_tiles
        for (int32_t i = tile_id + 1; i < unique_tiles; ++i) {
            indexing_offset[i] = total_tiles;
        }
    }
    else {
        // set offset of the tiles between the previous tile and the current tile to idx
        int32_t prev_tile_id = (int32_t)(tile_ids_encoded_depth_sorted[idx - 1] >> 32);
        for (int32_t i = prev_tile_id + 1; i <= tile_id; ++i) {
            indexing_offset[i] = idx;
        }
    }
}

__global__ void rasterize_kernel(
    const int32_t *indexing_offset,  // (unique_tiles,)
    const int32_t *gaussian_ids_sorted, // (total_tiles,)
    const float *points_image,  // (N*2,)
    const float *cov_inv_image,  // (N*3,)
    const float *opacities,  // (N,)
    const float *colors,  // (N*3,)
    const int32_t unique_tiles,
    const int32_t total_tiles,
    const int32_t num_tiles_per_row,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    const float alpha_threshold,
    const float transmittance_threshold,
    const float chi_squared_threshold,
    float *output_image  // (height*width*3,)
) {
    int32_t tile_id = blockIdx.x;  // each block processes one tile
    int32_t idx_in_tile = threadIdx.x;  // each thread processes one pixel
    if (tile_id >= unique_tiles) {
        return;
    }

    int32_t row = tile_id / num_tiles_per_row;
    int32_t col = tile_id % num_tiles_per_row;
    int32_t u = col * tile_size + (idx_in_tile % tile_size);
    int32_t v = row * tile_size + (idx_in_tile / tile_size);
    if (u >= width || v >= height) {
        return;
    }

    float transmittance = 1.0f;
    int32_t range_start = indexing_offset[tile_id];
    int32_t range_end = (tile_id + 1 < unique_tiles) ? indexing_offset[tile_id+1] : total_tiles;
    for (int32_t i = range_start; i < range_end; ++i) {
        int32_t gaussian_id = gaussian_ids_sorted[i];
        float du = u - points_image[gaussian_id*2];
        float dv = v - points_image[gaussian_id*2 + 1];
        float inv_cov_00 = cov_inv_image[gaussian_id*3];
        float inv_cov_01 = cov_inv_image[gaussian_id*3 + 1];
        float inv_cov_11 = cov_inv_image[gaussian_id*3 + 2];

        float exponent = inv_cov_00 * du * du + 2.0f * inv_cov_01 * du * dv + inv_cov_11 * dv * dv;
        if (exponent > chi_squared_threshold) {
            continue;
        }

        float alpha = expf(-0.5f * exponent) * opacities[gaussian_id];
        if (alpha < alpha_threshold) {
            continue;
        }

        float weight = alpha * transmittance;
        atomicAdd(&output_image[(v * width + u) * 3], weight * colors[gaussian_id*3]);
        atomicAdd(&output_image[(v * width + u) * 3 + 1], weight * colors[gaussian_id*3 + 1]);
        atomicAdd(&output_image[(v * width + u) * 3 + 2], weight * colors[gaussian_id*3 + 2]);

        transmittance *= (1.0f - alpha);
        if (transmittance < transmittance_threshold) {
            break;
        }
    }
}

__global__ void rasterize_backward_kernel(
    const float *grad_output_image,  // (height*width*3,)
    const int32_t *indexing_offset,  // (unique_tiles,)
    const int32_t *gaussian_ids_sorted, // (total_tiles,)
    const float *points_image,  // (N*2,)
    const float *cov_inv_image,  // (N*3,)
    const float *opacities,  // (N,)
    const float *colors,  // (N*3,)
    const int32_t unique_tiles,
    const int32_t total_tiles,
    const int32_t num_tiles_per_row,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    const float alpha_threshold,
    const float transmittance_threshold,
    const float chi_squared_threshold,
    float *grad_points_image,  // (N*2,)
    float *grad_cov_inv_image,  // (N*3,)
    float *grad_opacities,  // (N,)
    float *grad_colors  // (N*3,)
) {
    int32_t tile_id = blockIdx.x;  // each block processes one tile
    int32_t idx_in_tile = threadIdx.x;  // each thread processes one pixel
    if (tile_id >= unique_tiles) {
        return;
    }

    int32_t row = tile_id / num_tiles_per_row;
    int32_t col = tile_id % num_tiles_per_row;
    int32_t u = col * tile_size + (idx_in_tile % tile_size);
    int32_t v = row * tile_size + (idx_in_tile / tile_size);
    if (u >= width || v >= height) {
        return;
    }

    float grad_r = grad_output_image[(v * width + u) * 3];
    float grad_g = grad_output_image[(v * width + u) * 3 + 1];
    float grad_b = grad_output_image[(v * width + u) * 3 + 2];
    
    float transmittance = 1.0f;
    int32_t range_start = indexing_offset[tile_id];
    int32_t range_end = (tile_id + 1 < unique_tiles) ? indexing_offset[tile_id+1] : total_tiles;
    for (int32_t i = range_start; i < range_end; ++i) {
        int32_t gaussian_id = gaussian_ids_sorted[i];
        float du = u - points_image[gaussian_id*2];
        float dv = v - points_image[gaussian_id*2 + 1];
        float inv_cov_00 = cov_inv_image[gaussian_id*3];
        float inv_cov_01 = cov_inv_image[gaussian_id*3 + 1];
        float inv_cov_11 = cov_inv_image[gaussian_id*3 + 2];

        float exponent = inv_cov_00 * du * du + 2.0f * inv_cov_01 * du * dv + inv_cov_11 * dv * dv;
        if (exponent > chi_squared_threshold) {
            continue;
        }
        float density = expf(-0.5f * exponent);
        float alpha = density * opacities[gaussian_id];
        if (alpha < alpha_threshold) {
            continue;
        }
        float weight = alpha * transmittance;

        // compute gradient w.r.t. colors
        atomicAdd(&grad_colors[gaussian_id*3], weight * grad_r);
        atomicAdd(&grad_colors[gaussian_id*3 + 1], weight * grad_g);
        atomicAdd(&grad_colors[gaussian_id*3 + 2], weight * grad_b);

        // compute gradient w.r.t. opacity
        atomicAdd(&grad_opacities[gaussian_id], density * transmittance * (grad_r + grad_g + grad_b));

        // compute gradients w.r.t. points_image
        atomicAdd(&grad_points_image[gaussian_id*2], weight * (inv_cov_00 * du + inv_cov_01 * dv));
        atomicAdd(&grad_points_image[gaussian_id*2 + 1], weight * (inv_cov_01 * du + inv_cov_11 * dv));

        // compute gradients w.r.t. cov_inv_image
        atomicAdd(&grad_cov_inv_image[gaussian_id*3], weight * (-0.5f * du * du));
        atomicAdd(&grad_cov_inv_image[gaussian_id*3 + 1], weight * (-du * dv));
        atomicAdd(&grad_cov_inv_image[gaussian_id*3 + 2], weight * (-0.5f * dv * dv));

        transmittance *= (1.0f - alpha);
        if (transmittance < transmittance_threshold) {
            break;
        }
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
    const int32_t width,
    const int32_t height,
    torch::Tensor points_image,  // (N, 2)
    torch::Tensor depths,  // (N,)
    torch::Tensor cov_image,  // (N, 3)
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
        depths.data_ptr<float>(),
        cov_image.data_ptr<float>(),
        cov_inv_image.data_ptr<float>(),
        radii.data_ptr<float>(),
        mask.data_ptr<bool>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_project_points_backward_kernel(
    const torch::Tensor grad_points_image,  // (N, 2)
    const torch::Tensor grad_cov_inv_image,  // (N, 3)
    const torch::Tensor points_world,  // (N, 3)
    const torch::Tensor scales,  // (N, 3)
    const torch::Tensor quaternions,  // (N, 4)
    const torch::Tensor opacities,  // (N,)
    const torch::Tensor world_to_camera,  // (4, 4)
    const torch::Tensor intrinsic,  // (3, 3)
    const torch::Tensor cov_image,  // (N, 3)
    const torch::Tensor mask,  // (N,)
    torch::Tensor grad_points_world,  // (N, 3)
    torch::Tensor grad_scales,  // (N, 3)
    torch::Tensor grad_quaternions  // (N, 4)
) {
    int N = points_world.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    project_points_backward_kernel<<<blocks, threads_per_block>>>(
        N,
        grad_points_image.data_ptr<float>(),
        grad_cov_inv_image.data_ptr<float>(),
        points_world.data_ptr<float>(),
        scales.data_ptr<float>(),
        quaternions.data_ptr<float>(),
        opacities.data_ptr<float>(),
        world_to_camera.data_ptr<float>(),
        intrinsic.data_ptr<float>(),
        cov_image.data_ptr<float>(),
        mask.data_ptr<bool>(),
        grad_points_world.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_quaternions.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_evaluate_spherical_harmonics_kernel(
    const torch::Tensor camera_pos,
    const torch::Tensor points_world,
    const torch::Tensor sh_coeffs_dc,
    const torch::Tensor sh_coeffs_rest,
    const torch::Tensor mask,
    torch::Tensor colors
) {
    int N = points_world.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    evaluate_spherical_harmonics_kernel<<<blocks, threads_per_block>>>(
        N,
        camera_pos.data_ptr<float>(),
        points_world.data_ptr<float>(),
        sh_coeffs_dc.data_ptr<float>(),
        sh_coeffs_rest.data_ptr<float>(),
        mask.data_ptr<bool>(),
        colors.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_evaluate_spherical_harmonics_backward_kernel(
    const torch::Tensor grad_colors,
    const torch::Tensor camera_pos,
    const torch::Tensor points_world,
    const torch::Tensor sh_coeffs_dc,
    const torch::Tensor sh_coeffs_rest,
    const torch::Tensor mask,
    torch::Tensor grad_points_world,
    torch::Tensor grad_sh_coeffs_dc,
    torch::Tensor grad_sh_coeffs_rest
) {
    int N = points_world.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    evaluate_spherical_harmonics_backward_kernel<<<blocks, threads_per_block>>>(
        N,
        grad_colors.data_ptr<float>(),
        camera_pos.data_ptr<float>(),
        points_world.data_ptr<float>(),
        sh_coeffs_dc.data_ptr<float>(),
        sh_coeffs_rest.data_ptr<float>(),
        mask.data_ptr<bool>(),
        grad_points_world.data_ptr<float>(),
        grad_sh_coeffs_dc.data_ptr<float>(),
        grad_sh_coeffs_rest.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_count_tiles_kernel(
    const torch::Tensor points_image,
    const torch::Tensor radii,
    const torch::Tensor mask,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    torch::Tensor num_tiles
) {
    int N = points_image.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    count_tiles_kernel<<<blocks, threads_per_block>>>(
        N,
        points_image.data_ptr<float>(),
        radii.data_ptr<float>(),
        mask.data_ptr<bool>(),
        width,
        height,
        tile_size,
        num_tiles.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_compute_tile_intersection_kernel(
    const torch::Tensor points_image,
    const torch::Tensor radii,
    const torch::Tensor depths,
    const torch::Tensor cum_num_tiles,
    const torch::Tensor mask,
    const int32_t width,
    const int32_t height,
    const int32_t tile_size,
    torch::Tensor tile_ids_encoded_depth,
    torch::Tensor gaussian_ids
) {
    int N = points_image.size(0);
    if (N == 0) {
        return;
    }

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    compute_tile_intersection_kernel<<<blocks, threads_per_block>>>(
        N,
        points_image.data_ptr<float>(),
        radii.data_ptr<float>(),
        depths.data_ptr<float>(),
        cum_num_tiles.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        width,
        height,
        tile_size,
        tile_ids_encoded_depth.data_ptr<int64_t>(),
        gaussian_ids.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void radix_sort_double_buffer(
    const int64_t num_items,
    const torch::Tensor keys_in,  // tile_ids_encoded_depth
    const torch::Tensor values_in,  // gaussian_ids
    torch::Tensor keys_out,
    torch::Tensor values_out
) {
    if (num_items == 0) {
        return;
    }

    cub::DoubleBuffer<int64_t> d_keys(
        keys_in.data_ptr<int64_t>(), keys_out.data_ptr<int64_t>());
    cub::DoubleBuffer<int32_t> d_values(
        values_in.data_ptr<int32_t>(), values_out.data_ptr<int32_t>());

    // determine temporary device storage requirements
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    auto sort_status = cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_keys, d_values, num_items);
    C10_CUDA_CHECK(sort_status);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // run sorting operation
    sort_status = cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes, d_keys, d_values, num_items);
    C10_CUDA_CHECK(sort_status);
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaFree(d_temp));

    // get sorted results
    switch (d_keys.selector) {
        case 0:  // sorted items are stored in keys_in
            keys_out.set_(keys_in);
            break;
        case 1:  // sorted items are stored in keys_out
            break;
    }
    switch (d_values.selector) {
        case 0:  // sorted items are stored in values_in
            values_out.set_(values_in);
            break;
        case 1:  // sorted items are stored in values_out
            break;
    }
}

void launch_compute_indexing_offset_kernel(
    const torch::Tensor tile_ids_encoded_depth_sorted,
    torch::Tensor indexing_offset
) {
    int total_tiles = tile_ids_encoded_depth_sorted.size(0);
    if (total_tiles > 0) {
        int unique_tiles = indexing_offset.size(0);
        int threads_per_block = 256;
        int blocks = (total_tiles + threads_per_block - 1) / threads_per_block;
        compute_indexing_offset_kernel<<<blocks, threads_per_block>>>(
            total_tiles,
            unique_tiles,
            tile_ids_encoded_depth_sorted.data_ptr<int64_t>(),
            indexing_offset.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void launch_rasterize_kernel(
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
    const float chi_squared_threshold,
    torch::Tensor rendered_image  // (height, width, 3)
) {
    int unique_tiles = indexing_offset.size(0);
    int total_tiles = gaussian_ids_sorted.size(0);
    if (unique_tiles == 0 || total_tiles == 0) {
        return;
    }

    int num_tiles_per_row = (width + tile_size - 1) / tile_size;
    int threads_per_block = tile_size * tile_size;  // one thread per pixel in a tile
    int blocks = unique_tiles;  // one block per tile
    rasterize_kernel<<<blocks, threads_per_block>>>(
        indexing_offset.data_ptr<int32_t>(),
        gaussian_ids_sorted.data_ptr<int32_t>(),
        points_image.data_ptr<float>(),
        cov_inv_image.data_ptr<float>(),
        opacities.data_ptr<float>(),
        colors.data_ptr<float>(),
        unique_tiles,
        total_tiles,
        num_tiles_per_row,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
        rendered_image.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
) {
    int unique_tiles = indexing_offset.size(0);
    int total_tiles = gaussian_ids_sorted.size(0);
    if (unique_tiles == 0 || total_tiles == 0) {
        return;
    }

    int num_tiles_per_row = (width + tile_size - 1) / tile_size;
    int threads_per_block = tile_size * tile_size;  // one thread per pixel in a tile
    int blocks = unique_tiles;  // one block per tile
    rasterize_backward_kernel<<<blocks, threads_per_block>>>(
        grad_rendered_image.data_ptr<float>(),
        indexing_offset.data_ptr<int32_t>(),
        gaussian_ids_sorted.data_ptr<int32_t>(),
        points_image.data_ptr<float>(),
        cov_inv_image.data_ptr<float>(),
        opacities.data_ptr<float>(),
        colors.data_ptr<float>(),
        unique_tiles,
        total_tiles,
        num_tiles_per_row,
        width,
        height,
        tile_size,
        alpha_threshold,
        transmittance_threshold,
        chi_squared_threshold,
        grad_points_image.data_ptr<float>(),
        grad_cov_inv_image.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_colors.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
