// Performs surface reconstruction, i.e. updates the internal volume with data from the current frame
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

using Vec2ida = Eigen::Matrix<int, 2, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            __global__
            void update_tsdf_kernel(const PtrStepSz<float> depth_image, const PtrStepSz<uchar3> color_image,
                                    PtrStepSz<short2> tsdf_volume, PtrStepSz<uchar3> color_volume,
                                    int3 volume_size, float voxel_scale,
                                    CameraParameters cam_params, const float truncation_distance,
                                    Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation, Vec3fda translation)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume_size.x || y >= volume_size.y)
                    return;

                for (int z = 0; z < volume_size.z; ++z) {
                    const Vec3fda position((static_cast<float>(x) + 0.5f) * voxel_scale,
                                           (static_cast<float>(y) + 0.5f) * voxel_scale,
                                           (static_cast<float>(z) + 0.5f) * voxel_scale);
                    const Vec3fda camera_pos = rotation * position + translation;

                    if (camera_pos.z() <= 0)
                        continue;

                    const Vec2ida uv(
                            __float2int_rn(camera_pos.x() / camera_pos.z() * cam_params.focal_x + cam_params.principal_x),
                            __float2int_rn(camera_pos.y() / camera_pos.z() * cam_params.focal_y + cam_params.principal_y));

                    if (uv.x() < 0 || uv.x() >= depth_image.cols || uv.y() < 0 || uv.y() >= depth_image.rows)
                        continue;

                    const float depth = depth_image.ptr(uv.y())[uv.x()];

                    if (depth <= 0)
                        continue;

                    const Vec3fda xylambda(
                            (uv.x() - cam_params.principal_x) / cam_params.focal_x,
                            (uv.y() - cam_params.principal_y) / cam_params.focal_y,
                            1.f);
                    const float lambda = xylambda.norm();

                    const float sdf = (-1.f) * ((1.f / lambda) * camera_pos.norm() - depth);

                    if (sdf >= -truncation_distance) {
                        const float new_tsdf = fmin(1.f, sdf / truncation_distance);

                        short2 voxel_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];

                        const float current_tsdf = static_cast<float>(voxel_tuple.x) * DIVSHORTMAX;
                        const int current_weight = voxel_tuple.y;

                        const int add_weight = 1;

                        const float updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                   (current_weight + add_weight);

                        const int new_weight = min(current_weight + add_weight, MAX_WEIGHT);
                        const int new_value = max(-SHORTMAX, min(SHORTMAX, static_cast<int>(updated_tsdf * SHORTMAX)));

                        tsdf_volume.ptr(z * volume_size.y + y)[x] = make_short2(static_cast<short>(new_value),
                                                                                static_cast<short>(new_weight));

                        if (sdf <= truncation_distance / 2 && sdf >= -truncation_distance / 2) {
                            uchar3& model_color = color_volume.ptr(z * volume_size.y + y)[x];
                            const uchar3 image_color = color_image.ptr(uv.y())[uv.x()];

                            model_color.x = static_cast<uchar>(
                                    (current_weight * model_color.x + add_weight * image_color.x) /
                                    (current_weight + add_weight));
                            model_color.y = static_cast<uchar>(
                                    (current_weight * model_color.y + add_weight * image_color.y) /
                                    (current_weight + add_weight));
                            model_color.z = static_cast<uchar>(
                                    (current_weight * model_color.z + add_weight * image_color.z) /
                                    (current_weight + add_weight));
                        }
                    }
                }
            }


            void surface_reconstruction(const cv::cuda::GpuMat& depth_image, const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params, const float truncation_distance,
                                        const Eigen::Matrix4f& model_view)
            {
                const dim3 threads(32, 32);
                const dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                                  (volume.volume_size.y + threads.y - 1) / threads.y);

                update_tsdf_kernel<<<blocks, threads>>>(depth_image, color_image,
                        volume.tsdf_volume, volume.color_volume,
                        volume.volume_size, volume.voxel_scale,
                        cam_params, truncation_distance,
                        model_view.block(0, 0, 3, 3), model_view.block(0, 3, 3, 1));

                cudaThreadSynchronize();
            }
        }
    }
}
