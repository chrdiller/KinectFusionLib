// Predicts the surface, i.e. performs raycasting
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

using Vec3ida = Eigen::Matrix<int, 3, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            __device__ __forceinline__
            float interpolate_trilinearly(const Vec3fda& point, const PtrStepSz<short2>& volume,
                                          const int3& volume_size, const float voxel_scale)
            {
                Vec3ida point_in_grid = point.cast<int>();

                const float vx = (static_cast<float>(point_in_grid.x()) + 0.5f);
                const float vy = (static_cast<float>(point_in_grid.y()) + 0.5f);
                const float vz = (static_cast<float>(point_in_grid.z()) + 0.5f);

                point_in_grid.x() = (point.x() < vx) ? (point_in_grid.x() - 1) : point_in_grid.x();
                point_in_grid.y() = (point.y() < vy) ? (point_in_grid.y() - 1) : point_in_grid.y();
                point_in_grid.z() = (point.z() < vz) ? (point_in_grid.z() - 1) : point_in_grid.z();

                const float a = (point.x() - (static_cast<float>(point_in_grid.x()) + 0.5f));
                const float b = (point.y() - (static_cast<float>(point_in_grid.y()) + 0.5f));
                const float c = (point.z() - (static_cast<float>(point_in_grid.z()) + 0.5f));

                return static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX * (1 - a) * (1 - b) * (1 - c) +
                       static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x()].x) * DIVSHORTMAX * (1 - a) * (1 - b) * c +
                       static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX * (1 - a) * b * (1 - c) +
                       static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x()].x) * DIVSHORTMAX * (1 - a) * b * c +
                       static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX * a * (1 - b) * (1 - c) +
                       static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y())[point_in_grid.x() + 1].x) * DIVSHORTMAX * a * (1 - b) * c +
                       static_cast<float>(volume.ptr((point_in_grid.z()) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX * a * b * (1 - c) +
                       static_cast<float>(volume.ptr((point_in_grid.z() + 1) * volume_size.y + point_in_grid.y() + 1)[point_in_grid.x() + 1].x) * DIVSHORTMAX * a * b * c;
            }


            __device__ __forceinline__
            float get_min_time(const float3& volume_max, const Vec3fda& origin, const Vec3fda& direction)
            {
                float txmin = ((direction.x() > 0 ? 0.f : volume_max.x) - origin.x()) / direction.x();
                float tymin = ((direction.y() > 0 ? 0.f : volume_max.y) - origin.y()) / direction.y();
                float tzmin = ((direction.z() > 0 ? 0.f : volume_max.z) - origin.z()) / direction.z();

                return fmax(fmax(txmin, tymin), tzmin);
            }

            __device__ __forceinline__
            float get_max_time(const float3& volume_max, const Vec3fda& origin, const Vec3fda& direction)
            {
                float txmax = ((direction.x() > 0 ? volume_max.x : 0.f) - origin.x()) / direction.x();
                float tymax = ((direction.y() > 0 ? volume_max.y : 0.f) - origin.y()) / direction.y();
                float tzmax = ((direction.z() > 0 ? volume_max.z : 0.f) - origin.z()) / direction.z();

                return fmin(fmin(txmax, tymax), tzmax);
            }

            __global__
            void raycast_tsdf_kernel(const PtrStepSz<short2> tsdf_volume, const PtrStepSz<uchar3> color_volume,
                                     PtrStepSz<float3> model_vertex, PtrStepSz<float3> model_normal,
                                     PtrStepSz<uchar3> model_color,
                                     const int3 volume_size, const float voxel_scale,
                                     const CameraParameters cam_parameters,
                                     const float truncation_distance,
                                     const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation,
                                     const Vec3fda translation)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= model_vertex.cols || y >= model_vertex.rows)
                    return;

                const float3 volume_range = make_float3(volume_size.x * voxel_scale,
                                                        volume_size.y * voxel_scale,
                                                        volume_size.z * voxel_scale);

                const Vec3fda pixel_position(
                        (x - cam_parameters.principal_x) / cam_parameters.focal_x,
                        (y - cam_parameters.principal_y) / cam_parameters.focal_y,
                        1.f);

                Vec3fda ray_direction = (rotation * pixel_position);
                ray_direction.normalize();

                float ray_length = fmax(get_min_time(volume_range, translation, ray_direction), 0.f);
                if (ray_length >= get_max_time(volume_range, translation, ray_direction))
                    return;

                ray_length += voxel_scale;
                Vec3fda grid = (translation + (ray_direction * ray_length)) / voxel_scale;

                float tsdf = static_cast<float>(tsdf_volume.ptr(
                        __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(grid(0))].x) *
                             DIVSHORTMAX;

                const float max_search_length = ray_length + volume_range.x * sqrt(2.f);
                for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f) {
                    grid = ((translation + (ray_direction * (ray_length + truncation_distance * 0.5f))) / voxel_scale);

                    if (grid.x() < 1 || grid.x() >= volume_size.x - 1 || grid.y() < 1 ||
                        grid.y() >= volume_size.y - 1 ||
                        grid.z() < 1 || grid.z() >= volume_size.z - 1)
                        continue;

                    const float previous_tsdf = tsdf;
                    tsdf = static_cast<float>(tsdf_volume.ptr(
                            __float2int_rd(grid(2)) * volume_size.y + __float2int_rd(grid(1)))[__float2int_rd(
                            grid(0))].x) *
                           DIVSHORTMAX;

                    if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
                        break;
                    if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
                        const float t_star =
                                ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);

                        const auto vertex = translation + ray_direction * t_star;

                        const Vec3fda location_in_grid = (vertex / voxel_scale);
                        if (location_in_grid.x() < 1 | location_in_grid.x() >= volume_size.x - 1 ||
                            location_in_grid.y() < 1 || location_in_grid.y() >= volume_size.y - 1 ||
                            location_in_grid.z() < 1 || location_in_grid.z() >= volume_size.z - 1)
                            break;

                        Vec3fda normal, shifted;

                        shifted = location_in_grid;
                        shifted.x() += 1;
                        if (shifted.x() >= volume_size.x - 1)
                            break;
                        const float Fx1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        shifted = location_in_grid;
                        shifted.x() -= 1;
                        if (shifted.x() < 1)
                            break;
                        const float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        normal.x() = (Fx1 - Fx2);

                        shifted = location_in_grid;
                        shifted.y() += 1;
                        if (shifted.y() >= volume_size.y - 1)
                            break;
                        const float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        shifted = location_in_grid;
                        shifted.y() -= 1;
                        if (shifted.y() < 1)
                            break;
                        const float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        normal.y() = (Fy1 - Fy2);

                        shifted = location_in_grid;
                        shifted.z() += 1;
                        if (shifted.z() >= volume_size.z - 1)
                            break;
                        const float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        shifted = location_in_grid;
                        shifted.z() -= 1;
                        if (shifted.z() < 1)
                            break;
                        const float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size, voxel_scale);

                        normal.z() = (Fz1 - Fz2);

                        if (normal.norm() == 0)
                            break;

                        normal.normalize();

                        model_vertex.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
                        model_normal.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());

                        auto location_in_grid_int = location_in_grid.cast<int>();
                        model_color.ptr(y)[x] = color_volume.ptr(
                                location_in_grid_int.z() * volume_size.y +
                                location_in_grid_int.y())[location_in_grid_int.x()];

                        break;
                    }
                }
            }

            void surface_prediction(const VolumeData& volume,
                                    GpuMat& model_vertex, GpuMat& model_normal, GpuMat& model_color,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const Eigen::Matrix4f& pose)
            {
                model_vertex.setTo(0);
                model_normal.setTo(0);
                model_color.setTo(0);

                dim3 threads(32, 32);
                dim3 blocks((model_vertex.cols + threads.x - 1) / threads.x,
                            (model_vertex.rows + threads.y - 1) / threads.y);

                raycast_tsdf_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.color_volume,
                        model_vertex, model_normal, model_color,
                        volume.volume_size, volume.voxel_scale,
                        cam_parameters,
                        truncation_distance,
                        pose.block(0, 0, 3, 3), pose.block(0, 3, 3, 1));

                cudaThreadSynchronize();
            }
        }
    }
}