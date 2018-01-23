// Measures the surface, i.e. computes vertex and normal maps from a depth frame
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

using cv::cuda::GpuMat;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            __global__
            void kernel_compute_vertex_map(const PtrStepSz<float> depth_map, PtrStep<float3> vertex_map,
                                           const float depth_cutoff, const CameraParameters cam_params)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= depth_map.cols || y >= depth_map.rows)
                    return;

                float depth_value = depth_map.ptr(y)[x];
                if (depth_value > depth_cutoff) depth_value = 0.f; // Depth cutoff

                Vec3fda vertex(
                        (x - cam_params.principal_x) * depth_value / cam_params.focal_x,
                        (y - cam_params.principal_y) * depth_value / cam_params.focal_y,
                        depth_value);

                vertex_map.ptr(y)[x] = make_float3(vertex.x(), vertex.y(), vertex.z());
            }

            __global__
            void kernel_compute_normal_map(const PtrStepSz<float3> vertex_map, PtrStep<float3> normal_map)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < 1 || x >= vertex_map.cols - 1 || y < 1 || y >= vertex_map.rows - 1)
                    return;

                const Vec3fda left(&vertex_map.ptr(y)[x - 1].x);
                const Vec3fda right(&vertex_map.ptr(y)[x + 1].x);
                const Vec3fda upper(&vertex_map.ptr(y - 1)[x].x);
                const Vec3fda lower(&vertex_map.ptr(y + 1)[x].x);

                Vec3fda normal;

                if (left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
                    normal = Vec3fda(0.f, 0.f, 0.f);
                else {
                    Vec3fda hor(left.x() - right.x(), left.y() - right.y(), left.z() - right.z());
                    Vec3fda ver(upper.x() - lower.x(), upper.y() - lower.y(), upper.z() - lower.z());

                    normal = hor.cross(ver);
                    normal.normalize();

                    if (normal.z() > 0)
                        normal *= -1;
                }

                normal_map.ptr(y)[x] = make_float3(normal.x(), normal.y(), normal.z());
            }

            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params)
            {
                dim3 threads(32, 32);
                dim3 blocks((depth_map.cols + threads.x - 1) / threads.x, (depth_map.rows + threads.y - 1) / threads.y);

                kernel_compute_vertex_map << < blocks, threads >> > (depth_map, vertex_map, depth_cutoff, cam_params);

                cudaThreadSynchronize();
            }

            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map)
            {
                dim3 threads(32, 32);
                dim3 blocks((vertex_map.cols + threads.x - 1) / threads.x,
                            (vertex_map.rows + threads.y - 1) / threads.y);

                kernel_compute_normal_map<<<blocks, threads>>>(vertex_map, normal_map);

                cudaThreadSynchronize();
            }
        }
    }
}