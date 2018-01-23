// Estimates the current pose using ICP
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

#include "include/common.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

using Matf31da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            template<int SIZE>
            static __device__ __forceinline__
            void reduce(volatile double* buffer)
            {
                const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
                double value = buffer[thread_id];

                if (SIZE >= 1024) {
                    if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
                    __syncthreads();
                }
                if (SIZE >= 512) {
                    if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
                    __syncthreads();
                }
                if (SIZE >= 256) {
                    if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
                    __syncthreads();
                }
                if (SIZE >= 128) {
                    if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
                    __syncthreads();
                }

                if (thread_id < 32) {
                    if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
                    if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
                    if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
                    if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
                    if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
                    if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
                }
            }

            __global__
            void estimate_kernel(const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_current,
                                 const Matf31da translation_current,
                                 const PtrStep<float3> vertex_map_current, const PtrStep<float3> normal_map_current,
                                 const Eigen::Matrix<float, 3, 3, Eigen::DontAlign> rotation_previous_inv,
                                 const Matf31da translation_previous,
                                 const CameraParameters cam_params,
                                 const PtrStep<float3> vertex_map_previous, const PtrStep<float3> normal_map_previous,
                                 const float distance_threshold, const float angle_threshold, const int cols,
                                 const int rows,
                                 PtrStep<double> global_buffer)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                Matf31da n, d, s;
                bool correspondence_found = false;

                if (x < cols && y < rows) {
                    Matf31da normal_current;
                    normal_current.x() = normal_map_current.ptr(y)[x].x;

                    if (!isnan(normal_current.x())) {
                        Matf31da vertex_current;
                        vertex_current.x() = vertex_map_current.ptr(y)[x].x;
                        vertex_current.y() = vertex_map_current.ptr(y)[x].y;
                        vertex_current.z() = vertex_map_current.ptr(y)[x].z;

                        Matf31da vertex_current_global = rotation_current * vertex_current + translation_current;

                        Matf31da vertex_current_camera =
                                rotation_previous_inv * (vertex_current_global - translation_previous);

                        Eigen::Vector2i point;
                        point.x() = __float2int_rd(
                                vertex_current_camera.x() * cam_params.focal_x / vertex_current_camera.z() +
                                cam_params.principal_x + 0.5f);
                        point.y() = __float2int_rd(
                                vertex_current_camera.y() * cam_params.focal_y / vertex_current_camera.z() +
                                cam_params.principal_y + 0.5f);

                        if (point.x() >= 0 && point.y() >= 0 && point.x() < cols && point.y() < rows &&
                            vertex_current_camera.z() >= 0) {
                            Matf31da normal_previous_global;
                            normal_previous_global.x() = normal_map_previous.ptr(point.y())[point.x()].x;

                            if (!isnan(normal_previous_global.x())) {
                                Matf31da vertex_previous_global;
                                vertex_previous_global.x() = vertex_map_previous.ptr(point.y())[point.x()].x;
                                vertex_previous_global.y() = vertex_map_previous.ptr(point.y())[point.x()].y;
                                vertex_previous_global.z() = vertex_map_previous.ptr(point.y())[point.x()].z;

                                const float distance = (vertex_previous_global - vertex_current_global).norm();
                                if (distance <= distance_threshold) {
                                    normal_current.y() = normal_map_current.ptr(y)[x].y;
                                    normal_current.z() = normal_map_current.ptr(y)[x].z;

                                    Matf31da normal_current_global = rotation_current * normal_current;

                                    normal_previous_global.y() = normal_map_previous.ptr(point.y())[point.x()].y;
                                    normal_previous_global.z() = normal_map_previous.ptr(point.y())[point.x()].z;

                                    const float sine = normal_current_global.cross(normal_previous_global).norm();

                                    if (sine >= angle_threshold) {
                                        n = normal_previous_global;
                                        d = vertex_previous_global;
                                        s = vertex_current_global;

                                        correspondence_found = true;
                                    }
                                }
                            }
                        }
                    }
                }

                float row[7];

                if (correspondence_found) {
                    *(Matf31da*) &row[0] = s.cross(n);
                    *(Matf31da*) &row[3] = n;
                    row[6] = n.dot(d - s);
                } else
                    row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;

                __shared__ double smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];
                const int tid = threadIdx.y * blockDim.x + threadIdx.x;

                int shift = 0;
                for (int i = 0; i < 6; ++i) { // Rows
                    for (int j = i; j < 7; ++j) { // Columns and B
                        __syncthreads();
                        smem[tid] = row[i] * row[j];
                        __syncthreads();

                        reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(smem);

                        if (tid == 0)
                            global_buffer.ptr(shift++)[gridDim.x * blockIdx.y + blockIdx.x] = smem[0];
                    }
                }
            }

            __global__
            void reduction_kernel(PtrStep<double> global_buffer, const int length, PtrStep<double> output)
            {
                double sum = 0.0;
                for (int t = threadIdx.x; t < length; t += 512)
                    sum += *(global_buffer.ptr(blockIdx.x) + t);

                __shared__ double smem[512];

                smem[threadIdx.x] = sum;
                __syncthreads();

                reduce<512>(smem);

                if (threadIdx.x == 0)
                    output.ptr(blockIdx.x)[0] = smem[0];
            };

            void estimate_step(const Eigen::Matrix3f& rotation_current, const Matf31da& translation_current,
                               const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                               const Eigen::Matrix3f& rotation_previous_inv, const Matf31da& translation_previous,
                               const CameraParameters& cam_params,
                               const cv::cuda::GpuMat& vertex_map_previous, const cv::cuda::GpuMat& normal_map_previous,
                               float distance_threshold, float angle_threshold,
                               Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b)
            {
                const int cols = vertex_map_current.cols;
                const int rows = vertex_map_current.rows;

                dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
                dim3 grid(1, 1);
                grid.x = static_cast<unsigned int>(std::ceil(cols / block.x));
                grid.y = static_cast<unsigned int>(std::ceil(rows / block.y));

                cv::cuda::GpuMat sum_buffer { cv::cuda::createContinuous(27, 1, CV_64FC1) };
                cv::cuda::GpuMat global_buffer { cv::cuda::createContinuous(27, grid.x * grid.y, CV_64FC1) };

                estimate_kernel<<<grid, block>>>(rotation_current, translation_current,
                        vertex_map_current, normal_map_current,
                        rotation_previous_inv, translation_previous,
                        cam_params,
                        vertex_map_previous, normal_map_previous,
                        distance_threshold, angle_threshold,
                        cols, rows,
                        global_buffer);

                reduction_kernel<<<27, 512>>>(global_buffer, grid.x * grid.y, sum_buffer);

                cv::Mat host_data { 27, 1, CV_64FC1 };
                sum_buffer.download(host_data);

                int shift = 0;
                for (int i = 0; i < 6; ++i) { // Rows
                    for (int j = i; j < 7; ++j) { // Columns and B
                        double value = host_data.ptr<double>(shift++)[0];
                        if (j == 6)
                            b.data()[i] = value;
                        else
                            A.data()[j * 6 + i] = A.data()[i * 6 + j] = value;
                    }
                }
            }
        }
    }
}