// Extracts a surface mesh from the internal volume using the Marching Cubes algorithm
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

// Thrust, for prefix scanning
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// Internals
#include "include/common.h"
#include "include/mc_tables.h"

using cv::cuda::GpuMat;

namespace kinectfusion {
    namespace internal {
        namespace cuda {

            __device__ int global_count = 0;
            __device__ int output_count;
            __device__ unsigned int blocks_done = 0;

            //##### HELPERS #####
            static __device__ __forceinline__
            unsigned int lane_ID()
            {
                unsigned int ret;
                asm("mov.u32 %0, %laneid;" : "=r"(ret));
                return ret;
            }

            static __device__ __forceinline__
            int laneMaskLt()
            {
                unsigned int ret;
                asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret));
                return ret;
            }

            static __device__ __forceinline__
            int binaryExclScan(int ballot_mask)
            {
                return __popc(laneMaskLt() & ballot_mask);
            }

            __device__ __forceinline__
            float read_tsdf(const PtrStep<short2> tsdf_volume, const int3 volume_size,
                            const int x, const int y, const int z, short& weight)
            {
                short2 voxel_tuple = tsdf_volume.ptr(z * volume_size.y + y)[x];
                weight = voxel_tuple.y;
                return static_cast<float>(voxel_tuple.x) * DIVSHORTMAX;
            }

            __device__ __forceinline__
            int compute_cube_index(const PtrStep<short2> tsdf_volume, const int3 volume_size,
                                   const int x, const int y, const int z, float tsdf_values[8])
            {
                short weight;
                int cube_index = 0; // calculate flag indicating if each vertex is inside or outside isosurface

                cube_index += static_cast<int>(tsdf_values[0] = read_tsdf(tsdf_volume, volume_size, x, y, z, weight) < 0.f);
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[1] = read_tsdf(tsdf_volume, volume_size, x + 1, y, z, weight) < 0.f) << 1;
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[2] = read_tsdf(tsdf_volume, volume_size, x + 1, y + 1, z, weight) < 0.f) << 2;
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[3] = read_tsdf(tsdf_volume, volume_size, x, y + 1, z, weight) < 0.f) << 3;
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[4] = read_tsdf(tsdf_volume, volume_size, x, y, z + 1, weight) < 0.f) << 4;
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[5] = read_tsdf(tsdf_volume, volume_size, x + 1, y, z + 1, weight) < 0.f) << 5;
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[6] = read_tsdf(tsdf_volume, volume_size, x + 1, y + 1, z + 1, weight) < 0.f) << 6;
                if (weight == 0) return 0;
                cube_index += static_cast<int>(tsdf_values[7] = read_tsdf(tsdf_volume, volume_size, x, y + 1, z + 1, weight) < 0.f) << 7;
                if (weight == 0) return 0;

                return cube_index;
            }

            __device__ __forceinline__
            float3 get_node_coordinates(const int x, const int y, const int z, const float voxel_size)
            {
                float3 position;

                position.x = (x + 0.5f) * voxel_size;
                position.y = (y + 0.5f) * voxel_size;
                position.z = (z + 0.5f) * voxel_size;

                return position;
            }

            __device__ __forceinline__
            float3 vertex_interpolate(const float3 p0, const float3 p1, const float f0, const float f1)
            {
                float t = (0.f - f0) / (f1 - f0 + 1e-15f);
                return make_float3(p0.x + t * (p1.x - p0.x),
                                   p0.y + t * (p1.y - p0.y),
                                   p0.z + t * (p1.z - p0.z));
            }

            //##### KERNELS #####
            __global__
            void get_occupied_voxels_kernel(const PtrStep<short2> volume, const int3 volume_size,
                                            PtrStepSz<int> occupied_voxel_indices, PtrStepSz<int> number_vertices,
                                            const PtrStepSz<int> number_vertices_table)
            {
                const int x = threadIdx.x + blockIdx.x * blockDim.x;
                const int y = threadIdx.y + blockIdx.y * blockDim.y;

                if (__all(x >= volume_size.x) || __all(y >= volume_size.y))
                    return;

                const auto flattened_tid =
                        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                const auto warp_id = flattened_tid >> 5;
                const auto lane_id = lane_ID();

                volatile __shared__ int warps_buffer[32]; // Number of threads / Warp size

                for (int z = 0; z < volume_size.z - 1; ++z) {
                    int n_vertices = 0;
                    if (x + 1 < volume_size.x && y + 1 < volume_size.y) {
                        float tsdf_values[8];
                        const int cube_index = compute_cube_index(volume, volume_size, x, y, z, tsdf_values);
                        n_vertices = (cube_index == 0 || cube_index == 255) ? 0 : number_vertices_table.ptr(0)[cube_index];
                    }

                    const int total = __popc(__ballot(n_vertices > 0));

                    if (total == 0)
                        continue;

                    if (lane_id == 0) {
                        const int old = atomicAdd(&global_count, total);
                        warps_buffer[warp_id] = old;
                    }

                    const int old_global_voxels_count = warps_buffer[warp_id];

                    const int offset = binaryExclScan(__ballot(n_vertices > 0));

                    const int max_size = occupied_voxel_indices.cols;
                    if (old_global_voxels_count + offset < max_size && n_vertices > 0) {
                        const int current_voxel_index = volume_size.y * volume_size.x * z + volume_size.x * y + x;
                        occupied_voxel_indices.ptr(0)[old_global_voxels_count + offset] = current_voxel_index;
                        number_vertices.ptr(0)[old_global_voxels_count + offset] = n_vertices;
                    }

                    bool full = old_global_voxels_count + total >= max_size;

                    if (full)
                        break;
                }

                if (flattened_tid == 0) {
                    unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
                    unsigned int value = atomicInc(&blocks_done, total_blocks);

                    if (value == total_blocks - 1) {
                        output_count = min(occupied_voxel_indices.cols, global_count);
                        blocks_done = 0;
                        global_count = 0;
                    }
                }
            }

            __global__
            void generate_triangles_kernel(const PtrStep<short2> tsdf_volume, const int3 volume_size, const float voxel_size,
                                           const PtrStepSz<int> occupied_voxels, const PtrStepSz<int> vertex_offsets,
                                           const PtrStep<int> number_vertices_table, const PtrStep<int> triangle_table,
                                           PtrStep<float3> triangle_buffer)
            {
                const int idx = (blockIdx.y * 65536 + blockIdx.x) * 256 + threadIdx.x;

                if (idx >= occupied_voxels.cols)
                    return;

                const int voxel = occupied_voxels.ptr(0)[idx];

                const int z = voxel / (volume_size.x * volume_size.y);
                const int y = (voxel - z * volume_size.x * volume_size.y) / volume_size.x;
                const int x = (voxel - z * volume_size.x * volume_size.y) - y * volume_size.x;

                float tsdf_values[8];
                const int cube_index = compute_cube_index(tsdf_volume, volume_size, x, y, z, tsdf_values);

                float3 v[8];
                v[0] = get_node_coordinates(x, y, z, voxel_size);
                v[1] = get_node_coordinates(x + 1, y, z, voxel_size);
                v[2] = get_node_coordinates(x + 1, y + 1, z, voxel_size);
                v[3] = get_node_coordinates(x, y + 1, z, voxel_size);
                v[4] = get_node_coordinates(x, y, z + 1, voxel_size);
                v[5] = get_node_coordinates(x + 1, y, z + 1, voxel_size);
                v[6] = get_node_coordinates(x + 1, y + 1, z + 1, voxel_size);
                v[7] = get_node_coordinates(x, y + 1, z + 1, voxel_size);

                __shared__ float3 vertex_list[12][256];
                vertex_list[0][threadIdx.x] = vertex_interpolate(v[0], v[1], tsdf_values[0], tsdf_values[1]);
                vertex_list[1][threadIdx.x] = vertex_interpolate(v[1], v[2], tsdf_values[1], tsdf_values[2]);
                vertex_list[2][threadIdx.x] = vertex_interpolate(v[2], v[3], tsdf_values[2], tsdf_values[3]);
                vertex_list[3][threadIdx.x] = vertex_interpolate(v[3], v[0], tsdf_values[3], tsdf_values[0]);
                vertex_list[4][threadIdx.x] = vertex_interpolate(v[4], v[5], tsdf_values[4], tsdf_values[5]);
                vertex_list[5][threadIdx.x] = vertex_interpolate(v[5], v[6], tsdf_values[5], tsdf_values[6]);
                vertex_list[6][threadIdx.x] = vertex_interpolate(v[6], v[7], tsdf_values[6], tsdf_values[7]);
                vertex_list[7][threadIdx.x] = vertex_interpolate(v[7], v[4], tsdf_values[7], tsdf_values[4]);
                vertex_list[8][threadIdx.x] = vertex_interpolate(v[0], v[4], tsdf_values[0], tsdf_values[4]);
                vertex_list[9][threadIdx.x] = vertex_interpolate(v[1], v[5], tsdf_values[1], tsdf_values[5]);
                vertex_list[10][threadIdx.x] = vertex_interpolate(v[2], v[6], tsdf_values[2], tsdf_values[6]);
                vertex_list[11][threadIdx.x] = vertex_interpolate(v[3], v[7], tsdf_values[3], tsdf_values[7]);
                __syncthreads();

                const int n_vertices = number_vertices_table.ptr(0)[cube_index];

                for (int i = 0; i < n_vertices; i += 3) {
                    const int index = vertex_offsets.ptr(0)[idx] + i;

                    const int v1 = triangle_table.ptr(0)[(cube_index * 16) + i + 0];
                    const int v2 = triangle_table.ptr(0)[(cube_index * 16) + i + 1];
                    const int v3 = triangle_table.ptr(0)[(cube_index * 16) + i + 2];

                    triangle_buffer.ptr(0)[index + 0] = make_float3(vertex_list[v1][threadIdx.x].x,
                                                                    vertex_list[v1][threadIdx.x].y,
                                                                    vertex_list[v1][threadIdx.x].z);
                    triangle_buffer.ptr(0)[index + 1] = make_float3(vertex_list[v2][threadIdx.x].x,
                                                                    vertex_list[v2][threadIdx.x].y,
                                                                    vertex_list[v2][threadIdx.x].z);
                    triangle_buffer.ptr(0)[index + 2] = make_float3(vertex_list[v3][threadIdx.x].x,
                                                                    vertex_list[v3][threadIdx.x].y,
                                                                    vertex_list[v3][threadIdx.x].z);
                }
            }


            __global__
            void get_color_values_kernel(const PtrStep<uchar3> color_volume, const int3 volume_size, const float voxel_scale,
                                         const PtrStep<float3> vertices, PtrStepSz<uchar3> vertex_colors)
            {
                const auto thread_id = blockDim.x * blockIdx.x + threadIdx.x;

                if (thread_id >= vertex_colors.cols)
                    return;

                const float3 vertex = vertices.ptr(0)[thread_id];
                const int3 location_in_grid{static_cast<int>(vertex.x / voxel_scale),
                                            static_cast<int>(vertex.y / voxel_scale),
                                            static_cast<int>(vertex.z / voxel_scale)};

                uchar3 color_value = color_volume.ptr(
                        location_in_grid.z * volume_size.y + location_in_grid.y)[location_in_grid.x];

                vertex_colors.ptr(0)[thread_id] = color_value;
            }


            //##### HOST FUNCTIONS #####
            SurfaceMesh marching_cubes(const VolumeData& volume, const int triangles_buffer_size)
            {
                MeshData mesh_data(triangles_buffer_size / 3);

                // ### PREPARATION : Upload lookup tables ###
                GpuMat number_vertices_table, triangle_table;

                number_vertices_table = cv::cuda::createContinuous(256, 1, CV_32SC1);
                number_vertices_table.upload(cv::Mat(256, 1, CV_32SC1, number_vertices_table_host, cv::Mat::AUTO_STEP));

                triangle_table = cv::cuda::createContinuous(256, 16, CV_32SC1);
                triangle_table.upload(cv::Mat(256, 16, CV_32SC1, triangle_table_host, cv::Mat::AUTO_STEP));
                // ### ###

                //### KERNEL ONE : Get occupied voxels ###
                dim3 threads(32, 32);
                dim3 blocks(static_cast<unsigned>(std::ceil(volume.volume_size.x / threads.x)),
                            static_cast<unsigned>(std::ceil(volume.volume_size.y / threads.y)));

                get_occupied_voxels_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.volume_size,
                        mesh_data.occupied_voxel_ids_buffer, mesh_data.number_vertices_buffer,
                        number_vertices_table);

                cudaDeviceSynchronize();

                int active_voxels = 0;
                cudaMemcpyFromSymbol(&active_voxels, output_count, sizeof(active_voxels));
                // ### ###

                //### THRUST PART : Do an exclusive scan on the GPU ###
                mesh_data.create_view(active_voxels);

                thrust::device_ptr<int> beg = thrust::device_pointer_cast(mesh_data.number_vertices.ptr<int>(0));
                thrust::device_ptr<int> end = beg + active_voxels;

                thrust::device_ptr<int> out = thrust::device_pointer_cast(mesh_data.vertex_offsets.ptr<int>(0));
                thrust::exclusive_scan(beg, end, out);

                int last_element, last_scan_element;

                cudaMemcpy(&last_element, mesh_data.number_vertices.ptr<int>(0) + active_voxels - 1, sizeof(int),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(&last_scan_element, mesh_data.vertex_offsets.ptr<int>(0) + active_voxels - 1, sizeof(int),
                           cudaMemcpyDeviceToHost);

                const int total_vertices = last_element + last_scan_element;
                // ### ###

                //### KERNEL TWO ###
                const int n_threads = 256;
                dim3 block(n_threads);
                unsigned blocks_num = static_cast<unsigned>(std::ceil(active_voxels / n_threads));
                dim3 grid(min(blocks_num, 65536), static_cast<unsigned>(std::ceil(blocks_num / 65536)));
                grid.y = 1;

                generate_triangles_kernel<<<grid, block>>> (volume.tsdf_volume,
                        volume.volume_size, volume.voxel_scale,
                        mesh_data.occupied_voxel_ids, mesh_data.vertex_offsets,
                        number_vertices_table, triangle_table,
                        mesh_data.triangle_buffer);

                cudaDeviceSynchronize();
                // ### ###

                // Get triangle vertex colors
                GpuMat triangles_output(mesh_data.triangle_buffer, cv::Range::all(), cv::Range(0, total_vertices));
                GpuMat vertex_colors = cv::cuda::createContinuous(1, total_vertices, CV_8UC3);

                int n_blocks = static_cast<int>(std::ceil(total_vertices / 1024));
                get_color_values_kernel<<<n_blocks, 1024>>> (volume.color_volume,
                        volume.volume_size, volume.voxel_scale,
                        triangles_output, vertex_colors);

                cudaDeviceSynchronize();

                // Download triangles
                cv::Mat vertex_output {};
                triangles_output.download(vertex_output);
                cv::Mat color_output {};
                vertex_colors.download(color_output);

                return SurfaceMesh { vertex_output, color_output, total_vertices, total_vertices / 3 };
            }
        }
    }
}