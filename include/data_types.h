// This header contains globally used data types
// Namespace kinectfusion contains those meant for external use
// Internal structures are located in kinectfusion::internal
// Author: Christian Diller, git@christian-diller.de

#ifndef KINECTFUSION_DATA_TYPES_H
#define KINECTFUSION_DATA_TYPES_H

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#pragma GCC diagnostic pop
#else
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#endif

using cv::cuda::GpuMat;

namespace kinectfusion {
    /**
     *
     * \brief The camera intrinsics
     *
     * This structure stores the intrinsic camera parameters.
     *
     * Consists of:
     * 1) Image width and height,
     * 2) focal length in x and y direction and
     * 3) The principal point in x and y direction
     *
     */
    struct CameraParameters {
        int image_width, image_height;
        float focal_x, focal_y;
        float principal_x, principal_y;

        /**
         * Returns camera parameters for a specified pyramid level; each level corresponds to a scaling of pow(.5, level)
         * @param level The pyramid level to get the parameters for with 0 being the non-scaled version,
         * higher levels correspond to smaller spatial size
         * @return A CameraParameters structure containing the scaled values
         */
        CameraParameters level(const size_t level) const
        {
            if (level == 0) return *this;

            const float scale_factor = powf(0.5f, static_cast<float>(level));
            return CameraParameters { image_width >> level, image_height >> level,
                                      focal_x * scale_factor, focal_y * scale_factor,
                                      (principal_x + 0.5f) * scale_factor - 0.5f,
                                      (principal_y + 0.5f) * scale_factor - 0.5f };
        }
    };

    /**
     *
     * \brief Representation of a cloud of three-dimensional points (vertices)
     *
     * This data structure contains
     * (1) the world coordinates of the vertices,
     * (2) the corresponding normals and
     * (3) the corresponding RGB color value
     *
     * - vertices: A 1 x buffer_size opencv matrix with CV_32FC3 values, representing the coordinates
     * - normals: A 1 x buffer_size opencv matrix with CV_32FC3 values, representing the normal direction
     * - color: A 1 x buffer_size opencv matrix with CV_8UC3 values, representing the RGB color
     *
     * Same indices represent the same point
     *
     * The total number of valid points in those buffers is stored in num_points
     *
     */
    struct PointCloud {
        // World coordinates of all vertices
        cv::Mat vertices;
        // Normal directions
        cv::Mat normals;
        // RGB color values
        cv::Mat color;

        // Total number of valid points
        int num_points;
    };

    /**
     *
     * \brief Representation of a dense surface mesh
     *
     * This data structure contains
     * (1) the mesh triangles (triangular faces) and
     * (2) the colors of the corresponding vertices
     *
     * - triangles: A 1 x num_vertices opencv matrix with CV_32FC3 values, representing the coordinates of one vertex;
     *              a sequence of three vertices represents one triangle
     * - colors: A 1 x num_vertices opencv matrix with CV_8Uc3 values, representing the RGB color of each vertex
     *
     * Same indices represent the same point
     *
     * Total number of vertices stored in num_vertices, total number of triangles in num_triangles
     *
     */
    struct SurfaceMesh {
        // Triangular faces
        cv::Mat triangles;
        // Colors of the vertices
        cv::Mat colors;

        // Total number of vertices
        int num_vertices;
        // Total number of triangles
        int num_triangles;
    };

    /**
     *
     * \brief The global configuration
     *
     * This data structure contains several parameters that control the workflow of the overall pipeline.
     * Most of them are based on the KinectFusion paper
     *
     * For an explanation of a specific parameter, see the corresponding comment
     *
     * The struct is preset with some default values so that you can use the configuration without modification.
     * However, you will probably want to adjust most of them to your specific use case.
     *
     * Spatial parameters are always represented in millimeters (mm).
     *
     */
    struct GlobalConfiguration {
        // The overall size of the volume (in mm). Will be allocated on the GPU and is thus limited by the amount of
        // storage you have available. Dimensions are (x, y, z).
        int3 volume_size { make_int3(512, 512, 512) };

        // The amount of mm one single voxel will represent in each dimension. Controls the resolution of the volume.
        float voxel_scale { 2.f };

        // Parameters for the Bilateral Filter, applied to incoming depth frames.
        // Directly passed to cv::cuda::bilateralFilter(...); for further information, have a look at the opencv docs.
        int bfilter_kernel_size { 5 };
        float bfilter_color_sigma { 1.f };
        float bfilter_spatial_sigma { 1.f };

        // The initial distance of the camera from the volume center along the z-axis (in mm)
        float init_depth { 1000.f };

        // Downloads the model frame for each frame (for visualization purposes). If this is set to true, you can
        // retrieve the frame with Pipeline::get_last_model_frame()
        bool use_output_frame = { true };

        // The truncation distance for both updating and raycasting the TSDF volume
        float truncation_distance { 25.f };

        // The distance (in mm) after which to set the depth in incoming depth frames to 0.
        // Can be used to separate an object you want to scan from the background
        float depth_cutoff_distance { 1000.f };

        // The number of pyramid levels to generate for each frame, including the original frame level
        int num_levels { 3 };

        // The maximum buffer size for exporting triangles; adjust if you run out of memory when exporting
        int triangles_buffer_size { 3 * 2000000 };
        // The maximum buffer size for exporting pointclouds; adjust if you run out of memory when exporting
        int pointcloud_buffer_size { 3 * 2000000 };

        // ICP configuration
        // The distance threshold (as described in the paper) in mm
        float distance_threshold { 10.f };
        // The angle threshold (as described in the paper) in degrees
        float angle_threshold { 20.f };
        // Number of ICP iterations for each level from original level 0 to highest scaled level (sparse to coarse)
        std::vector<int> icp_iterations {10, 5, 4};
    };


    namespace internal {
        /*
         * Contains the internal data representation of one single frame as read by the depth camera
         * Consists of depth, smoothed depth and color pyramids as well as vertex and normal pyramids
         */
        struct FrameData {
            std::vector<GpuMat> depth_pyramid;
            std::vector<GpuMat> smoothed_depth_pyramid;
            std::vector<GpuMat> color_pyramid;

            std::vector<GpuMat> vertex_pyramid;
            std::vector<GpuMat> normal_pyramid;

            explicit FrameData(const size_t pyramid_height) :
                    depth_pyramid(pyramid_height), smoothed_depth_pyramid(pyramid_height),
                    color_pyramid(pyramid_height), vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height)
            { }

            // No copying
            FrameData(const FrameData&) = delete;
            FrameData& operator=(const FrameData& other) = delete;

            FrameData(FrameData&& data) noexcept :
                    depth_pyramid(std::move(data.depth_pyramid)),
                    smoothed_depth_pyramid(std::move(data.smoothed_depth_pyramid)),
                    color_pyramid(std::move(data.color_pyramid)),
                    vertex_pyramid(std::move(data.vertex_pyramid)),
                    normal_pyramid(std::move(data.normal_pyramid))
            { }

            FrameData& operator=(FrameData&& data) noexcept
            {
                depth_pyramid = std::move(data.depth_pyramid);
                smoothed_depth_pyramid = std::move(data.smoothed_depth_pyramid);
                color_pyramid = std::move(data.color_pyramid);
                vertex_pyramid = std::move(data.vertex_pyramid);
                normal_pyramid = std::move(data.normal_pyramid);
                return *this;
            }
        };

        /*
         * Contains the internal data representation of one single frame as raycast by surface prediction
         * Consists of depth, smoothed depth and color pyramids as well as vertex and normal pyramids
         */
        struct ModelData {
            std::vector<GpuMat> vertex_pyramid;
            std::vector<GpuMat> normal_pyramid;
            std::vector<GpuMat> color_pyramid;

            ModelData(const size_t pyramid_height, const CameraParameters camera_parameters) :
                    vertex_pyramid(pyramid_height), normal_pyramid(pyramid_height),
                    color_pyramid(pyramid_height)
            {
                for (size_t level = 0; level < pyramid_height; ++level) {
                    vertex_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_32FC3);
                    normal_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_32FC3);
                    color_pyramid[level] =
                            cv::cuda::createContinuous(camera_parameters.level(level).image_height,
                                                       camera_parameters.level(level).image_width,
                                                       CV_8UC3);
                    vertex_pyramid[level].setTo(0);
                    normal_pyramid[level].setTo(0);
                }
            }

            // No copying
            ModelData(const ModelData&) = delete;
            ModelData& operator=(const ModelData& data) = delete;

            ModelData(ModelData&& data) noexcept :
                    vertex_pyramid(std::move(data.vertex_pyramid)),
                    normal_pyramid(std::move(data.normal_pyramid)),
                    color_pyramid(std::move(data.color_pyramid))
            { }

            ModelData& operator=(ModelData&& data) noexcept
            {
                vertex_pyramid = std::move(data.vertex_pyramid);
                normal_pyramid = std::move(data.normal_pyramid);
                color_pyramid = std::move(data.color_pyramid);
                return *this;
            }
        };

        /*
         *
         * \brief Contains the internal volume representation
         *
         * This internal representation contains two volumes:
         * (1) TSDF volume: The global volume used for depth frame fusion and
         * (2) Color volume: Simple color averaging for colorized vertex output
         *
         * It also contains two important parameters:
         * (1) Volume size: The x, y and z dimensions of the volume (in mm)
         * (2) Voxel scale: The scale of a single voxel (in mm)
         *
         */
        struct VolumeData {
            GpuMat tsdf_volume; //short2
            GpuMat color_volume; //uchar4
            int3 volume_size;
            float voxel_scale;

            VolumeData(const int3 _volume_size, const float _voxel_scale) :
                    tsdf_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC2)),
                    color_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_8UC3)),
                    volume_size(_volume_size), voxel_scale(_voxel_scale)
            {
                tsdf_volume.setTo(0);
                color_volume.setTo(0);
            }
        };

        /*
         *
         * \brief Contains the internal pointcloud representation
         *
         * This is only used for exporting the data kept in the internal volumes
         *
         * It holds GPU containers for vertices, normals and vertex colors
         * It also contains host containers for this data and defines the total number of points
         *
         */
        struct CloudData {
            GpuMat vertices;
            GpuMat normals;
            GpuMat color;

            cv::Mat host_vertices;
            cv::Mat host_normals;
            cv::Mat host_color;

            int* point_num;
            int host_point_num;

            explicit CloudData(const int max_number) :
                    vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
                    host_vertices{}, host_normals{}, host_color{}, point_num{nullptr}, host_point_num{}
            {
                vertices.setTo(0.f);
                normals.setTo(0.f);
                color.setTo(0.f);

                cudaMalloc(&point_num, sizeof(int));
                cudaMemset(point_num, 0, sizeof(int));
            }

            // No copying
            CloudData(const CloudData&) = delete;
            CloudData& operator=(const CloudData& data) = delete;

            void download()
            {
                vertices.download(host_vertices);
                normals.download(host_normals);
                color.download(host_color);

                cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);
            }
        };

        /*
         *
         * \brief Contains the internal surface mesh representation
         *
         * This is only used for exporting the data kept in the internal volumes
         *
         * It holds several GPU containers needed for the MarchingCubes algorithm
         *
         */
        struct MeshData {
            GpuMat occupied_voxel_ids_buffer;
            GpuMat number_vertices_buffer;
            GpuMat vertex_offsets_buffer;
            GpuMat triangle_buffer;

            GpuMat occupied_voxel_ids;
            GpuMat number_vertices;
            GpuMat vertex_offsets;

            explicit MeshData(const int buffer_size):
                    occupied_voxel_ids_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
                    number_vertices_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
                    vertex_offsets_buffer{cv::cuda::createContinuous(1, buffer_size, CV_32SC1)},
                    triangle_buffer{cv::cuda::createContinuous(1, buffer_size * 3, CV_32FC3)},
                    occupied_voxel_ids{}, number_vertices{}, vertex_offsets{}
            { }

            void create_view(const int length)
            {
                occupied_voxel_ids = GpuMat(1, length, CV_32SC1, occupied_voxel_ids_buffer.ptr<int>(0),
                                            occupied_voxel_ids_buffer.step);
                number_vertices = GpuMat(1, length, CV_32SC1, number_vertices_buffer.ptr<int>(0),
                                         number_vertices_buffer.step);
                vertex_offsets = GpuMat(1, length, CV_32SC1, vertex_offsets_buffer.ptr<int>(0),
                                        vertex_offsets_buffer.step);
            }
        };
    }
}

#endif //KINECTFUSION_DATA_TYPES_H
