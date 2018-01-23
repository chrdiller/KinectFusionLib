// This is the CPU part of the surface measurement
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#pragma GCC diagnostic pop

using cv::cuda::GpuMat;

namespace kinectfusion {
    namespace internal {

        namespace cuda { // Forward declare CUDA functions
            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);
        }

        FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                      const CameraParameters& camera_params,
                                      const size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, const float spatial_sigma)
        {
            // Initialize frame data
            FrameData data(num_levels);

            // Allocate GPU memory
            for (size_t level = 0; level < num_levels; ++level) {
                const int width = camera_params.level(level).image_width;
                const int height = camera_params.level(level).image_height;

                data.depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);
                data.smoothed_depth_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC1);

                data.color_pyramid[level] = cv::cuda::createContinuous(height, width, CV_8UC3);

                data.vertex_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
                data.normal_pyramid[level] = cv::cuda::createContinuous(height, width, CV_32FC3);
            }

            // Start by uploading original frame to GPU
            data.depth_pyramid[0].upload(input_frame);

            // Build pyramids and filter bilaterally on GPU
            cv::cuda::Stream stream;
            for (size_t level = 1; level < num_levels; ++level)
                cv::cuda::pyrDown(data.depth_pyramid[level - 1], data.depth_pyramid[level], stream);
            for (size_t level = 0; level < num_levels; ++level) {
                cv::cuda::bilateralFilter(data.depth_pyramid[level], // source
                                          data.smoothed_depth_pyramid[level], // destination
                                          kernel_size,
                                          color_sigma,
                                          spatial_sigma,
                                          cv::BORDER_DEFAULT,
                                          stream);
            }
            stream.waitForCompletion();

            // Compute vertex and normal maps
            for (size_t level = 0; level < num_levels; ++level) {
                cuda::compute_vertex_map(data.smoothed_depth_pyramid[level], data.vertex_pyramid[level],
                                         depth_cutoff, camera_params.level(level));
                cuda::compute_normal_map(data.vertex_pyramid[level], data.normal_pyramid[level]);
            }

            return data;
        }
    }
}