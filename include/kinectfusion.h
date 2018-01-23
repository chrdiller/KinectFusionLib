// This is the KinectFusion Pipeline
// Author: Christian Diller, git@christian-diller.de

#ifndef KINECTFUSION_H
#define KINECTFUSION_H

#include "data_types.h"

namespace kinectfusion {
    /*
     *
     * \brief This is the KinectFusion pipeline that processes incoming frames and fuses them into one volume
     *
     * It implements the basic four steps described in the KinectFusion paper:
     * (1) Surface Measurement: Compute vertex and normal maps and their pyramids
     * (2) Pose Estimation: Use ICP with measured depth and predicted surface to localize camera
     * (3) Surface reconstruction: Integration of surface measurements into a global volume
     * (4) Surface prediction: Raycast volume in order to compute a surface prediction
     *
     * After construction, the pipeline allows you to insert new frames consisting of depth and color.
     * In the end, you can export the internal volume either as a pointcloud or a dense surface mesh.
     * You can also export the camera poses and (depending on your configuration) visualize the last model frame.
     *
     */
    class Pipeline {
    public:
        /**
         * Constructs the pipeline, sets up the interal volume and camera.
         * @param _camera_parameters The \ref{CameraParameters} that you want this pipeline to use
         * @param _configuration The \ref{GlobalConfiguration} with all parameters the pipeline should use
         */
        Pipeline(const CameraParameters _camera_parameters,
                 const GlobalConfiguration _configuration);

        ~Pipeline() = default;

        /**
         * Invoke this for every frame you want to fuse into the global volume
         * @param depth_map The depth map for the current frame. Must consist of float values representing the depth in mm
         * @param color_map The RGB color map. Must be a matrix (datatype CV_8UC3)
         * @return Whether the frame has been fused successfully. Will only be false if the ICP failed.
         */
        bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map);

        /**
         * Retrieve all camera poses computed so far
         * @return A vector for 4x4 camera poses, consisting of rotation and translation
         */
        std::vector<Eigen::Matrix4f> get_poses() const;

        /**
         * Use this to get a visualization of the last raycasting
         * @return The last (colorized) model frame from raycasting the internal volume
         */
        cv::Mat get_last_model_frame() const;

        /**
         * Extract a point cloud
         * @return A PointCloud representation (see description of PointCloud for more information on the data layout)
         */
        PointCloud extract_pointcloud() const;

        /**
         * Extract a dense surface mesh
         * @return A SurfaceMesh representation (see description of SurfaceMesh for more information on the data layout)
         */
        SurfaceMesh extract_mesh() const;

    private:
        // Internal parameters, not to be changed after instantiation
        const CameraParameters camera_parameters;
        const GlobalConfiguration configuration;

        // The global volume (containing tsdf and color)
        internal::VolumeData volume;

        // The model data for the current frame
        internal::ModelData model_data;

        // Poses: Current and all previous
        Eigen::Matrix4f current_pose;
        std::vector<Eigen::Matrix4f> poses;

        // Frame ID and raycast result for output purposes
        size_t frame_id;
        cv::Mat last_model_frame;
    };

    /**
     * Store a PointCloud instance as a PLY file.
     * If file cannot be saved, nothing will be done
     * @param filename The path and name of the file to write to; if it does not exists, it will be created and
     *                 if it exists it will be overwritten
     * @param point_cloud The PointCloud instance
     */
    void export_ply(const std::string& filename, const PointCloud& point_cloud);

    /**
     * Store a SurfaceMesh instance as a PLY file.
     * If file cannot be saved, nothing will be done
     * @param filename The path and name of the file to write to; if it does not exists, it will be created and
     *                 if it exists it will be overwritten
     * @param surface_mesh The SurfaceMesh instance
     */
    void export_ply(const std::string& filename, const SurfaceMesh& surface_mesh);


    namespace internal {

        /*
         * Step 1: SURFACE MEASUREMENT
         * Compute vertex and normal maps and their pyramids
         */
        FrameData surface_measurement(const cv::Mat_<float>& input_frame,
                                      const CameraParameters& camera_params,
                                      const size_t num_levels, const float depth_cutoff,
                                      const int kernel_size, const float color_sigma, const float spatial_sigma);


        /*
         * Step 2: POSE ESTIMATION
         * Use ICP with measured depth and predicted surface to localize camera
         */
        bool pose_estimation(Eigen::Matrix4f& pose,
                             const FrameData& frame_data,
                             const ModelData& model_data,
                             const CameraParameters& cam_params,
                             const int pyramid_height,
                             const float distance_threshold, const float angle_threshold,
                             const std::vector<int>& iterations);

        namespace cuda {

            /*
             * Step 3: SURFACE RECONSTRUCTION
             * Integration of surface measurements into a global volume
             */
            void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
                                        const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params,
                                        const float truncation_distance,
                                        const Eigen::Matrix4f& model_view);


            /*
             * Step 4: SURFACE PREDICTION
             * Raycast volume in order to compute a surface prediction
             */
            void surface_prediction(const VolumeData& volume,
                                    cv::cuda::GpuMat& model_vertex,
                                    cv::cuda::GpuMat& model_normal,
                                    cv::cuda::GpuMat& model_color,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const Eigen::Matrix4f& pose);

            PointCloud extract_points(const VolumeData& volume, const int buffer_size);

            SurfaceMesh marching_cubes(const VolumeData& volume, const int triangles_buffer_size);
        }

    }
}
#endif //KINECTFUSION_H
