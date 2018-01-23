// This is the KinectFusion Pipeline Implementation
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

#include <fstream>

using cv::cuda::GpuMat;

namespace kinectfusion {

    Pipeline::Pipeline(const CameraParameters _camera_parameters,
                       const GlobalConfiguration _configuration) :
            camera_parameters(_camera_parameters), configuration(_configuration),
            volume(_configuration.volume_size, _configuration.voxel_scale),
            model_data(_configuration.num_levels, _camera_parameters),
            current_pose{}, poses{}, frame_id{0}, last_model_frame{}
    {
        // The pose starts in the middle of the cube, offset along z by the initial depth
        current_pose.setIdentity();
        current_pose(0, 3) = _configuration.volume_size.x / 2 * _configuration.voxel_scale;
        current_pose(1, 3) = _configuration.volume_size.y / 2 * _configuration.voxel_scale;
        current_pose(2, 3) = _configuration.volume_size.z / 2 * _configuration.voxel_scale - _configuration.init_depth;
    }

    bool Pipeline::process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map)
    {
        // STEP 1: Surface measurement
        internal::FrameData frame_data = internal::surface_measurement(depth_map, camera_parameters,
                                                                       configuration.num_levels,
                                                                       configuration.depth_cutoff_distance,
                                                                       configuration.bfilter_kernel_size,
                                                                       configuration.bfilter_color_sigma,
                                                                       configuration.bfilter_spatial_sigma);
        frame_data.color_pyramid[0].upload(color_map);

        // STEP 2: Pose estimation
        bool icp_success { true };
        if (frame_id > 0) { // Do not perform ICP for the very first frame
            icp_success = internal::pose_estimation(current_pose, frame_data, model_data, camera_parameters,
                                                    configuration.num_levels,
                                                    configuration.distance_threshold, configuration.angle_threshold,
                                                    configuration.icp_iterations);
        }
        if (!icp_success)
            return false;

        poses.push_back(current_pose);

        // STEP 3: Surface reconstruction
        internal::cuda::surface_reconstruction(frame_data.depth_pyramid[0], frame_data.color_pyramid[0],
                                               volume, camera_parameters, configuration.truncation_distance,
                                               current_pose.inverse());

        // Step 4: Surface prediction
        for (int level = 0; level < configuration.num_levels; ++level)
            internal::cuda::surface_prediction(volume, model_data.vertex_pyramid[level],
                                               model_data.normal_pyramid[level],
                                               model_data.color_pyramid[level],
                                               camera_parameters.level(level), configuration.truncation_distance,
                                               current_pose);

        if (configuration.use_output_frame) // Not using the output will speed up the processing
            model_data.color_pyramid[0].download(last_model_frame);

        ++frame_id;

        return true;
    }

    cv::Mat Pipeline::get_last_model_frame() const
    {
        if (configuration.use_output_frame)
            return last_model_frame;

        return cv::Mat(1, 1, CV_8UC1);
    }

    std::vector<Eigen::Matrix4f> Pipeline::get_poses() const
    {
        for (auto pose : poses)
            pose.block(0, 0, 3, 3) = pose.block(0, 0, 3, 3).inverse();
        return poses;
    }

    PointCloud Pipeline::extract_pointcloud() const
    {
        PointCloud cloud_data = internal::cuda::extract_points(volume, configuration.pointcloud_buffer_size);
        return cloud_data;
    }

    SurfaceMesh Pipeline::extract_mesh() const
    {
        SurfaceMesh surface_mesh = internal::cuda::marching_cubes(volume, configuration.triangles_buffer_size);
        return surface_mesh;
    }

    void export_ply(const std::string& filename, const PointCloud& point_cloud)
    {
        std::ofstream file_out { filename };
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << point_cloud.num_points << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property float nx" << std::endl;
        file_out << "property float ny" << std::endl;
        file_out << "property float nz" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "end_header" << std::endl;

        for (int i = 0; i < point_cloud.num_points; ++i) {
            float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
            float3 normal = point_cloud.normals.ptr<float3>(0)[i];
            uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                     << normal.z << " ";
            file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                     << static_cast<int>(color.z) << std::endl;
        }
    }

    void export_ply(const std::string& filename, const SurfaceMesh& surface_mesh)
    {
        std::ofstream file_out { filename };
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << surface_mesh.num_vertices << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "element face " << surface_mesh.num_triangles << std::endl;
        file_out << "property list uchar int vertex_index" << std::endl;
        file_out << "end_header" << std::endl;

        for (int v_idx = 0; v_idx < surface_mesh.num_vertices; ++v_idx) {
            float3 vertex = surface_mesh.triangles.ptr<float3>(0)[v_idx];
            uchar3 color = surface_mesh.colors.ptr<uchar3>(0)[v_idx];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " ";
            file_out << (int) color.z << " " << (int) color.y << " " << (int) color.x << std::endl;
        }

        for (int t_idx = 0; t_idx < surface_mesh.num_vertices; t_idx += 3) {
            file_out << 3 << " " << t_idx + 1 << " " << t_idx << " " << t_idx + 2 << std::endl;
        }
    }
}