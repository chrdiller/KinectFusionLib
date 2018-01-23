// This is the CPU part of the ICP implementation
// Author: Christian Diller, git@christian-diller.de

#include <kinectfusion.h>

using Matf31da = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
using Matrix3frm = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

namespace kinectfusion {
    namespace internal {

        namespace cuda { // Forward declare CUDA functions
            void estimate_step(const Eigen::Matrix3f& rotation_current, const Matf31da& translation_current,
                               const cv::cuda::GpuMat& vertex_map_current, const cv::cuda::GpuMat& normal_map_current,
                               const Eigen::Matrix3f& rotation_previous_inv, const Matf31da& translation_previous,
                               const CameraParameters& cam_params,
                               const cv::cuda::GpuMat& vertex_map_previous, const cv::cuda::GpuMat& normal_map_previous,
                               float distance_threshold, float angle_threshold,
                               Eigen::Matrix<double, 6, 6, Eigen::RowMajor>& A, Eigen::Matrix<double, 6, 1>& b);
        }

        bool pose_estimation(Eigen::Matrix4f& pose,
                             const FrameData& frame_data,
                             const ModelData& model_data,
                             const CameraParameters& cam_params,
                             const int pyramid_height,
                             const float distance_threshold, const float angle_threshold,
                             const std::vector<int>& iterations)
        {
            // Get initial rotation and translation
            Eigen::Matrix3f current_global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3f current_global_translation = pose.block(0, 3, 3, 1);

            Eigen::Matrix3f previous_global_rotation_inverse(current_global_rotation.inverse());
            Eigen::Vector3f previous_global_translation = pose.block(0, 3, 3, 1);

            // ICP loop, from coarse to sparse
            for (int level = pyramid_height - 1; level >= 0; --level) {
                for (int iteration = 0; iteration < iterations[level]; ++iteration) {
                    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A {};
                    Eigen::Matrix<double, 6, 1> b {};

                    // Estimate one step on the CPU
                    cuda::estimate_step(current_global_rotation, current_global_translation,
                                        frame_data.vertex_pyramid[level], frame_data.normal_pyramid[level],
                                        previous_global_rotation_inverse, previous_global_translation,
                                        cam_params.level(level),
                                        model_data.vertex_pyramid[level], model_data.normal_pyramid[level],
                                        distance_threshold, sinf(angle_threshold * 3.14159254f / 180.f),
                                        A, b);

                    // Solve equation to get alpha, beta and gamma
                    double det = A.determinant();
                    if (fabs(det) < 100000 /*1e-15*/ || std::isnan(det))
                        return false;
                    Eigen::Matrix<float, 6, 1> result { A.fullPivLu().solve(b).cast<float>() };
                    float alpha = result(0);
                    float beta = result(1);
                    float gamma = result(2);

                    // Update rotation
                    auto camera_rotation_incremental(
                            Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) *
                            Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()));
                    auto camera_translation_incremental = result.tail<3>();

                    // Update translation
                    current_global_translation =
                            camera_rotation_incremental * current_global_translation + camera_translation_incremental;
                    current_global_rotation = camera_rotation_incremental * current_global_rotation;
                }
            }

            // Return the new pose
            pose.block(0, 0, 3, 3) = current_global_rotation;
            pose.block(0, 3, 3, 1) = current_global_translation;

            return true;
        }
    }
}