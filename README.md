KinectFusion
============

This is an implementation of KinectFusion, based on _Newcombe, Richard A., et al._
**KinectFusion: Real-time dense surface mapping and tracking.**
It makes heavy use of graphics hardware and thus allows for real-time fusion of
depth image scans. Furthermore, exporting of the resulting fused volume is possible either as a pointcloud or a dense surface mesh.

Features
--------
* Real-time fusion of depth scans and corresponding RGB color images
* Easy to use, modern C++14 interface
* Export of the resulting volume as pointcloud
* Export also as dense surface mesh using the MarchingCubes algorithm
* Functions for easy export of pointclouds and meshes into the PLY file format
* Retrieval of calculated camera poses for further processing

Dependencies
------------
* **GCC 5** as higher versions do not work with current nvcc (as of 2017).
* **CUDA 8.0**. In order to provide real-time reconstruction, this library relies on graphics hardware.
Running it exclusively on the CPU is not possible.
* **OpenCV 3.0** or higher. This library heavily depends on the GPU features of OpenCV that have been refactored in the 3.0 release.
Therefore, OpenCV 2 is not supported.
* **Eigen3** for efficient matrix and vector operations.

Prerequisites
-------------
* Adjust CUDA architecture: Set the CUDA architecture version to that of your graphics hardware
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_52,code=sm_52)
Tested with a nVidia GeForce 970, compute capability 5.2, Maxwell architecture
* Set custom opencv path (if necessary):
SET("OpenCV_DIR" "/opt/opencv/usr/local/share/OpenCV")

Usage
-----
```Cpp
#include <kinectfusion.h>

// Define the data source
XtionCamera camera {};

// Get a global configuration (comes with default values) and adjust some parameters
kinectfusion::GlobalConfiguration configuration;
configuration.voxel_scale = 2.f;
configuration.init_depth = 700.f;
configuration.distance_threshold = 10.f;
configuration.angle_threshold = 20.f;

// Create a KinectFusion pipeline with the camera intrinsics and the global configuration
kinectfusion::Pipeline pipeline { camera.get_parameters(), configuration };

// Then, just loop over the incoming frames
while ( !end ) {
    // 1) Grab a frame from the data source
    InputFrame frame = camera.grab_frame();

    // 2) Have the pipeline fuse it into the global volume
    bool success = pipeline.process_frame(frame.depth_map, frame.color_map);
    if (!success)
        std::cout << "Frame could not be processed" << std::endl;
}

// Retrieve camera poses
auto poses = pipeline.get_poses();

// Export surface mesh
auto mesh = pipeline.extract_mesh();
kinectfusion::export_ply("data/mesh.ply", mesh);

// Export pointcloud
auto pointcloud = pipeline.extract_pointcloud();
kinectfusion::export_ply("data/pointcloud.ply", pointcloud);
```
For a more in-depth example and implementations of the data sources, have a look at the [KinectFusionApp](https://github.com/chrdiller/KinectFusionApp).

License
-------
This library is licensed under MIT.
