// CUDA kernel header with commonly used definitions, functions and data structures
// Author: Christian Diller, git@christian-diller.de

//If you are working with CUDA code in CLion
#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
using blockDim = struct { int x; int y; };
using threadIdx = struct { int x; int y; int z; };
using blockIdx = struct { int x; int y; int z; };
#endif

#include <data_types.h>

#define DIVSHORTMAX 0.0000305185f //1.f / SHRT_MAX;
#define SHORTMAX 32767 //SHRT_MAX;
#define MAX_WEIGHT 128

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;
using cv::cuda::GpuMat;

using Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
