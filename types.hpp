#ifndef DEPTH_TYPES_H_
#define DEPTH_TYPES_H_

#include <boost/shared_ptr.hpp>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace DepthEstimation {

enum Algorithm {
    OPENCV,
    OPENCV_GPU,
    CPU,
    BM,
    BM_NO_DEF,
    BMLMEM,
    BMLMEM_NO_DEF,
    DUAL,
    DUAL_NO_DEF,
    TEXTURE,
    PYRAMID,
    BT // Birchfield & Tomasi
};

enum Option {
    USE_CROSS_CHECK_KERNEL = 1 << 1,
    USE_NORMALIZE_KERNEL = 1 << 2,
    USE_DIFFING_KERNEL =  1 << 3,
    USE_IMAGE2D = 1 << 4,
    USE_MIN_MAX = 1 << 5,
    USE_OCCLUSION_FILL = 1 << 6,
};

// #define KERNEL_SOURCE "data/depthestimation/kernels/kernels.cl"

typedef boost::shared_ptr<cl::Buffer> Buffer_ptr;
typedef boost::shared_ptr<cl::Image2D> Image2D_ptr;
typedef boost::shared_ptr<cl::Kernel> Kernel_ptr;

};
#endif
