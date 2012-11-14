#ifndef DEPTHESTIMATOR_OPENCL_H_
#define DEPTHESTIMATOR_OPENCL_H_

#include "types.hpp"
#include "depthestimation.hpp"
#include "oclUtils.h"

#define __CL_ENABLE_EXCEPTIONS

#include <map>
#include <string>
#include <vector>

#include <CL/cl.hpp>
#include <boost/shared_ptr.hpp>
#include <logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>        // GPU structures and methods



namespace DepthEstimation {

/* forward declaration */
class KernelUnit;

class OpenCL : public DepthEstimator {
public:

    OpenCL();
    ~OpenCL();

    void estimate(uint8_t *input_l, uint8_t *input_r, uint8_t *result_l, uint8_t *result_r);

    /*
     * device <-> host copiers
     */
    void copyHostToDevice(uint8_t *host_buffer, cl::Buffer device_buffer);
    void copyHostToDevice(uint8_t *host_buffer, cl::Image2D device_image);
    void copyHostToDevice(uint8_t *host_buffer, cl::Image2D device_image, int width, int height);
    void copyHostToDeviceRect(uint8_t *host_buffer, cl::Buffer device_buffer, int row_pitch);
    void copyHostToDeviceRect(uint8_t *host_buffer, cl::Image2D device_image, int row_pitch);

    void copyDeviceToHost(cl::Buffer device_buffer, uint8_t *host_buffer);
    void copyDeviceToHost(cl::Image2D device_image, uint8_t *host_buffer);
    void copyDeviceToHostRect(cl::Buffer device_buffer, uint8_t *host_buffer);
    void copyDeviceToHostRect(cl::Image2D device_image, uint8_t *host_buffer);

    void copyDeviceToDevice(cl::Buffer buffer_src, cl::Buffer buffer_dst);
    void copyDeviceToDevice(cl::Image2D image_src, cl::Image2D image_dst);

    /* This sections really should be private */
//private:

    void errorHandler(cl::Error error);
    int roundUp(int value, int multiple);

    std::vector<cl::Platform> m_cl_platforms;
    std::vector<cl::Device> m_cl_devices;
    cl::Context m_cl_context;
    cl::CommandQueue m_cl_cqueue;

    void initOpenCLVariables();
    void initializeKernels();
    bool m_kernels_initialized;

    KernelUnit *algorithm_kernel;
};

/*
 * Abstract-ish base class
 */
class KernelUnit {
public:

    virtual ~KernelUnit() {};

    /*
     * These functions must be overridden by a concrete KernelUnit subclass
     */
    virtual void prepare() = 0;
    virtual void setArgs() = 0;
    virtual void postpare()= 0;
    virtual void launch();

protected:
    KernelUnit(OpenCL &parent, const char *kernel_source);

    /*
     * Each KernelUnit has at least these variables
     */
    OpenCL &p;
    cl::Program program;
    cl::Kernel kernel;
    cl::NDRange local_group_size;
    cl::NDRange global_group_size;
    cl::NDRange launch_offset;

    void compileSource(std::string compile_options);

    /* Dumps the kernel 'binary' ptx code to a filename named <kernel_name>.ptx
       in the working directory */
    void dumpBinary(std::string filename);


    /* and this debug function */
    void printKernelParams();
    void printKernelOptions(std::string options);
};

class NormalizeKernel : public KernelUnit {
public:
    NormalizeKernel(OpenCL &parent);

    cl::Kernel kernel_buffer;
    cl::Kernel kernel_2d;

    Buffer_ptr md_input_buffer, md_tmp_buffer;
    Image2D_ptr md_input_image, md_tmp_image;

    void setBuffer(Buffer_ptr input);
    void setBuffer(Image2D_ptr input);

    void prepare();
    void setArgs();
    void postpare();
};



class CrossCheckKernel : public KernelUnit {
public:

    CrossCheckKernel(OpenCL &parent);
    cl::Kernel kernel_2d;

    int m_width, m_height;

    Buffer_ptr md_input_l,
               md_input_r;

    Buffer_ptr md_cross_checked_l,
               md_cross_checked_r;

    Image2D_ptr md_input_l_2d,
                md_input_r_2d;

    std::map<int, Image2D_ptr> md_cross_checked_l_2d;
    std::map<int, Image2D_ptr> md_cross_checked_r_2d;

    bool m_using_image2d;

    void setBuffers(Buffer_ptr input_l, Buffer_ptr input_r);
    void setBuffers(Image2D_ptr input_l, Image2D_ptr input_r);

    void prepare();
    void setArgs();
    void postpare();
};

class OcclusionFillCPU : public KernelUnit {
public:
    OcclusionFillCPU(OpenCL &parent);

    void prepare() {};
    void setArgs() {};
    void postpare() {};

    void launch();

};

class BlockMatchingAlgorithm : public KernelUnit {
public:
    BlockMatchingAlgorithm(OpenCL &parent, bool nodef);

    void prepare();
    void setArgs();
    void postpare();

    bool cur_buf;
    bool nodef;

    Buffer_ptr md_input_l[2];
    Buffer_ptr md_input_r[2];
    Buffer_ptr md_result_l[2];
    Buffer_ptr md_result_r[2];

    NormalizeKernel *normalize_kernel;
};

class BlockMatchingLMEMAlgorithm : public KernelUnit {
public:
    BlockMatchingLMEMAlgorithm(OpenCL &parent);

    void prepare();
    void setArgs();
    void postpare();

    bool cur_buf;

    Buffer_ptr md_input_l[2];
    Buffer_ptr md_input_r[2];
    Buffer_ptr md_result_l[2];
    Buffer_ptr md_result_r[2];

    NormalizeKernel *normalize_kernel;
};

class DualBlockMatchingAlgorithm : public KernelUnit {
public:
    DualBlockMatchingAlgorithm(OpenCL &parent);
    DualBlockMatchingAlgorithm(OpenCL &parent, bool);
    void prepare();
    void setArgs();
    void postpare();

    NormalizeKernel *normalize_kernel;
    CrossCheckKernel *crosscheck_kernel;
    OcclusionFillCPU *occlusion_fill_kernel;

    bool no_def;

    int m_padded_size;
    int m_padded_width;
    int m_padded_height;

    bool cur_buf;

    Buffer_ptr md_input_l[2];
    Buffer_ptr md_input_r[2];
    Buffer_ptr md_result_l[2];
    Buffer_ptr md_result_r[2];
};

class PyramidAlgorithm : public KernelUnit {
public:
    PyramidAlgorithm(OpenCL &parent);

    void prepare();
    void setArgs();
    void postpare();

    /* Override the launch function */
    void launch();

    NormalizeKernel *normalize_kernel;
    CrossCheckKernel *crosscheck_kernel;

    void resizer();
    void createBuffers();
    void transferInput();

    bool cur_buf;
    bool buffers_created;

    cl::Kernel lvl1_kernel, lvl2_kernel, lvl3_kernel;

    cl::NDRange global_group_size_lvl1;
    cl::NDRange global_group_size_lvl2;
    cl::NDRange global_group_size_lvl3;

    Image2D_ptr md_input_l_2d[2];
    Image2D_ptr md_input_r_2d[2];
    Image2D_ptr md_result_l_2d[2];
    Image2D_ptr md_result_r_2d[2];

    cv::Mat lvl1_l, lvl1_r;
    cv::Mat lvl2_l, lvl2_r;
    cv::Mat lvl3_l, lvl3_r;

    Image2D_ptr md_lvl1_l, md_lvl1_r,
                md_lvl2_l, md_lvl2_r,
                md_lvl3_l, md_lvl3_r;

    Image2D_ptr md_lvl1_res_l, md_lvl1_res_r,
                md_lvl2_res_l, md_lvl2_res_r,
                md_lvl3_res_l, md_lvl3_res_r;


};     // class


class CPUAlgorithm : public KernelUnit {
public:
    CPUAlgorithm(OpenCL &parent);

    uchar* costs;
    uchar* aggregated_costs;

    void calculateCosts();
    void aggregateCosts();
    void disparitySelection();
    void launch();

    /* just override with no implementation */
    void prepare() {};
    void setArgs() {};
    void postpare() {};

};

class OpenCVCPUAlgorithm : public KernelUnit {
public:
    OpenCVCPUAlgorithm(OpenCL &parent);

    cv::Mat left, right, result;

    void launch();

    void prepare() {};
    void setArgs() {};
    void postpare() {};
};

class OpenCVAlgorithm : public KernelUnit {
public:
    OpenCVAlgorithm(OpenCL &parent);

    cv::Mat left, right;
    cv::gpu::GpuMat d_left, d_right;

    cv::gpu::StereoBM_GPU bm;

    void launch();

    /* just override with no implementation */
    void prepare() {};
    void setArgs() {};
    void postpare() {};
};


class BirchfieldTomasiAlgorithm : public KernelUnit {
public:
    BirchfieldTomasiAlgorithm(OpenCL &parent);

    Image2D_ptr md_input_l;
    Image2D_ptr md_input_r;
    Image2D_ptr md_result_l;
    Image2D_ptr md_result_r;

    void prepare();
    void setArgs();
    void postpare();
};


};     // namespace
#endif // include guard
