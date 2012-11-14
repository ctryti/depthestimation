#include "depthestimation_opencl.hpp"
#include "oclUtils.h"

#include <string>
#include <CL/cl.hpp>
#include <logging.h>

//#include <iostream>
#include <fstream>

namespace DepthEstimation {


OpenCL::OpenCL() {

    /* First, setup the opencl platform and opencl-c program  */
    try {
        /* get the platform */
        cl::Platform::get(&m_cl_platforms);
        if(m_cl_platforms.size() == 0) {
            ERROR("No OpenCL platforms found");
            exit(1);
        }

        INFO_OP("Platforms: " << m_cl_platforms.size());

        bool gpu_platform_found = false;
        for(size_t i = 0; i < m_cl_platforms.size(); i++) {

            std::string platform_name = m_cl_platforms[i].getInfo<CL_PLATFORM_VENDOR>();
            INFO_OP("Info: " << platform_name);

            if(platform_name == "NVIDIA Corporation") {
                /* get the context */
                cl_context_properties props[] = {
                    CL_CONTEXT_PLATFORM,
                    (cl_context_properties)(m_cl_platforms[i])(),
                    0
                };
                m_cl_context = cl::Context(CL_DEVICE_TYPE_GPU, props);

                /* get the device */
                m_cl_devices = m_cl_context.getInfo<CL_CONTEXT_DEVICES>();

                /* create the command queue */
                m_cl_cqueue = cl::CommandQueue(m_cl_context, m_cl_devices[0]);

                gpu_platform_found = true;
            }
        }

        if(!gpu_platform_found) {
            ERROR("No GPU platform found");
            exit(-1);
        }

    } catch(cl::Error error) {
        errorHandler(error);
    }

    m_kernels_initialized = false;
}

OpenCL::~OpenCL() {}


void OpenCL::initializeKernels() {

    if(m_kernels_initialized)
        return;

    //compileKernels();
    try {
        switch(m_algorithm) {

        case BM:
            algorithm_kernel = new BlockMatchingAlgorithm(*this, false);
            break;

        case BM_NO_DEF:
            algorithm_kernel = new BlockMatchingAlgorithm(*this, true);
            break;

        case DUAL:
            algorithm_kernel = new DualBlockMatchingAlgorithm(*this);
            break;

        case DUAL_NO_DEF:
            algorithm_kernel = new DualBlockMatchingAlgorithm(*this, true);
            break;

        case PYRAMID:
            algorithm_kernel = new PyramidAlgorithm(*this);
            break;

        case CPU:
            algorithm_kernel = new CPUAlgorithm(*this);
            break;

        case OPENCV_GPU:
            algorithm_kernel = new OpenCVAlgorithm(*this);
            break;

        case OPENCV:
            algorithm_kernel = new OpenCVCPUAlgorithm(*this);
            break;

        case BT:
            algorithm_kernel = new BirchfieldTomasiAlgorithm(*this);
            break;

        default:
            DBG_OP("Invalid algorithm: " << m_algorithm);
        }
    } catch(cl::Error e) {
        errorHandler(e);
    }
    m_kernels_initialized = true;
}



void OpenCL::estimate(uint8_t *input_l, uint8_t *input_r, uint8_t *result_l, uint8_t *result_r) {

    m_input_l = input_l;
    m_input_r = input_r;
    m_result_l = result_l;
    m_result_r = result_r;

    initializeKernels();

    NorthLight::TimeSpec begin;// = NorthLight::getTime();
    NorthLight::TimeSpec duration;
    try {
        begin = NorthLight::getTime();
        algorithm_kernel->launch();
        duration = NorthLight::getTime() - begin;
    } catch(cl::Error e) {
        errorHandler(e);
    }
    m_timing_db_ptr->insert(duration);
    INFO_OP("Frame #" << frame_num << "(" << ((double)duration) * 1000 << " ms, mean: "
            << ((double)m_timing_db_ptr->mean()) * 1000 << " ms)");
}

void OpenCL::copyHostToDevice(uint8_t *host_buffer, cl::Buffer device_buffer) {
    m_cl_cqueue.enqueueWriteBuffer(device_buffer, CL_FALSE, 0, m_size, host_buffer);
}

void OpenCL::copyHostToDevice(uint8_t *host_buffer, cl::Image2D device_image) {
    /* call the overloaded method with the input size */
    copyHostToDevice(host_buffer, device_image, m_width, m_height);
}

void OpenCL::copyHostToDevice(uint8_t *host_buffer, cl::Image2D device_image,
                              int width, int height) {
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    m_cl_cqueue.enqueueWriteImage(device_image, CL_FALSE, offset, region, 0, 0, host_buffer);
}


void OpenCL::copyDeviceToHost(cl::Buffer device_buffer, uint8_t *host_buffer) {
    m_cl_cqueue.enqueueReadBuffer(device_buffer, CL_FALSE, 0, m_size, host_buffer);
}

void OpenCL::copyDeviceToHost(cl::Image2D device_image, uint8_t *host_buffer) {
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    cl::size_t<3> region;
    region[0] = m_width;
    region[1] = m_height;
    region[2] = 1;

    m_cl_cqueue.enqueueReadImage(device_image, CL_FALSE, offset, region, 0, 0, host_buffer);
}

void OpenCL::copyDeviceToDevice(cl::Buffer buffer_src, cl::Buffer buffer_dst) {
    m_cl_cqueue.enqueueCopyBuffer(buffer_src, buffer_dst, 0, 0, m_width*m_height);
}

void OpenCL::copyDeviceToDevice(cl::Image2D image_src, cl::Image2D image_dst) {
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    cl::size_t<3> region;
    region[0] = image_src.getImageInfo<CL_IMAGE_WIDTH>();
    region[1] = image_src.getImageInfo<CL_IMAGE_HEIGHT>();
    region[2] = 1;

    m_cl_cqueue.enqueueCopyImage(image_src, image_dst, offset, offset, region);
}

void OpenCL::copyHostToDeviceRect(uint8_t *host_buffer, cl::Buffer device_buffer, int row_pitch) {
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;

    cl::size_t<3> region;
    region[0] = m_width;
    region[1] = m_height;
    region[2] = 1;
    m_cl_cqueue.enqueueWriteBufferRect(device_buffer, CL_FALSE, offset, offset, region,
                                       row_pitch, 0, m_width, 0, host_buffer);
}

void OpenCL::copyDeviceToHostRect(cl::Buffer device_buffer, uint8_t *host_buffer) {}
void OpenCL::copyDeviceToHostRect(cl::Image2D device_image, uint8_t *host_buffer) {}


/*
 * Handy cl::Error-to-string function.
 *
 * Also prints the build log for kernels that fail to compile.
 */
void OpenCL::errorHandler(cl::Error error) {
    ERROR_OP(error.what() << " : " << oclErrorString(error.err()));
    exit(1);
}

int OpenCL::roundUp(int value, int multiple) {
    int remainder = value % multiple;
    if(remainder != 0)
        value += multiple - remainder;
    return value;
}




/***********
 *
 * Below are KernelUnit implementations
 *
 ***********/


KernelUnit::KernelUnit(OpenCL &parent, const char *kernel_source)
    : p(parent),
      local_group_size(0,0),
      global_group_size(0,0),
      launch_offset(0,0) {

    if(kernel_source) {
        try {

            DBG_OP("Loading kernel source file from: " << kernel_source);

            size_t kernel_size;
            const char *source_str = oclLoadProgSource(kernel_source, "", &kernel_size);

            if(source_str == NULL) {
                DBG("oclLoadProgSource returned NULL");
                exit(-1);
            }

            DBG_OP("kernel_size: " << kernel_size);

            cl::Program::Sources source(1, std::make_pair(source_str, kernel_size));
            program = cl::Program(p.m_cl_context, source);
            DBG_OP("Created KernelUnit from source: " << kernel_source);

        } catch(cl::Error e) {
            p.errorHandler(e);
        }
    }
}

void KernelUnit::launch() {
    DBG_OP("Launching kernel: " << kernel.getInfo<CL_KERNEL_FUNCTION_NAME>());
    prepare();
    setArgs();
    printKernelParams();
    p.m_cl_cqueue.enqueueNDRangeKernel(kernel, launch_offset,
                                               global_group_size,
                                               local_group_size);
    p.m_cl_cqueue.finish();
    postpare();
}

void KernelUnit::printKernelParams() {
    DBG_OP("  local size:  " << local_group_size[0]  << " / " << local_group_size[1]);
    DBG_OP("  global size: " << global_group_size[0] << " / " << global_group_size[1]);
}

void KernelUnit::printKernelOptions(std::string options) {
    DBG("Compiling kernel with the following options:");
    std::istringstream iss(options, std::istringstream::in);
    std::string word;
    while(iss >> word)
        DBG_OP("  " << word);
}

void KernelUnit::compileSource(std::string compile_options) {

    DBG("compileSource()");

    /* These are the standard defines that most kernels need. Prepend them to compile_options */
    std::stringstream options("");
    options << " -DWIDTH="         << p.getWidth();
    options << " -DHEIGHT="        << p.getHeight();
    options << " -DSIZE="          << p.getSize();

    options << compile_options;

    printKernelOptions(options.str());

    try {
        program.build(p.m_cl_devices, options.str().c_str());
    } catch(cl::Error e) {
        ERROR_OP(e.what() << " : " << oclErrorString(e.err()));
        if(e.err() == CL_BUILD_PROGRAM_FAILURE || e.err() == CL_INVALID_BINARY) {
            std::string buildLog;
            program.getBuildInfo(p.m_cl_devices[0], CL_PROGRAM_BUILD_LOG, &buildLog);
            ERROR_OP("\n" << buildLog);
        }
    }
}

void KernelUnit::dumpBinary(std::string filename) {

    DBG("dumpBinary()");

    /* Find the size of the binary */
    size_t size = program.getInfo<CL_PROGRAM_BINARY_SIZES>()[0];

    /* Initialize a char pointer with the binary size */
    std::vector<char *> binary;
    binary.insert(binary.end(), new char[size]);

    /* extract the binary */
    program.getInfo(CL_PROGRAM_BINARIES, &binary);

    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);

    if(!file.is_open()) {
        ERROR_OP("Error while writing binary program: " << filename);
        exit(-1);
    }

    file.write(binary[0], size);

    delete binary[0];

}


/******
 *
 * Block Matching kernel
 *
 ******/
BlockMatchingAlgorithm::BlockMatchingAlgorithm(OpenCL &parent, bool _nodef)
    : KernelUnit(parent, "data/depthestimation/kernels/blockmatching.cl"),
      nodef(_nodef),
      normalize_kernel(NULL) {


    std::stringstream compile_options("");

    compile_options << " -DAGGR_DIM="    << p.m_aggregation_window_dimension;
    compile_options << " -DAGGR_RADIUS=" << (p.m_aggregation_window_dimension/2);
    compile_options << " -DMAX_DISP="    << p.m_maximum_disparity;

    compileSource(compile_options.str());

    if(nodef)
        kernel = cl::Kernel(program, "calculateDisparity_no_def");
    else
        kernel = cl::Kernel(program, "calculateDisparity");

    int dim = 32;
    int lx = dim;
    int ly = dim;

    int gx = p.roundUp(p.getWidth(), lx);
    int gy = p.roundUp(p.getHeight(), ly);

    local_group_size = cl::NDRange(lx,ly);
    global_group_size = cl::NDRange(gx,gy);

    if(p.getOptions() & USE_NORMALIZE_KERNEL) {
        normalize_kernel = new NormalizeKernel(p);
    }
}


void BlockMatchingAlgorithm::prepare() {

    /* if buffers don't exist, create them! */
    if(!md_input_l[cur_buf])
        md_input_l[cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, p.m_size));

    if(!(md_input_l[!cur_buf]))
        md_input_l[!cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, p.m_size));

    if(!md_input_r[cur_buf])
        md_input_r[cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, p.m_size));

    if(!md_input_r[!cur_buf])
        md_input_r[!cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, p.m_size));

    if(!md_result_l[cur_buf])
        md_result_l[cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE,p.m_size));

    if(!md_result_l[!cur_buf])
        md_result_l[!cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, p.m_size));

    /* transfer input to the gpu */
    p.copyHostToDevice(p.m_input_l, *md_input_l[cur_buf]);
    p.copyHostToDevice(p.m_input_r, *md_input_r[cur_buf]);
}

void BlockMatchingAlgorithm::setArgs() {
    int i = 0;
    kernel.setArg(i++, sizeof(cl_mem), &*md_input_l[cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_input_r[cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_result_l[cur_buf]);
    if(nodef) {
        kernel.setArg(i++, sizeof(int), &p.m_width);
        kernel.setArg(i++, sizeof(int), &p.m_aggregation_window_dimension);
        int radius = p.m_aggregation_window_dimension / 2;
        kernel.setArg(i++, sizeof(int), &radius);
        kernel.setArg(i++, sizeof(int), &p.m_maximum_disparity);
    }
}

void BlockMatchingAlgorithm::postpare() {

    /* If normalization is needed, run the kernel
     * else just copy the disparity map back to the host */
    if(p.getOptions() & USE_NORMALIZE_KERNEL) {
        normalize_kernel->launch();
    } else {
        p.copyDeviceToHost(*md_result_l[cur_buf], p.m_result_l);
    }
}



/******
 *
 * Cross checking kernel
 *
 ******/
CrossCheckKernel::CrossCheckKernel(OpenCL &parent)
    : KernelUnit(parent, "data/depthestimation/kernels/crosscheck.cl"),
      m_width(-1),
      m_height(-1) {

    compileSource(" -DTHRESHOLD=2 ");

    // if(m_use_image2d)
    // else
    kernel = cl::Kernel(program, "crossCheck");
    kernel_2d = cl::Kernel(program, "crossCheck_image2d");

    // int lx = 512;
    // int ly = 1;
    // local_group_size = cl::NDRange(lx,ly);

    // int gx = p.roundUp(p.m_width, lx);
    // int gy = p.roundUp(p.m_height, ly);
    // global_group_size = cl::NDRange(gx,gy);
}

void CrossCheckKernel::setBuffers(Buffer_ptr input_l, Buffer_ptr input_r) {
    DBG("Setting cl::Buffer buffers");
    md_input_l = input_l;
    md_input_r = input_r;
    m_width = p.m_width;
    m_height = p.m_height;
}

void CrossCheckKernel::setBuffers(Image2D_ptr input_l, Image2D_ptr input_r) {
    DBG("Setting image2d buffers");
    md_input_l_2d = input_l;
    md_input_r_2d = input_r;
    m_width  = (*input_l).getImageInfo<CL_IMAGE_WIDTH>();
    m_height = (*input_l).getImageInfo<CL_IMAGE_HEIGHT>();
    kernel = kernel_2d;
}

void CrossCheckKernel::prepare() {

    DBG("  prepare()");

    if(md_input_l && md_input_r) {
        if(!md_cross_checked_l && !md_cross_checked_r) {
            md_cross_checked_l.reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));
            md_cross_checked_r.reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));
        }
    } else if(md_input_l_2d && md_input_r_2d) {

        int key = m_width * m_height;

        if(md_cross_checked_l_2d.count(key) == 0) {
            DBG_OP("Creating crosschecking buffers for key: " << key);
            cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);
            md_cross_checked_l_2d[key].reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE,
                                             format, m_width, m_height));
            md_cross_checked_r_2d[key].reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE,
                                             format, m_width, m_height));
        }
    } else {
        ERROR("CrossCheckKernel::setBuffers() not called");
        exit(1);
    }

    int lx = 512;
    int ly = 1;
    local_group_size = cl::NDRange(lx, ly);

    int gx = p.roundUp(m_width, lx);
    int gy = p.roundUp(m_height, ly);
    global_group_size = cl::NDRange(gx, gy);
}

void CrossCheckKernel::setArgs() {
    DBG("  setArgs()");
    int i = 0;
    if(md_input_l) {
        kernel.setArg(i++, sizeof(cl_mem), &*md_input_l);
        kernel.setArg(i++, sizeof(cl_mem), &*md_input_r);
        kernel.setArg(i++, sizeof(cl_mem), &*md_cross_checked_l);
        kernel.setArg(i++, sizeof(cl_mem), &*md_cross_checked_r);
    } else {
        int key = m_width * m_height;
        kernel_2d.setArg(i++, *md_input_l_2d);
        kernel_2d.setArg(i++, *md_input_r_2d);
        kernel_2d.setArg(i++, *md_cross_checked_l_2d[key]);
        kernel_2d.setArg(i++, *md_cross_checked_r_2d[key]);
    }
}

void CrossCheckKernel::postpare() {
    DBG("  postpare()");
    if(md_cross_checked_l) {
        p.copyDeviceToDevice(*md_cross_checked_l, *md_input_l);
        p.copyDeviceToDevice(*md_cross_checked_r, *md_input_r);
    } else {
        int key = m_width * m_height;
        p.copyDeviceToDevice(*md_cross_checked_l_2d[key], *md_input_l_2d);
        p.copyDeviceToDevice(*md_cross_checked_r_2d[key], *md_input_r_2d);
    }
    md_input_l.reset();
    md_input_r.reset();
    md_input_l_2d.reset();
    md_input_r_2d.reset();
}



/******
 *
 * OcclusionFillCPU: Fills occluded areas
 *
 ******/
OcclusionFillCPU::OcclusionFillCPU(OpenCL &parent)
    : KernelUnit(parent, NULL) {
    DBG("OcclusionFill object created");
}

void OcclusionFillCPU::launch() {

    uchar tmp[p.m_size];

    // for(int i = 0; i < p.m_size; i++)
    //     tmp[i] = 0;

    int previous_disp = 0;
    for(int i = 0; i < p.m_height; i++) {
        previous_disp = p.m_result_l[i * p.m_width];
        for(int j = 0; j < p.m_width; j++) {
            if(p.m_result_l[i * p.m_width + j] == 0) {
                p.m_result_l[i * p.m_width + j] = previous_disp;
                // tmp[i * p.m_width + j] = 1;
            }
            previous_disp = p.m_result_l[i * p.m_width + j];
        }
    }

    // for(int i = 0; i < p.m_height; i++) {
    //     for(int j = 0; j < p.m_width; j++) {
    //         if(tmp[i*p.m_width+j] == 1) {
    //             if(i < p.m_height-1 && i > 0 && j < p.m_width-1 && j > 0) {
    //                 int interpolated = (p.m_result_l[i*p.m_width+j] + p.m_result_l[(i-1)*p.m_width+j]
    //                                  + p.m_result_l[(i+1)*p.m_width+j] + p.m_result_l[(i)*p.m_width+j+1]
    //                                  + p.m_result_l[(i)*p.m_width+j-1]) / 5;
    //                 p.m_result_l[i*p.m_width+j] = interpolated;
    //             }
    //         }
    //     }
    // }

}


/******
 *
 * Normalize kernel
 *
 ******/
NormalizeKernel::NormalizeKernel(OpenCL &parent)
    : KernelUnit(parent, "data/depthestimation/kernels/normalize.cl") {

    //kernel = cl::Kernel(program, "normalizeResults");

    compileSource("");

    kernel_buffer = cl::Kernel(program, "normalizeResults");
    kernel_2d = cl::Kernel(program, "normalizeResults_image2d");

    int lx = 512;
    int ly = 1;
    local_group_size = cl::NDRange(lx,ly);

    int gx = p.roundUp(p.m_width, lx);
    int gy = p.m_height;
    global_group_size = cl::NDRange(gx,gy);
}

void NormalizeKernel::setBuffer(Buffer_ptr input) {
    md_input_buffer = input;
    md_tmp_buffer.reset(new cl::Buffer(p.m_cl_context, CL_MEM_WRITE_ONLY, p.m_size));
    kernel = kernel_buffer;
}

void NormalizeKernel::setBuffer(Image2D_ptr input) {
    cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);
    md_input_image = input;
    md_tmp_image.reset(new cl::Image2D(p.m_cl_context, CL_MEM_WRITE_ONLY, format, p.m_width, p.m_height));
    kernel = kernel_2d;
}

void NormalizeKernel::setArgs() {

    int i = 0;
    if(md_tmp_buffer) {
        kernel.setArg(i++, *md_input_buffer);
        kernel.setArg(i++, *md_tmp_buffer);
    } else if(md_tmp_image) {
        kernel_2d.setArg(i++, *md_input_image);
        kernel_2d.setArg(i++, *md_tmp_image);
        kernel_2d.setArg(i++, (float)15*16);
    } else {
        ERROR("NormalizeKernel::setBuffer() not called!");
        exit(1);
    }


    // int i = 0;
    // kernel.setArg(i++, sizeof(cl_mem), &*(md_result_l[cur_buf]));

    // /* If there is a right result, set it, else set NULL and the
    //    kernel won't calculate for it */
    // if(md_result_r[cur_buf])
    //     kernel.setArg(i++, sizeof(cl_mem), &*(md_result_r[cur_buf]));
    // else
    //     kernel.setArg(i++, sizeof(cl_mem), NULL);
}

void NormalizeKernel::prepare() {}
void NormalizeKernel::postpare() {
    if(md_tmp_buffer) {
       p.copyDeviceToDevice(*md_tmp_buffer, *md_input_buffer);
       md_tmp_buffer.reset();
   } else if(md_tmp_image) {
    p.copyDeviceToDevice(*md_tmp_image, *md_input_image);
    md_tmp_image.reset();
   }
}



/******
 *
 * Dual kernel
 *
 ******/
DualBlockMatchingAlgorithm::DualBlockMatchingAlgorithm(OpenCL &parent)
    : KernelUnit(parent, "data/depthestimation/kernels/blockmatching_dual.cl"),
      normalize_kernel(NULL),
      crosscheck_kernel(NULL),
      cur_buf(0) {

    m_padded_height = p.roundUp(p.m_height + p.m_aggregation_window_dimension - 1, 16);
    m_padded_width = p.roundUp(p.m_width + p.m_aggregation_window_dimension - 1, 16);
    m_padded_size = m_padded_width * m_padded_height;

    std::stringstream compile_options("");
    compile_options << " -DPADDED_WIDTH="  << m_padded_width;
    compile_options << " -DPADDED_HEIGHT=" << m_padded_height;
    compile_options << " -DPADDED_SIZE="   << m_padded_size;
    compile_options << " -DAGGR_DIM="      << p.m_aggregation_window_dimension;
    compile_options << " -DAGGR_RADIUS="   << (p.m_aggregation_window_dimension/2);
    compile_options << " -DMAX_DISP="      << p.m_maximum_disparity;

    no_def = false;

    compileSource(compile_options.str());

    kernel = cl::Kernel(program, "calculateDisparityDual");

    int dim = 16;
    int lx = dim;
    int ly = dim;
    local_group_size = cl::NDRange(lx,ly);

    int gx = p.roundUp(p.m_width, dim);
    int gy = p.roundUp(p.m_height, dim);
    global_group_size = cl::NDRange(gx,gy);

    if(p.m_options & USE_CROSS_CHECK_KERNEL)
        crosscheck_kernel = new CrossCheckKernel(p);
    if(p.m_options & USE_NORMALIZE_KERNEL)
        normalize_kernel = new NormalizeKernel(p);
    if(p.m_options & USE_OCCLUSION_FILL)
        occlusion_fill_kernel = new OcclusionFillCPU(p);
}

/* TODO: Should probably do something about these two constructors... */
DualBlockMatchingAlgorithm::DualBlockMatchingAlgorithm(OpenCL &parent, bool)
    : KernelUnit(parent, "data/depthestimation/kernels/blockmatching_dual_no_def.cl"),
      normalize_kernel(NULL),
      crosscheck_kernel(NULL),
      cur_buf(0) {

    no_def = true;

    m_padded_height = p.roundUp(p.m_height + p.m_aggregation_window_dimension - 1, 16);
    m_padded_width = p.roundUp(p.m_width + p.m_aggregation_window_dimension - 1, 16);
    m_padded_size = m_padded_width * m_padded_height;

    compileSource("");

    kernel = cl::Kernel(program, "calculateDisparityDual_no_def");

    int dim = 16;
    int lx = dim;
    int ly = dim;
    local_group_size = cl::NDRange(lx,ly);

    int gx = p.roundUp(p.m_width, dim);
    int gy = p.roundUp(p.m_height, dim);
    global_group_size = cl::NDRange(gx,gy);

    if(p.m_options & USE_CROSS_CHECK_KERNEL)
        crosscheck_kernel = new CrossCheckKernel(p);
    if(p.m_options & USE_NORMALIZE_KERNEL)
        normalize_kernel = new NormalizeKernel(p);
}


void DualBlockMatchingAlgorithm::prepare() {

    // if(!md_input_l[cur_buf])
        md_input_l[cur_buf].reset(  new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, m_padded_size));

    // if(!md_input_l[!cur_buf])
        md_input_l[!cur_buf].reset( new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, m_padded_size));

    // if(!md_input_r[cur_buf])
        md_input_r[cur_buf].reset(  new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, m_padded_size));

    // if(!md_input_r[!cur_buf])
        md_input_r[!cur_buf].reset( new cl::Buffer(p.m_cl_context, CL_MEM_READ_ONLY, m_padded_size));

    // if(!md_result_l[cur_buf])
        md_result_l[cur_buf].reset( new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));

    // if(!md_result_l[!cur_buf])
        md_result_l[!cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));

    // if(!md_result_r[cur_buf])
        md_result_r[cur_buf].reset( new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));

    // if(!md_result_r[!cur_buf])
        md_result_r[!cur_buf].reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));

    // if(!md_diff_map)
    //     md_diff_map.reset(new cl::Buffer(p.m_cl_context, CL_MEM_READ_WRITE, p.m_size));

    /* transfer input to the gpu */
    p.copyHostToDeviceRect(p.m_input_l, *md_input_l[cur_buf], m_padded_width);
    p.copyHostToDeviceRect(p.m_input_r, *md_input_r[cur_buf], m_padded_width);
    DBG("Prepare complete");
}

void DualBlockMatchingAlgorithm::postpare() {
    if(crosscheck_kernel) {
        crosscheck_kernel->setBuffers(md_result_l[cur_buf], md_result_r[cur_buf]);
        crosscheck_kernel->launch();
    }

    // if(normalize_kernel) {
    //     normalize_kernel->setBuffer(md_result_l[cur_buf]);
    //     normalize_kernel->launch();
    //     normalize_kernel->setBuffer(md_result_r[cur_buf]);
    //     normalize_kernel->launch();
    // }

    p.copyDeviceToHost(*md_result_l[cur_buf], p.m_result_l);
    p.copyDeviceToHost(*md_result_r[cur_buf], p.m_result_r);

    if(occlusion_fill_kernel) {
        occlusion_fill_kernel->launch();
    }

    cur_buf = !cur_buf;

}

void DualBlockMatchingAlgorithm::setArgs() {

    /* local memory buffers */
    int lx = local_group_size[0];
    int ly = local_group_size[1];
    int local_width = p.roundUp(lx + p.m_aggregation_window_dimension - 1 +
                                p.m_maximum_disparity, lx);
    int local_height = p.roundUp(((p.m_aggregation_window_dimension - 1) + ly), ly);
    int local_size = local_height * local_width;

    int i = 0;
    kernel.setArg(i++, sizeof(cl_mem), &*md_input_l[cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_input_r[cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_result_l[cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_result_r[cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_result_l[!cur_buf]);
    kernel.setArg(i++, sizeof(cl_mem), &*md_result_r[!cur_buf]);
    kernel.setArg(i++, local_size, NULL);
    kernel.setArg(i++, local_size, NULL);
    kernel.setArg(i++, sizeof(int),    &local_width);
    kernel.setArg(i++, sizeof(int),    &local_height);
    //kernel.setArg(i++, sizeof(cl_mem), &*p.md_diff_map);
    kernel.setArg(i++, sizeof(cl_mem), NULL);

    /*  */
    if(no_def) {
        kernel.setArg(i++, sizeof(int), &p.m_width);
        kernel.setArg(i++, sizeof(int), &p.m_height);
        kernel.setArg(i++, sizeof(int), &m_padded_width);
        kernel.setArg(i++, sizeof(int), &p.m_maximum_disparity);
        int radius = p.m_aggregation_window_dimension/2;
        kernel.setArg(i++, sizeof(int), &radius);
        kernel.setArg(i++, sizeof(int), &p.m_aggregation_window_dimension);
    }
    DBG("setArgs complete");
}



/******
 *
 * Pyramid algorithm
 *
 ******/
PyramidAlgorithm::PyramidAlgorithm(OpenCL &parent)
    : KernelUnit(parent, "data/depthestimation/kernels/pyramid.cl"),
      normalize_kernel(NULL),
      crosscheck_kernel(NULL),
      cur_buf(false),
      buffers_created(false) {

    resizer();

    std::stringstream options("");

    options << " -DAGGR_DIM="      << p.m_aggregation_window_dimension;
    options << " -DAGGR_RADIUS="   << (p.m_aggregation_window_dimension/2);
    options << " -DMAX_DISP="      << p.m_maximum_disparity;
   /* Pyramid leve values */
    options << " -DMAX_DISP_1="      << round(p.m_maximum_disparity * 0.5);
    options << " -DMAX_DISP_2="      << round(p.m_maximum_disparity * 0.25);
    options << " -DMAX_DISP_3="      << round(p.m_maximum_disparity * 0.125);
    options << " -DWIDTH_1="         << lvl1_l.cols;//round(p.m_width  / 2.0);
    options << " -DHEIGHT_1="        << lvl1_l.rows;//round(p.m_height / 2.0);
    options << " -DWIDTH_2="         << lvl2_l.cols;//round(p.m_width  / 4.0);
    options << " -DHEIGHT_2="        << lvl2_l.rows;//round(p.m_height / 4.0);
    options << " -DWIDTH_3="         << lvl3_l.cols;//round(p.m_width  / 8.0);
    options << " -DHEIGHT_3="        << lvl3_l.rows;//round(p.m_height / 8.0);

    if(p.m_options & USE_MIN_MAX)
        options << " -DUSE_MIN_MAX";


    compileSource(options.str());

    // kernel = cl::Kernel(program, "calculateDisparityPyramid");
    // lvl1_kernel = cl::Kernel(program, "pyramidLvl1");
    // lvl2_kernel = cl::Kernel(program, "pyramidLvl2");
    // lvl3_kernel = cl::Kernel(program, "pyramidLvl3");

    kernel =      cl::Kernel(program, "pyramid0");
    lvl1_kernel = cl::Kernel(program, "pyramid1");
    lvl2_kernel = cl::Kernel(program, "pyramid2");
    lvl3_kernel = cl::Kernel(program, "pyramid3");


    int dim = 16;
    int lx = dim;
    int ly = dim;
    local_group_size = cl::NDRange(lx,ly);

    int gx = p.roundUp(p.m_width, dim);
    int gy = p.roundUp(p.m_height, dim);
    global_group_size = cl::NDRange(gx,gy);

    /* set up scaled work sizes */

    gx = p.roundUp(lvl1_l.cols, dim);
    gy = p.roundUp(lvl1_l.rows, dim);
    global_group_size_lvl1 = cl::NDRange(gx,gy);

    gx = p.roundUp(lvl2_l.cols, dim);
    gy = p.roundUp(lvl2_l.rows, dim);
    global_group_size_lvl2 = cl::NDRange(gx,gy);

    gx = p.roundUp(lvl3_l.cols, dim);
    gy = p.roundUp(lvl3_l.rows, dim);
    global_group_size_lvl3 = cl::NDRange(gx,gy);

    if(p.m_options & USE_CROSS_CHECK_KERNEL) {
        crosscheck_kernel = new CrossCheckKernel(p);
    }
    if(p.m_options & USE_NORMALIZE_KERNEL) {
        normalize_kernel = new NormalizeKernel(p);
    }
}

/*
 * This code can probably be done with OpenCL, but i'm not familiar
 * enough with resizing algorithms, so i'll just use OpenCV for now.
 */
void PyramidAlgorithm::resizer() {
    NorthLight::TimeSpec begin = NorthLight::getTime();

    /* wrap the input in a cv::Mat */
    cv::Mat orig_l = cv::Mat(p.m_height, p.m_width, CV_8UC1);
    orig_l.data = p.m_input_l;

    cv::Mat orig_r = cv::Mat(p.m_height, p.m_width, CV_8UC1);
    orig_r.data = p.m_input_r;

    /* Then create scaled copies of it */
    int interpolation = cv::INTER_CUBIC;

    /* Half size */
    cv::resize(orig_l, lvl1_l, cv::Size(), 0.5, 0.5, interpolation);
    cv::resize(orig_r, lvl1_r, cv::Size(), 0.5, 0.5,interpolation);

    /* Half size of the above */
    cv::resize(orig_l, lvl2_l, cv::Size(), 0.25, 0.25, interpolation);
    cv::resize(orig_r, lvl2_r, cv::Size(), 0.25, 0.25, interpolation);

    /* Half size of the above */
    cv::resize(orig_l, lvl3_l, cv::Size(), 0.125, 0.125, interpolation);
    cv::resize(orig_r, lvl3_r, cv::Size(), 0.125, 0.125, interpolation);

    NorthLight::TimeSpec duration = NorthLight::getTime() - begin;
    DBG_OP("Resizing images took: " << ((double)duration) * 1000 << " ms");

}

void PyramidAlgorithm::createBuffers() {

    if(buffers_created)
        return;

    cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);

    md_input_l_2d[cur_buf].reset(  new cl::Image2D(p.m_cl_context, CL_MEM_READ_ONLY, format, p.m_width, p.m_height));
    md_input_r_2d[cur_buf].reset(  new cl::Image2D(p.m_cl_context, CL_MEM_READ_ONLY, format, p.m_width, p.m_height));
    md_input_l_2d[!cur_buf].reset( new cl::Image2D(p.m_cl_context, CL_MEM_READ_ONLY, format, p.m_width, p.m_height));
    md_input_r_2d[!cur_buf].reset( new cl::Image2D(p.m_cl_context, CL_MEM_READ_ONLY, format, p.m_width, p.m_height));

    md_result_l_2d[cur_buf].reset( new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, p.m_width, p.m_height));
    md_result_r_2d[cur_buf].reset( new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, p.m_width, p.m_height));
    md_result_l_2d[!cur_buf].reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, p.m_width, p.m_height));
    md_result_r_2d[!cur_buf].reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, p.m_width, p.m_height));

    /* Create the scaled input and scaled result buffers */
    md_lvl1_l.reset(    new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl1_l.cols, lvl1_l.rows));
    md_lvl1_r.reset(    new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl1_r.cols, lvl1_r.rows));
    md_lvl1_res_l.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl1_l.cols, lvl1_l.rows));
    md_lvl1_res_r.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl1_r.cols, lvl1_r.rows));

    md_lvl2_l.reset(    new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl2_l.cols, lvl2_l.rows));
    md_lvl2_r.reset(    new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl2_r.cols, lvl2_r.rows));
    md_lvl2_res_l.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl2_l.cols, lvl2_l.rows));
    md_lvl2_res_r.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl2_r.cols, lvl2_r.rows));

    md_lvl3_l.reset(    new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl3_l.cols, lvl3_l.rows));
    md_lvl3_r.reset(    new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl3_r.cols, lvl3_r.rows));
    md_lvl3_res_l.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl3_l.cols, lvl3_l.rows));
    md_lvl3_res_r.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, lvl3_r.cols, lvl3_r.rows));

    buffers_created = true;
}

void PyramidAlgorithm::transferInput() {

    p.copyHostToDevice(p.m_input_l, *md_input_l_2d[cur_buf]);
    p.copyHostToDevice(p.m_input_r, *md_input_r_2d[cur_buf]);

    p.copyHostToDevice(lvl1_l.data, *md_lvl1_l, lvl1_l.cols, lvl1_l.rows);
    p.copyHostToDevice(lvl1_r.data, *md_lvl1_r, lvl1_r.cols, lvl1_r.rows);

    p.copyHostToDevice(lvl2_l.data, *md_lvl2_l, lvl2_l.cols, lvl2_l.rows);
    p.copyHostToDevice(lvl2_r.data, *md_lvl2_r, lvl2_r.cols, lvl2_r.rows);

    p.copyHostToDevice(lvl3_l.data, *md_lvl3_l, lvl3_l.cols, lvl3_l.rows);
    p.copyHostToDevice(lvl3_r.data, *md_lvl3_r, lvl3_r.cols, lvl3_r.rows);
}

void PyramidAlgorithm::setArgs() {

    int i;

    /* local memory buffer */
    int local_w3 = p.roundUp(local_group_size[0] + 4 +
                            round(p.m_maximum_disparity * 0.125),
                            local_group_size[0]);

    int local_h3 = p.roundUp(4 + local_group_size[1], local_group_size[1]);

    int local_size3 = local_h3 * local_w3;

    /* Lowest level kernel */
    i = 0;
    // lvl3_kernel.setArg(i++, 8);
    lvl3_kernel.setArg(i++, *md_lvl3_l);
    lvl3_kernel.setArg(i++, *md_lvl3_r);
    // lvl3_kernel.setArg(i++, *md_lvl3_res_l);
    // lvl3_kernel.setArg(i++, *md_lvl3_res_r);
    lvl3_kernel.setArg(i++, *md_lvl3_res_l);
    lvl3_kernel.setArg(i++, *md_lvl3_res_r);
    lvl3_kernel.setArg(i++, local_size3, NULL);
    lvl3_kernel.setArg(i++, local_size3, NULL);
    lvl3_kernel.setArg(i++, local_w3);
    lvl3_kernel.setArg(i++, local_h3);

    /* local memory buffer */
    int local_w2 = p.roundUp(local_group_size[0] + p.m_aggregation_window_dimension -
                        1 + round(p.m_maximum_disparity * 0.25),
                        local_group_size[0]);

    int local_h2 = p.roundUp(local_group_size[1] + p.m_aggregation_window_dimension -
                        1, local_group_size[1]);

    int local_size2 = local_h2 * local_w2;

    i = 0;
    // lvl2_kernel.setArg(i++, 4);
    lvl2_kernel.setArg(i++, *md_lvl2_l);
    lvl2_kernel.setArg(i++, *md_lvl2_r);
    lvl2_kernel.setArg(i++, *md_lvl3_res_l);
    lvl2_kernel.setArg(i++, *md_lvl3_res_r);
    lvl2_kernel.setArg(i++, *md_lvl2_res_l);
    lvl2_kernel.setArg(i++, *md_lvl2_res_r);
    lvl2_kernel.setArg(i++, local_size2, NULL);
    lvl2_kernel.setArg(i++, local_size2, NULL);
    lvl2_kernel.setArg(i++, local_w2);
    lvl2_kernel.setArg(i++, local_h2);

    /* local memory buffer */
    int local_w1 = p.roundUp(local_group_size[0] + p.m_aggregation_window_dimension -
                        1 + round(p.m_maximum_disparity * 0.5),
                        local_group_size[0]);

    int local_h1 = p.roundUp(local_group_size[1] + p.m_aggregation_window_dimension -
                        1, local_group_size[1]);

    int local_size1 = local_h1 * local_w1;

    i = 0;
    // lvl1_kernel.setArg(i++, 2);
    lvl1_kernel.setArg(i++, *md_lvl1_l);
    lvl1_kernel.setArg(i++, *md_lvl1_r);
    lvl1_kernel.setArg(i++, *md_lvl2_res_l);
    lvl1_kernel.setArg(i++, *md_lvl2_res_r);
    lvl1_kernel.setArg(i++, *md_lvl1_res_l);
    lvl1_kernel.setArg(i++, *md_lvl1_res_r);
    lvl1_kernel.setArg(i++, local_size1, NULL);
    lvl1_kernel.setArg(i++, local_size1, NULL);
    lvl1_kernel.setArg(i++, local_w1);
    lvl1_kernel.setArg(i++, local_h1);


    /* local memory buffer */
    int local_w = p.roundUp(local_group_size[0] + p.m_aggregation_window_dimension -
                        1 + p.m_maximum_disparity, local_group_size[0]);

    int local_h = p.roundUp(local_group_size[1] + p.m_aggregation_window_dimension -
                        1, local_group_size[1]);

    int local_size = local_h * local_w;

    i = 0;
    // kernel.setArg(i++, 1);
    kernel.setArg(i++, *md_input_l_2d[cur_buf]);
    kernel.setArg(i++, *md_input_r_2d[cur_buf]);
    kernel.setArg(i++, *md_lvl1_res_l);
    kernel.setArg(i++, *md_lvl1_res_r);
    kernel.setArg(i++, *md_result_l_2d[cur_buf]);
    kernel.setArg(i++, *md_result_r_2d[cur_buf]);
    kernel.setArg(i++, local_size, NULL);
    kernel.setArg(i++, local_size, NULL);
    kernel.setArg(i++, local_w);
    kernel.setArg(i++, local_h);

}

void PyramidAlgorithm::prepare() {

    /* First, create the downscaled version of the input */
    resizer();

    /* Now initialize gpu buffers */
    createBuffers();

    /* Now copy over the input and the scaled inputs */
    transferInput();
}

void PyramidAlgorithm::postpare() {

    if(normalize_kernel) {
        normalize_kernel->setBuffer(md_result_l_2d[cur_buf]);
        normalize_kernel->launch();
        normalize_kernel->setBuffer(md_result_r_2d[cur_buf]);
        normalize_kernel->launch();
    }

    p.copyDeviceToHost(*md_result_l_2d[cur_buf], p.m_result_l);
    p.copyDeviceToHost(*md_result_r_2d[cur_buf], p.m_result_r);
    p.m_cl_cqueue.finish();
    cur_buf = !cur_buf;
}

void PyramidAlgorithm::launch() {

    prepare();
    setArgs();

    DBG( "Launching pyramid3");
    p.m_cl_cqueue.enqueueNDRangeKernel(lvl3_kernel,
                                       launch_offset,
                                       global_group_size_lvl3,
                                       local_group_size);
    if(crosscheck_kernel) {
        crosscheck_kernel->setBuffers(md_lvl3_res_l, md_lvl3_res_r);
        crosscheck_kernel->launch();
    }

    DBG( "Launching pyramid2");
    p.m_cl_cqueue.enqueueNDRangeKernel(lvl2_kernel,
                                       launch_offset,
                                       global_group_size_lvl2,
                                       local_group_size);
    if(crosscheck_kernel) {
        crosscheck_kernel->setBuffers(md_lvl2_res_l, md_lvl2_res_r);
        crosscheck_kernel->launch();
    }


    DBG( "Launching pyramid1");
    p.m_cl_cqueue.enqueueNDRangeKernel(lvl1_kernel,
                                       launch_offset,
                                       global_group_size_lvl1,
                                       local_group_size);
    if(crosscheck_kernel) {
        crosscheck_kernel->setBuffers(md_lvl1_res_l, md_lvl1_res_r);
        crosscheck_kernel->launch();
    }

    DBG( "Launching pyramid0");
    p.m_cl_cqueue.enqueueNDRangeKernel(kernel,
                                       launch_offset,
                                       global_group_size,
                                       local_group_size);
    if(crosscheck_kernel) {
        crosscheck_kernel->setBuffers(md_result_l_2d[cur_buf],
                                      md_result_r_2d[cur_buf]);
        crosscheck_kernel->launch();
    }


    postpare();
}


/******
 *
 * CPU version of the Block Matching algorithm
 *
 ******/
CPUAlgorithm::CPUAlgorithm(OpenCL &parent)
    : KernelUnit(parent, NULL) {

    costs = (uchar*) malloc(p.m_size * p.m_maximum_disparity);
    //aggregated_costs = (uchar*) malloc(p.m_size * p.m_maximum_disparity);

}

void CPUAlgorithm::calculateCosts() {
    DBG("CPUAlgorithm::calculateCosts");
    for(int y = 0; y < p.m_height; y++) {
        for(int x = 0; x < p.m_width - p.m_maximum_disparity; x++) {
            for(int d = 0; d < p.m_maximum_disparity; d++) {
                int diff = (int)p.m_input_r[y * p.m_width + x] - (int)p.m_input_l[y * p.m_width + x + d];
                costs[(p.m_size * d) + (y * p.m_width + x)] = abs(diff);
            }
        }
    }
}

void CPUAlgorithm::aggregateCosts() {

    DBG("CPUAlgorithm::aggregateCosts");

    int radius = p.m_aggregation_window_dimension/2;
    for(int y = radius; y < p.m_height - radius; y++) {
        for(int x = radius; x < p.m_width - radius; x++) {
            uint best_sum = -1;
            int best_d = 0;
            for(int d = 1; d < p.m_maximum_disparity; d++) {
                uint sum = 0;
                if(y + radius < p.m_height && x + d < p.m_width) {
                    for(int a = y - radius; a <= y + radius; a++) {
                        for(int b = x - radius; b <= x + radius; b++) {
                            sum += costs[(p.m_size * d) + (a * p.m_width + b)];
                        }
                    }
                    if(sum < best_sum) {
                        best_sum = sum;
                        best_d = d;
                    }
                    //aggregated_costs[(p.m_size * d) + (y * p.m_width + x)] = sum;
                }
            }
            p.m_result_l[y * p.m_width + x] = best_d;
        }
    }
}

void CPUAlgorithm::disparitySelection() {

    DBG("CPUAlgorithm::disparitySelection");

    // int radius = p.m_aggregation_window_dimension/2;
    // for(int y = radius; y < p.m_width - radius; y++) {
    //     for(int x = radius; x < p.m_height - radius; x++) {
    //         uint best_match = -1;
    //         int best_disparity = 0;
    //         for(int d = 0; d < p.m_maximum_disparity; d++) {
    //             if(aggregated_costs[p.m_size * d + y * p.m_width + x] < best_match) {
    //                 best_match = aggregated_costs[(p.m_size * d) + (y * p.m_width + x)];
    //                 best_disparity = d;
    //             }
    //         }
    //         p.m_result_l[y * p.m_width + x] = best_disparity * 4;
    //     }
    // }
}

void CPUAlgorithm::launch() {
    calculateCosts();
    aggregateCosts();
    disparitySelection();
}



/******
 *
 * OpenCV CPU version of the Block Matching algorithm
 *
 ******/
OpenCVCPUAlgorithm::OpenCVCPUAlgorithm(OpenCL &parent)
    : KernelUnit(parent, NULL) {

}

void OpenCVCPUAlgorithm::launch() {

    DBG("OpenCVCPUAlgorithm");

    cv::StereoBM bm(cv::StereoBM::BASIC_PRESET, 16*5, 21); //p.m_maximum_disparity, p.m_aggregation_window_dimension);

    left.create(p.m_height, p.m_width, CV_8UC1);
    right.create(p.m_height, p.m_width, CV_8UC1);
    result.create(p.m_height, p.m_width, CV_16S);

    left.data = p.m_input_l;
    right.data = p.m_input_r;

    bm(left, right, result, CV_16S);

    double minVal, maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal );
    DBG_OP("Minval: " << minVal << "Maxval: " << maxVal);

    cv::Mat result2 = cv::Mat(left.size(), CV_8UC1);

    result.convertTo( result2, CV_8UC1, 255/(maxVal - minVal));


    cv::namedWindow("test");
    cv::imshow("test", result2);

}


/******
 *
 * OpenCV GPU version of the Block Matching algorithm
 *
 ******/
OpenCVAlgorithm::OpenCVAlgorithm(OpenCL &parent)
    : KernelUnit(parent, NULL) {
}

void OpenCVAlgorithm::launch() {

    DBG("OpenCVAlgorithm");


    bm.ndisp = p.m_maximum_disparity;
    bm.winSize = 9;

    left.create(p.m_height, p.m_width, CV_8UC1);
    right.create(p.m_height, p.m_width, CV_8UC1);

    left.data = p.m_input_l;
    right.data = p.m_input_r;

    d_left.upload(left);
    d_right.upload(right);

    cv::Mat disp(left.size(), CV_8U);
    disp.data = p.m_result_l;
    cv::gpu::GpuMat d_disp(left.size(), CV_8U);

    bm(d_left, d_right, d_disp);

    d_disp.download(disp);

}


/******
 *
 * Birtchfield & Tomasi sampling-insensitive matching algorithm
 *
 ******/
BirchfieldTomasiAlgorithm::BirchfieldTomasiAlgorithm(OpenCL &parent)
    : KernelUnit(parent, "data/depthestimation/kernels/birch_tomasi.cl") {


    std::stringstream compile_options("");
    compile_options << " -DMAX_DISP="      << p.m_maximum_disparity;
    compile_options << " -DAGGR_DIM="      << p.m_aggregation_window_dimension;

    compileSource(compile_options.str());

    kernel = cl::Kernel(program, "birchfieldTomasi");

    int dim = 16;
    int lx = dim;
    int ly = dim;
    local_group_size = cl::NDRange(lx,ly);

    int gx = p.roundUp(p.m_width, lx);
    int gy = p.roundUp(p.m_height, ly);
    global_group_size = cl::NDRange(gx,gy);

    DBG("BirchfieldTomasiAlgorithm object created");
}

void BirchfieldTomasiAlgorithm::prepare() {

    cl::ImageFormat format(CL_R, CL_UNSIGNED_INT8);

    md_input_l.reset( new cl::Image2D(p.m_cl_context, CL_MEM_READ_ONLY, format, p.m_width, p.m_height));
    md_input_r.reset( new cl::Image2D(p.m_cl_context, CL_MEM_READ_ONLY, format, p.m_width, p.m_height));
    md_result_l.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, p.m_width, p.m_height));
    md_result_r.reset(new cl::Image2D(p.m_cl_context, CL_MEM_READ_WRITE, format, p.m_width, p.m_height));

    p.copyHostToDevice(p.m_input_l, *md_input_l);
    p.copyHostToDevice(p.m_input_r, *md_input_r);

    DBG("Prepare complete");
}

void BirchfieldTomasiAlgorithm::setArgs() {

    int i = 0;
    kernel.setArg(i++, *md_input_l);
    kernel.setArg(i++, *md_input_r);
    kernel.setArg(i++, *md_result_l);
    kernel.setArg(i++, *md_result_r);

    DBG("setArgs complete");
}

void BirchfieldTomasiAlgorithm::postpare() {

    p.copyDeviceToHost(*md_result_l, p.m_result_l);
    p.m_cl_cqueue.finish();
    DBG("postpare complete");
}


} // namespace Depth
