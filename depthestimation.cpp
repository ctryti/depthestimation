#include "depthestimation.hpp"

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <logging.h>
#include <rrdb.h>

#include "oclUtils.h"

namespace DepthEstimation {

DepthEstimator::DepthEstimator()
    : m_timing_db_ptr(new NorthLight::RRDB<NorthLight::TimeSpec>(100)) {

    m_width = -1;
    m_height = -1;
    m_maximum_disparity = 255;
    m_aggregation_window_dimension = 9;

    m_options = 0;
    frame_num = 0;

    // NorthLight::logSetFormat(NorthLight::LOG_FMT_DATETIME |
    //                          NorthLight::LOG_FMT_LVL);

    // NorthLight::logSetFormat(NorthLight::LOG_FMT_NONE);
    // NorthLight::logSetLevel(NorthLight::LOG_LVL_INFO);
}

DepthEstimator::~DepthEstimator() {}

/*
 * Rounds up value to nearest multiple of 'multiple'
 */

/*
 * Set dimensions of input images
 */
void DepthEstimator::setSize(int width, int height) {
    m_height = height;
    m_width = width;
    m_size = width * height;
}

/*
 * Max disparity can be up to uint8_t MAX_VALUE
 */
void DepthEstimator::setMaximumDisparity(int value) {
    m_maximum_disparity = value;
    if(m_maximum_disparity > 255)
        m_maximum_disparity = 255;
}

/*
 * Aggregation dimensions should be an odd value
 */
void DepthEstimator::setAggregationWindowDimensions(int value) {
    m_aggregation_window_dimension = value % 2 == 0 ? value + 1 : value;
}

/*
 * Change the range of disparity values from [0, MAX_DISP] to [0,
 * 255]. Makes disparities more visible when operating with low
 * MAX_DISP.
 */
void DepthEstimator::setNormalize(bool value) {
    if(value)
        m_options |= USE_NORMALIZE_KERNEL;
}

void DepthEstimator::setOcclusionFill(bool value) {
    if(value)
        m_options |= USE_OCCLUSION_FILL;
}

/*
 * Select which disparity calculation kernel to use
 */
void DepthEstimator::setAlgorithm(std::string name) {
    boost::to_upper(name);
    if(name == "BM")
        m_algorithm = BM;
    else if(name == "CPU")
        m_algorithm = CPU;
    else if(name == "OPENCV")
        m_algorithm = OPENCV;
    else if(name == "OPENCV_GPU")
        m_algorithm = OPENCV_GPU;
    else if(name == "BM_NO_DEF")
        m_algorithm = BM_NO_DEF;
    else if(name == "BMLMEM")
        m_algorithm = BMLMEM;
    else if(name == "BMLMEM_NO_DEF")
        m_algorithm = BMLMEM_NO_DEF;
    else if(name == "DUAL")
        m_algorithm = DUAL;
    else if(name == "DUAL_NO_DEF")
        m_algorithm = DUAL_NO_DEF;
    else if(name == "TEXTURE")
        m_algorithm = TEXTURE;
    else if(name == "PYRAMID")
        m_algorithm = PYRAMID;
    else if(name == "BT")
        m_algorithm = BT;
    else {
        LOG_OP(LOG_LVL_DEBUG, "Invalid algorithm: " << name);
        throw 1; // TODO!
    }
}

/*
 * Select which cost function to use
 */
void DepthEstimator::setCostFunction(std::string name) {
    m_cost_function = name;
}

/*
 * Launch cross-checking kernel? Only works with algorithms that
 * produce both left and right disparity maps
 */
void DepthEstimator::setCrossCheck(bool value) {
    if(value)
        m_options |= USE_CROSS_CHECK_KERNEL;
}

/*
 * Launch diffing kernel? Only useful for video streams
 */
void DepthEstimator::useDiffingKernel(bool value) {
    if(value)
        m_options |= USE_DIFFING_KERNEL;
}

void DepthEstimator::useMinMax(bool value) {
    if(value)
        m_options |= USE_MIN_MAX;
}

} // namespace DepthEstimation
