#ifndef LIBDEPTH_H_
#define LIBDEPTH_H_

#include "types.hpp"
//#include "depthestimation_opencl.hpp"

#include <map>
#include <string>
#include <sys/time.h>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <rrdb.h>
#include <timeutils.h>

namespace DepthEstimation {

class DepthEstimator {

public:

    DepthEstimator();
    virtual ~DepthEstimator();

    void setSize(int width, int height);
    void setMaximumDisparity(int value);
    void setAggregationWindowDimensions(int value);
    void setNormalize(bool normalize);
    void setAlgorithm(std::string name);
    void setCostFunction(std::string name);
    void setCrossCheck(bool value);
    void useDiffingKernel(bool value);
    void useTextureMemory(bool value);
    void setOcclusionFill(bool value);
    void useMinMax(bool value);

    virtual void estimate(uint8_t *input_l, uint8_t *input_r, uint8_t *result_l, uint8_t *result_r) = 0;

    int getWidth() { return m_width; };
    int getHeight() { return m_height; };
    int getSize() { return m_size; };
    int getOptions() { return m_options; };

    uint8_t* m_input_l;
    uint8_t* m_input_r;
    uint8_t* m_result_l;
    uint8_t* m_result_r;

    int m_options;

    int frame_num;

    int m_width;
    int m_height;
    int m_size;

    int m_maximum_disparity;
    int m_aggregation_window_dimension;

    std::string m_cost_function;

    Algorithm m_algorithm;

    NorthLight::RRDB<NorthLight::TimeSpec>::PtrType m_timing_db_ptr;


};     // class
};     // namespace
#endif // include guard
